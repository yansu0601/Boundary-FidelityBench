"""REF: raw-response logits plus zero-initialized residual corrections."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SequenceHeadOutput:
    raw_event_logit: Tensor
    raw_location_logits: Tensor
    residual_event_logit: Tensor
    residual_location_logits: Tensor
    event_logit: Tensor
    location_logits: Tensor


class FullSequenceRayHead(nn.Module):
    """An ordered MLP applied independently to every one-sided ray."""

    def __init__(
        self,
        scales: int,
        response_dim: int,
        causes: int,
        hidden_dim: int = 32,
        context_dim: int = 0,
        residual_dim: int | None = None,
    ):
        super().__init__()
        if min(scales, response_dim, causes, hidden_dim) <= 0:
            raise ValueError("head dimensions must be positive")
        if context_dim < 0:
            raise ValueError("context_dim must be nonnegative")
        if residual_dim is not None and residual_dim <= 0:
            raise ValueError("residual_dim must be positive")
        self.scales = scales
        self.response_dim = response_dim
        self.causes = causes
        self.context_dim = context_dim
        self.residual_dim = response_dim if residual_dim is None else residual_dim
        raw_width = context_dim + scales * (response_dim + 2)
        innovation_width = scales * (self.residual_dim + 2)
        output_width = 1 + scales * causes
        self.raw = nn.Sequential(
            nn.Linear(raw_width, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_width),
        )
        self.residual = nn.Sequential(
            nn.Linear(innovation_width, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_width)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        response: Tensor,
        residual_features: Tensor,
        radii: Tensor,
        valid_prefix_mask: Tensor,
        context: Tensor | None = None,
        *,
        residual_enabled: bool = True,
    ) -> SequenceHeadOutput:
        if response.ndim != 4 or residual_features.ndim != 4:
            raise ValueError("response and residual_features must be rank four")
        batch, rays, scales, output_dim = response.shape
        if (scales, output_dim) != (self.scales, self.response_dim):
            raise ValueError("input shape differs from the frozen head shape")
        if radii.shape != (batch, rays, scales) or valid_prefix_mask.shape != radii.shape:
            raise ValueError("radii and valid_prefix_mask must have shape [B,R,M]")
        if residual_features.shape != (batch, rays, scales, self.residual_dim):
            raise ValueError(
                "residual_features must have shape [B,R,M,residual_dim]"
            )
        if context is None:
            context = response.new_empty(batch, 0)
        if context.shape != (batch, self.context_dim):
            raise ValueError("context must have shape [B,context_dim]")
        mask = valid_prefix_mask[..., None]
        raw_sequence = torch.cat(
            (
                torch.where(mask, response, 0),
                torch.where(mask, radii[..., None], 0),
                mask.to(response.dtype),
            ),
            dim=-1,
        ).flatten(start_dim=2)
        raw_input = torch.cat(
            (context[:, None, :].expand(-1, rays, -1), raw_sequence), dim=-1
        )
        innovation_input = torch.cat(
            (
                torch.where(mask, residual_features, 0),
                torch.where(mask, radii[..., None], 0),
                mask.to(response.dtype),
            ),
            dim=-1,
        ).flatten(start_dim=2)
        raw = self.raw(raw_input)
        residual = self.residual(innovation_input) if residual_enabled else torch.zeros_like(raw)
        final = raw + residual
        return SequenceHeadOutput(
            raw_event_logit=raw[..., 0],
            raw_location_logits=raw[..., 1:].reshape(batch, rays, scales, self.causes),
            residual_event_logit=residual[..., 0],
            residual_location_logits=residual[..., 1:].reshape(batch, rays, scales, self.causes),
            event_logit=final[..., 0],
            location_logits=final[..., 1:].reshape(batch, rays, scales, self.causes),
        )

    def parameter_counts(self) -> dict[str, int]:
        return {
            "raw": sum(parameter.numel() for parameter in self.raw.parameters()),
            "residual": sum(parameter.numel() for parameter in self.residual.parameters()),
        }
