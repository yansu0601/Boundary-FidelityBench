"""FBR: masked location--cause probabilities and no-crossing mass."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from .types import EventDistancePosterior


def expand_valid_cause_mask(mask: Tensor, shape: tuple[int, int, int, int]) -> Tensor:
    """Expand common compact cause masks to ``[B,R,M,C]``."""

    batch, rays, _, causes = shape
    if mask.dtype != torch.bool or mask.shape[-1] != causes:
        raise ValueError("valid_cause_mask must be bool with trailing cause dimension C")
    if mask.ndim == 1:
        mask = mask.reshape(1, 1, 1, causes)
    elif mask.ndim == 2:
        mask = mask.reshape(mask.shape[0], 1, 1, causes)
    elif mask.ndim == 3:
        if mask.shape[:2] != (batch, rays):
            raise ValueError("three-dimensional cause mask must have shape [B,R,C]")
        mask = mask.unsqueeze(-2)
    elif mask.ndim != 4:
        raise ValueError("valid_cause_mask must have 1 to 4 dimensions")
    try:
        return torch.broadcast_to(mask, shape)
    except RuntimeError as error:
        raise ValueError("valid_cause_mask is not broadcastable to [B,R,M,C]") from error


def _masked_log_softmax(logits: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
    flat_logits = logits.flatten(start_dim=2)
    flat_support = support.flatten(start_dim=2)
    has_support = flat_support.any(dim=-1)
    safe_support = flat_support.clone()
    safe_support[..., 0] |= ~has_support
    safe_logits = flat_logits.masked_fill(~safe_support, -torch.inf)
    log_probs = F.log_softmax(safe_logits, dim=-1)
    probs = torch.where(flat_support, log_probs.exp(), 0)
    log_probs = torch.where(flat_support, log_probs, -torch.inf)
    return log_probs.reshape_as(logits), probs.reshape_as(logits)


def _median_interval(
    interval_probs: Tensor,
    tail: Tensor,
    radii: Tensor,
    valid_prefix_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    bin_probs = interval_probs.sum(dim=-1)
    cumulative = bin_probs.cumsum(dim=-1)
    reaches = cumulative >= 0.5
    reaches_any = reaches.any(dim=-1)
    first = reaches.to(torch.int64).argmax(dim=-1)
    gathered_upper = radii.gather(-1, first[..., None]).squeeze(-1)
    previous = torch.cat((torch.zeros_like(radii[..., :1]), radii[..., :-1]), dim=-1)
    gathered_lower = previous.gather(-1, first[..., None]).squeeze(-1)
    last_index = valid_prefix_mask.to(torch.int64).sum(dim=-1).sub(1).clamp_min(0)
    last_radius = radii.gather(-1, last_index[..., None]).squeeze(-1)
    lower = torch.where(reaches_any, gathered_lower, last_radius)
    upper = torch.where(reaches_any, gathered_upper, torch.full_like(gathered_upper, torch.inf))
    return torch.stack((lower, upper), dim=-1), ~reaches_any


def factorized_posterior(
    event_logit: Tensor,
    conditional_location_logits: Tensor,
    valid_prefix_mask: Tensor,
    valid_cause_mask: Tensor,
    realized_radii: Tensor,
) -> EventDistancePosterior:
    """Build the detection--localization posterior on legal bin-cause support."""

    if conditional_location_logits.ndim != 4:
        raise ValueError("conditional_location_logits must have shape [B,R,M,C]")
    shape = tuple(conditional_location_logits.shape)
    batch, rays, scales, _ = shape
    if event_logit.shape != (batch, rays):
        raise ValueError("event_logit must have shape [B,R]")
    if valid_prefix_mask.shape != (batch, rays, scales) or valid_prefix_mask.dtype != torch.bool:
        raise ValueError("valid_prefix_mask must be bool [B,R,M]")
    if realized_radii.shape != (batch, rays, scales):
        raise ValueError("realized_radii must have shape [B,R,M]")
    expanded = expand_valid_cause_mask(valid_cause_mask, shape)
    support = valid_prefix_mask[..., None] & expanded
    _, conditional = _masked_log_softmax(conditional_location_logits, support)
    has_support = support.flatten(start_dim=2).any(dim=-1)
    event_prob = torch.where(has_support, torch.sigmoid(event_logit), 0)
    interval = event_prob[..., None, None] * conditional
    tail = 1 - event_prob
    risk = interval.sum(dim=-1).cumsum(dim=-1)
    median, median_is_tail = _median_interval(interval, tail, realized_radii, valid_prefix_mask)
    masses = torch.cat((interval.flatten(start_dim=2), tail[..., None]), dim=-1)
    entropy = -(
        torch.where(
            masses > 0,
            masses * masses.clamp_min(torch.finfo(masses.dtype).tiny).log(),
            0,
        )
    ).sum(dim=-1)
    return EventDistancePosterior(
        boundary_within_range_prob=event_prob,
        conditional_interval_cause_probs=conditional,
        interval_cause_probs=interval,
        tail_probability=tail,
        risk_within_radius=risk,
        median_interval=median,
        median_is_tail=median_is_tail,
        posterior_entropy=entropy,
        valid_prefix_mask=valid_prefix_mask,
        valid_cause_mask=expanded,
        valid_event_support=support,
        all_mask_count=int((~has_support).sum().item()),
    )


def _reduce(loss: Tensor, valid: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return torch.where(valid, loss, 0)
    selected = loss[valid]
    if selected.numel() == 0:
        raise ValueError("loss batch contains no valid rays")
    if reduction == "mean":
        return selected.mean()
    if reduction == "sum":
        return selected.sum()
    raise ValueError("reduction must be none, mean, or sum")
