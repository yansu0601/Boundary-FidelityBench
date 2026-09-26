"""LCR residuals and SAE envelope/excess features."""

from __future__ import annotations

import torch
from torch import Tensor
from .types import RayResponseBatch, RadiusInnovationBatch, IntervalTrainingTargets

# Only the numerical epsilon used below is retained from the experiment recipe.
RECIPE = {"excess_epsilon": 1e-8}


def radius_innovation(batch: RayResponseBatch) -> RadiusInnovationBatch:
    """Compute unnormalized linear-continuation innovation on irregular radii."""

    response = batch.response
    radii = batch.realized_radii
    valid = batch.valid_prefix_mask
    innovation = torch.zeros_like(response)
    extrapolated = torch.zeros_like(response)
    ratios = torch.zeros_like(radii)
    innovation_valid = torch.zeros_like(valid)
    if response.shape[2] < 2:
        return RadiusInnovationBatch(innovation, extrapolated, ratios, innovation_valid)
    previous_previous_radii = torch.cat((torch.zeros_like(radii[..., :1]), radii[..., :-2]), dim=-1)
    denominator = radii[..., :-1] - previous_previous_radii
    numerator = radii[..., 1:] - radii[..., :-1]
    current_valid = valid[..., 1:] & valid[..., :-1] & (denominator > 0) & (numerator > 0)
    ratio = torch.where(
        current_valid,
        numerator / denominator.clamp_min(torch.finfo(radii.dtype).tiny),
        0,
    )
    previous_previous_response = torch.cat(
        (
            batch.center[:, None, None, :].expand(-1, response.shape[1], 1, -1),
            response[..., :-2, :],
        ),
        dim=2,
    )
    estimate = (
        (1 + ratio[..., None]) * response[..., :-1, :]
        - ratio[..., None] * previous_previous_response
    )
    innovation[..., 1:, :] = torch.where(
        current_valid[..., None], response[..., 1:, :] - estimate, 0
    )
    extrapolated[..., 1:, :] = torch.where(current_valid[..., None], estimate, 0)
    ratios[..., 1:] = ratio
    innovation_valid[..., 1:] = current_valid
    return RadiusInnovationBatch(innovation, extrapolated, ratios, innovation_valid)


def fit_interval_envelope(
    response: RayResponseBatch,
    target: IntervalTrainingTargets,
    response_available: Tensor,
    quantile: float,
) -> tuple[Tensor, dict]:
    """Fit q-innovation using only train labels that prove a point is pre-boundary."""
    if not 0 < quantile < 1:
        raise ValueError("envelope quantile must lie in (0,1)")
    n, rays, scales, width = response.response.shape
    if rays != 1 or response_available.shape != (n,) or response_available.dtype != torch.bool:
        raise ValueError("the B2 bridge requires one ray per row and bool availability")
    if target.valid_prefix_mask.shape != (n, rays, scales) or not torch.equal(
        target.valid_prefix_mask, response.valid_prefix_mask
    ):
        raise ValueError("target and response prefix support differ")
    innovation = radius_innovation(response)
    eligible = target.train_eligible & response_available[:, None]
    event = eligible & ~target.is_censored
    censored = eligible & target.is_censored
    lower = target.interval[..., 0]
    provably_before = response.realized_radii < lower[..., None]
    selected = innovation.innovation_valid_mask & (
        censored[..., None] | (event[..., None] & provably_before)
    )
    envelope = response.response.new_zeros(scales, width)
    counts = []
    for scale in range(scales):
        values = innovation.innovation[:, :, scale][selected[:, :, scale]].abs()
        counts.append(int(values.shape[0]))
        if scale and not values.shape[0]:
            raise ValueError("every innovation-valid scale needs train-only envelope support")
        if values.shape[0]:
            envelope[scale] = torch.quantile(values, quantile, dim=0)
    if not torch.isfinite(envelope).all() or torch.any(envelope < 0):
        raise ValueError("fitted envelope must be finite and nonnegative")
    return envelope, {
        "quantile": quantile,
        "sample_counts_per_scale": counts,
        "source_policy": (
            "eligible_censored_all_valid_innovations_or_eligible_event_radius_strictly_below_dL"
        ),
        "uses_validation": False,
        "uses_point_signature_truth": False,
        "model_error_envelope": "zero_environment_only_evaluator",
    }


def residual_features(
    response: RayResponseBatch, envelope: Tensor, component: str
) -> Tensor:
    """Build the frozen full features, or zero them for geometry-only simplification."""
    scales, width = response.response.shape[2:]
    if envelope.shape != (scales, width) or component not in {"full", "geometry_only"}:
        raise ValueError("envelope shape or residual component is invalid")
    if not torch.isfinite(envelope).all() or torch.any(envelope < 0):
        raise ValueError("envelope must be finite and nonnegative")
    innovation = radius_innovation(response)
    expanded = envelope.to(response.response)[None, None].expand_as(response.response)
    magnitude = torch.linalg.vector_norm(innovation.innovation, dim=-1)
    envelope_magnitude = torch.linalg.vector_norm(expanded, dim=-1)
    epsilon = float(RECIPE["excess_epsilon"])
    excess = torch.log((magnitude + epsilon) / (envelope_magnitude + epsilon))
    previous_radius = torch.cat(
        (torch.zeros_like(response.realized_radii[..., :1]), response.realized_radii[..., :-1]),
        dim=-1,
    )
    gap = response.realized_radii - previous_radius
    features = torch.cat(
        (innovation.innovation, excess[..., None], gap[..., None], expanded), dim=-1
    )
    features = torch.where(
        innovation.innovation_valid_mask[..., None], features, torch.zeros_like(features)
    )
    return features if component == "full" else torch.zeros_like(features)
