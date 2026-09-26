"""Core implementation excerpts accompanying Boundary-FidelityBench."""
from .evidence import radius_innovation, fit_interval_envelope, residual_features
from .model import FullSequenceRayHead
from .posterior import factorized_posterior
from .losses import interval_factorized_nll, pairwise_ranking_loss
from .types import RayResponseBatch, IntervalTrainingTargets

__all__ = [
    "radius_innovation", "fit_interval_envelope", "residual_features",
    "FullSequenceRayHead", "factorized_posterior", "interval_factorized_nll",
    "pairwise_ranking_loss", "RayResponseBatch", "IntervalTrainingTargets",
]
