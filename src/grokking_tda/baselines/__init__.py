"""Non-topological baselines, registered on import so topology is never reported alone."""

from grokking_tda.baselines import fourier, lid, weight_norm  # noqa: F401

__all__ = ["fourier", "lid", "weight_norm"]
