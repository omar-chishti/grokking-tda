"""Vietoris-Rips persistent homology via ripser: one diagram per dimension."""

from __future__ import annotations

import warnings

import numpy as np
from ripser import ripser

from grokking_tda.config.schema import HomologyCfg


def compute_persistence(
    points: np.ndarray, cfg: HomologyCfg, metric: str = "euclidean"
) -> dict[int, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    # A diverged run's weights are NaN, and divergence is a result to record, not a crash
    if not np.isfinite(points).all():
        return {dim: np.empty((0, 2)) for dim in range(cfg.maxdim + 1)}

    kwargs: dict = {"maxdim": cfg.maxdim, "coeff": cfg.coeff, "metric": metric}
    if cfg.thresh is not None and cfg.thresh > 0:
        kwargs["thresh"] = float(cfg.thresh)
    with warnings.catch_warnings():
        # these clouds have d >= n by construction, and ripser reads both that and the square
        # case as a distance matrix passed by mistake
        warnings.filterwarnings("ignore", message=".*more columns than rows.*")
        warnings.filterwarnings("ignore", message=".*input matrix is square.*")
        diagrams = ripser(points, **kwargs)["dgms"]
    return {dim: diagrams[dim] for dim in range(len(diagrams))}
