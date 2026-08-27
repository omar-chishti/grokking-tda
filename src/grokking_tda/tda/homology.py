"""Vietoris-Rips persistent homology via ripser.

Returns one persistence diagram per homology dimension (``{0: dgm0, 1: dgm1, ...}``),
each an array of ``[birth, death]`` pairs (death may be ``inf`` for essential classes).
"""

from __future__ import annotations

import warnings

import numpy as np
from ripser import ripser

from grokking_tda.config.schema import HomologyCfg


def compute_persistence(
    points: np.ndarray, cfg: HomologyCfg, metric: str = "euclidean"
) -> dict[int, np.ndarray]:
    """Compute persistence diagrams for ``H_0 .. H_maxdim`` on a point cloud."""
    points = np.asarray(points, dtype=np.float64)
    # A diverged run's weights are NaN, and divergence is a result to record rather
    # than a crash: ripser raises from deep inside sklearn's distance code, which
    # otherwise takes down the whole run's analysis and its figures with it.
    if not np.isfinite(points).all():
        return {dim: np.empty((0, 2)) for dim in range(cfg.maxdim + 1)}

    kwargs: dict = {"maxdim": cfg.maxdim, "coeff": cfg.coeff, "metric": metric}
    if cfg.thresh is not None and cfg.thresh > 0:
        kwargs["thresh"] = float(cfg.thresh)
    with warnings.catch_warnings():
        # Embedding clouds are intentionally (n_points, n_dims) with d > n; ripser's
        # "more columns than rows" transpose hint is a false positive here.
        warnings.filterwarnings("ignore", message=".*more columns than rows.*")
        diagrams = ripser(points, **kwargs)["dgms"]
    return {dim: diagrams[dim] for dim in range(len(diagrams))}
