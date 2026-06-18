"""Turn a representation matrix into a point cloud for persistent homology.

The construction choices (normalisation, metric, subsampling) materially affect the
persistence diagram, so they are explicit config — never silent defaults. The metric
itself is applied later by ripser; here we only normalise and (optionally) subsample.
"""

from __future__ import annotations

import numpy as np

from grokking_tda.config.schema import PointCloudCfg


def _maxmin_landmarks(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Greedy farthest-point (maxmin) landmark indices — good coverage for H1/H2."""
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(x.shape[0]))]
    min_dist = np.linalg.norm(x - x[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(min_dist.argmax())
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(x - x[nxt], axis=1))
    return np.sort(np.asarray(chosen))


def build_point_cloud(matrix: np.ndarray, cfg: PointCloudCfg, *, seed: int = 0) -> np.ndarray:
    """Normalise (and optionally subsample) a ``(n, d)`` matrix into a point cloud."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {x.shape}")

    if getattr(cfg, "drop_first", False):  # residue 0: outside the multiplicative group
        x = x[1:]

    if cfg.normalize == "none":
        pass
    elif cfg.normalize == "center":
        x = x - x.mean(axis=0, keepdims=True)
    elif cfg.normalize == "unit_norm":
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    elif cfg.normalize == "standardize":
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
    else:
        raise ValueError(f"unknown normalize {cfg.normalize!r}")

    if cfg.max_points and x.shape[0] > cfg.max_points:
        method = getattr(cfg, "subsample", "random")
        if method == "random":
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(x.shape[0], size=cfg.max_points, replace=False))
        elif method == "maxmin":
            idx = _maxmin_landmarks(x, cfg.max_points, seed)
        else:
            raise ValueError(f"unknown subsample {method!r}; choices: random, maxmin")
        x = x[idx]
    return x
