"""Turn a representation matrix into a point cloud: normalise, and optionally subsample."""

from __future__ import annotations

import numpy as np

from grokking_tda.config.schema import PointCloudCfg


def _maxmin_landmarks(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Greedy farthest-point landmarks, on squared distances so each step is one matvec."""
    rng = np.random.default_rng(seed)
    sq_norm = np.einsum("ij,ij->i", x, x)
    chosen = [int(rng.integers(x.shape[0]))]
    min_sq = sq_norm - 2.0 * (x @ x[chosen[0]]) + sq_norm[chosen[0]]
    for _ in range(k - 1):
        nxt = int(min_sq.argmax())
        chosen.append(nxt)
        np.minimum(min_sq, sq_norm - 2.0 * (x @ x[nxt]) + sq_norm[nxt], out=min_sq)
    return np.sort(np.asarray(chosen))


def build_point_cloud(matrix: np.ndarray, cfg: PointCloudCfg, *, seed: int = 0) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {x.shape}")

    if cfg.drop_first:  # residue 0 sits outside the multiplicative group (mul/div)
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
        method = cfg.subsample
        if method == "random":
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(x.shape[0], size=cfg.max_points, replace=False))
        elif method == "maxmin":
            idx = _maxmin_landmarks(x, cfg.max_points, seed)
        else:
            raise ValueError(f"unknown subsample {method!r}; choices: random, maxmin")
        x = x[idx]
    return x
