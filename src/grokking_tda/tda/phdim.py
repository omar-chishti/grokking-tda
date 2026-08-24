"""Persistent-homology dimension of a training trajectory (Birdal et al., NeurIPS 2021).

For a point set ``W`` and exponent ``alpha``, the weighted lifetime sum

    E_alpha(W) = sum over the minimum spanning tree's edges of (edge length)^alpha

grows with sample size ``n`` as ``E_alpha ~ n^((d - alpha) / d)``. Regressing
``log E_alpha`` on ``log n`` therefore recovers an intrinsic dimension ``d`` — the
PH-dimension — which Birdal et al. show tracks generalisation error over training.

The H0 persistence of a Vietoris-Rips filtration is exactly the minimum spanning
tree's edge lengths, so this reuses the same ripser backend as everything else. Run
it over a sliding window of iterates to get a dimension per training step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grokking_tda.config.schema import HomologyCfg
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.summaries import finite_lifetimes


def alpha_weighted_lifetime_sum(points: np.ndarray, alpha: float = 1.0) -> float:
    """``E_alpha`` — the alpha-weighted total H0 persistence of a point set."""
    diagram = compute_persistence(np.asarray(points, dtype=np.float64), HomologyCfg(maxdim=0))
    lifetimes = finite_lifetimes(diagram.get(0))
    return float((lifetimes**alpha).sum()) if lifetimes.size else 0.0


def ph_dimension(
    points: np.ndarray,
    *,
    alpha: float = 1.0,
    n_subsets: int = 8,
    min_points: int = 40,
    seed: int = 0,
) -> float:
    """PH-dimension of ``points``, from the growth of ``E_alpha`` with sample size.

    Returns ``nan`` when the window is too small to fit a slope, or when the fitted
    slope leaves the admissible range (a degenerate window rather than a dimension).
    """
    x = np.asarray(points, dtype=np.float64)
    n_total = x.shape[0]
    if n_total < min_points * 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    sizes = np.unique(np.linspace(min_points, n_total, n_subsets, dtype=int))
    logs_n, logs_e = [], []
    for size in sizes:
        idx = rng.choice(n_total, size=size, replace=False)
        energy = alpha_weighted_lifetime_sum(x[idx], alpha)
        if energy > 0:
            logs_n.append(np.log(size))
            logs_e.append(np.log(energy))
    if len(logs_n) < 3:
        return float("nan")

    slope = float(np.polyfit(logs_n, logs_e, 1)[0])
    if not 0.0 < slope < 1.0:
        return float("nan")
    return alpha / (1.0 - slope)


def ph_dimension_over_training(
    steps: np.ndarray,
    points: np.ndarray,
    *,
    window: int = 200,
    stride: int = 50,
    alpha: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """PH-dimension in a sliding window along the trajectory.

    Each row is labelled with the step at the window's right edge, so the series is
    comparable against the grokking step without look-ahead.
    """
    rows = []
    for end in range(window, len(steps) + 1, stride):
        rows.append(
            {
                "step": int(steps[end - 1]),
                "ph_dim": ph_dimension(points[end - window : end], alpha=alpha, seed=seed),
            }
        )
    return pd.DataFrame(rows, columns=["step", "ph_dim"])
