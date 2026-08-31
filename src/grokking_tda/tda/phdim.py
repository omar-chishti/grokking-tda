"""PH-dimension of a training trajectory (Birdal et al., NeurIPS 2021)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grokking_tda.config.schema import HomologyCfg
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.summaries import finite_lifetimes


def alpha_weighted_lifetime_sum(points: np.ndarray, alpha: float = 1.0) -> float:
    diagram = compute_persistence(np.asarray(points, dtype=np.float64), HomologyCfg(maxdim=0))
    lifetimes = finite_lifetimes(diagram.get(0))
    return float((lifetimes**alpha).sum()) if lifetimes.size else 0.0


def ph_dimension_fit(
    points: np.ndarray,
    *,
    alpha: float = 1.0,
    n_subsets: int = 8,
    min_points: int = 40,
    n_draws: int = 1,
    seed: int = 0,
) -> dict[str, float]:
    """The fit, not only the value: ``dim = alpha / (1 - slope)`` is stiff at the ends of its
    range, so the slope comes back beside the dimension. ``n_draws`` averages ``E_alpha`` over
    subsamples per size, as Birdal et al. do; the default of one is the series as measured."""
    x = np.asarray(points, dtype=np.float64)
    n_total = x.shape[0]
    blank = {"ph_dim": float("nan"), "slope": float("nan"), "r2": float("nan"), "n_sizes": 0.0}
    if n_total < min_points * 2:
        return blank

    rng = np.random.default_rng(seed)
    sizes = np.unique(np.linspace(min_points, n_total, n_subsets, dtype=int))
    logs_n, logs_e = [], []
    for size in sizes:
        draws = 1 if size >= n_total else n_draws  # the full set has one subsample
        energies = [
            alpha_weighted_lifetime_sum(x[rng.choice(n_total, size=size, replace=False)], alpha)
            for _ in range(draws)
        ]
        energy = float(np.mean(energies))
        if energy > 0:
            logs_n.append(np.log(size))
            logs_e.append(np.log(energy))
    if len(logs_n) < 3:
        return blank

    slope, intercept = np.polyfit(logs_n, logs_e, 1)
    predicted = intercept + slope * np.asarray(logs_n)
    residual = np.asarray(logs_e) - predicted
    total = np.asarray(logs_e) - np.mean(logs_e)
    r2 = 1.0 - float(residual @ residual) / float(total @ total) if total.any() else float("nan")
    admissible = 0.0 < slope < 1.0
    return {
        "ph_dim": alpha / (1.0 - slope) if admissible else float("nan"),
        "slope": float(slope),
        "r2": r2,
        "n_sizes": float(len(logs_n)),
    }


def ph_dimension(points: np.ndarray, **kwargs) -> float:
    """PH-dimension of ``points``; ``nan`` for a degenerate window or an inadmissible slope."""
    return ph_dimension_fit(points, **kwargs)["ph_dim"]


def ph_dimension_over_training(
    steps: np.ndarray,
    points: np.ndarray,
    *,
    window: int = 200,
    stride: int = 50,
    alpha: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for end in range(window, len(steps) + 1, stride):
        rows.append(
            {
                "step": int(steps[end - 1]),
                "ph_dim": ph_dimension(points[end - window : end], alpha=alpha, seed=seed),
            }
        )
    return pd.DataFrame(rows, columns=["step", "ph_dim"])
