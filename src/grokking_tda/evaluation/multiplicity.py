"""False-discovery control across the robustness grid; the conditions share a null bank."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values, q: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """``(rejected, adjusted)`` at FDR ``q``; NaN p-values carry through as non-rejected."""
    return _step_up(p_values, q=q, penalty=1.0)


def benjamini_yekutieli(p_values, q: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """The same, every threshold divided by the harmonic number: valid under any dependence."""
    n = int(np.isfinite(np.asarray(p_values, dtype=float)).sum())
    penalty = float(np.sum(1.0 / np.arange(1, n + 1))) if n else 1.0
    return _step_up(p_values, q=q, penalty=penalty)


def _step_up(p_values, *, q: float, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)
    adjusted = np.full(p.shape, np.nan)

    finite = np.isfinite(p)
    if not finite.any():
        return rejected, adjusted

    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    n = ranked.size

    # Step-up: the largest k with p_(k) <= k/n * q, then reject everything up to it.
    thresholds = (np.arange(1, n + 1) / (n * penalty)) * q
    below = np.where(ranked <= thresholds)[0]
    cut = below.max() if below.size else -1

    reject_sorted = np.zeros(n, dtype=bool)
    if cut >= 0:
        reject_sorted[: cut + 1] = True

    # the running minimum of n/k * p_(k) from the top down, which enforces monotonicity
    adjusted_sorted = np.minimum.accumulate(
        (n * penalty / np.arange(n, 0, -1)) * ranked[::-1]
    )[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    out_reject = np.zeros(n, dtype=bool)
    out_adjusted = np.empty(n)
    out_reject[order] = reject_sorted
    out_adjusted[order] = adjusted_sorted

    rejected[finite] = out_reject
    adjusted[finite] = out_adjusted
    return rejected, adjusted
