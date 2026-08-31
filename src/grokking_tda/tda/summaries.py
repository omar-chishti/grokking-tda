"""Scalar summaries of a persistence diagram. Essential bars are excluded, as is conventional."""

from __future__ import annotations

import numpy as np


def finite_bars(diagram: np.ndarray | None) -> np.ndarray:
    if diagram is None or diagram.size == 0:
        return np.empty((0, 2))
    return diagram[np.isfinite(diagram[:, 1])]


def finite_lifetimes(diagram: np.ndarray | None) -> np.ndarray:
    bars = finite_bars(diagram)
    return bars[:, 1] - bars[:, 0] if bars.size else np.empty(0)


def total_persistence(diagram: np.ndarray | None) -> float:
    lifetimes = finite_lifetimes(diagram)
    return float(lifetimes.sum()) if lifetimes.size else 0.0


def max_persistence(diagram: np.ndarray | None) -> float:
    lifetimes = finite_lifetimes(diagram)
    return float(lifetimes.max()) if lifetimes.size else 0.0


def n_features(diagram: np.ndarray | None, min_persistence: float = 0.0) -> int:
    lifetimes = finite_lifetimes(diagram)
    return int((lifetimes > min_persistence).sum())


def persistence_entropy(diagram: np.ndarray | None) -> float:
    """Entropy of the normalised lifetime distribution: 0 for one dominant bar, log n for many."""
    lifetimes = finite_lifetimes(diagram)
    total = lifetimes.sum()
    if lifetimes.size == 0 or total <= 0:
        return 0.0
    pmf = lifetimes / total
    pmf = pmf[pmf > 0]
    return float(-(pmf * np.log(pmf)).sum())
