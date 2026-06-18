"""Scalar summaries of a persistence diagram.

These are the topological observables tracked over training. ``max`` and ``total``
H1 persistence are the signatures reported by Tang et al. Infinite (essential) bars
are excluded from lifetime sums, as is conventional.
"""

from __future__ import annotations

import numpy as np


def finite_lifetimes(diagram: np.ndarray) -> np.ndarray:
    """Lifetimes (death - birth) of the finite bars in a diagram."""
    if diagram is None or diagram.size == 0:
        return np.empty(0)
    finite = diagram[np.isfinite(diagram[:, 1])]
    if finite.size == 0:
        return np.empty(0)
    return finite[:, 1] - finite[:, 0]


def total_persistence(diagram: np.ndarray) -> float:
    lifetimes = finite_lifetimes(diagram)
    return float(lifetimes.sum()) if lifetimes.size else 0.0


def max_persistence(diagram: np.ndarray) -> float:
    lifetimes = finite_lifetimes(diagram)
    return float(lifetimes.max()) if lifetimes.size else 0.0


def n_features(diagram: np.ndarray, min_persistence: float = 0.0) -> int:
    """Number of finite bars with lifetime strictly above ``min_persistence``."""
    lifetimes = finite_lifetimes(diagram)
    return int((lifetimes > min_persistence).sum())


def persistence_entropy(diagram: np.ndarray) -> float:
    """Shannon entropy of the normalised lifetime distribution (nats).

    A single robust scalar for how *organised* a diagram is: one dominant bar gives
    entropy near 0; many equal bars give ``log(n)``. Empty diagrams return 0.
    """
    lifetimes = finite_lifetimes(diagram)
    total = lifetimes.sum()
    if lifetimes.size == 0 or total <= 0:
        return 0.0
    pmf = lifetimes / total
    pmf = pmf[pmf > 0]
    return float(-(pmf * np.log(pmf)).sum())
