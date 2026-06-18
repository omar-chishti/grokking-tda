"""Distances between persistence diagrams, and the trajectory-velocity series.

The distance between *consecutive* snapshots' diagrams is a topological "speed" of
the training trajectory: flat while the representation drifts, peaking when its
shape reorganises. Its changepoint can be compared against the grokking step — a
cheap, high-value trajectory-topology result (thesis figure 6.3). Distances also
serve distance-to-null comparisons and the circularity probe (distance of the
embedding diagram to that of an ideal p-gon).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from persim import bottleneck, sliced_wasserstein


def _finite(diagram: np.ndarray | None) -> np.ndarray:
    if diagram is None or diagram.size == 0:
        return np.empty((0, 2))
    return diagram[np.isfinite(diagram[:, 1])]


def diagram_distance(
    d1: np.ndarray | None, d2: np.ndarray | None, metric: str = "sliced_wasserstein"
) -> float:
    """Distance between two diagrams (finite bars only).

    ``bottleneck`` is the stability-theorem metric; ``sliced_wasserstein`` is a fast,
    smoother proxy better suited to velocity curves. An empty diagram is treated as
    diagonal-only, so the bottleneck distance to it is half the longest lifetime.
    """
    a, b = _finite(d1), _finite(d2)
    if a.size == 0 and b.size == 0:
        return 0.0
    if metric == "bottleneck":
        if a.size == 0 or b.size == 0:
            other = b if a.size == 0 else a
            return float((other[:, 1] - other[:, 0]).max() / 2.0)
        return float(bottleneck(a, b))
    if metric == "sliced_wasserstein":
        try:
            return float(sliced_wasserstein(a, b))
        except Exception:  # persim can struggle with degenerate/empty inputs
            return float("nan")
    raise ValueError(f"unknown metric {metric!r}; choices: bottleneck, sliced_wasserstein")


def trajectory_velocity(
    steps: Sequence[int],
    diagrams: Sequence[np.ndarray | None],
    metric: str = "sliced_wasserstein",
) -> pd.DataFrame:
    """Per-step distance between consecutive diagrams.

    Row ``i`` (at ``steps[i]``) holds the distance between the diagrams at
    ``steps[i-1]`` and ``steps[i]``. Returns a ``(step, distance)`` DataFrame.
    """
    if len(steps) != len(diagrams):
        raise ValueError("steps and diagrams must have equal length")
    rows = [
        {"step": int(steps[i]), "distance": diagram_distance(diagrams[i - 1], diagrams[i], metric)}
        for i in range(1, len(steps))
    ]
    return pd.DataFrame(rows, columns=["step", "distance"])
