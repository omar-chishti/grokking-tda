"""Distances between persistence diagrams, and the trajectory-velocity series."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from persim import bottleneck, sliced_wasserstein

from grokking_tda.tda.summaries import finite_bars


def diagram_distance(
    d1: np.ndarray | None, d2: np.ndarray | None, metric: str = "sliced_wasserstein"
) -> float:
    """Distance between two diagrams; an empty one is diagonal-only."""
    a, b = finite_bars(d1), finite_bars(d2)
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
    if len(steps) != len(diagrams):
        raise ValueError("steps and diagrams must have equal length")
    rows = [
        {"step": int(steps[i]), "distance": diagram_distance(diagrams[i - 1], diagrams[i], metric)}
        for i in range(1, len(steps))
    ]
    return pd.DataFrame(rows, columns=["step", "distance"])
