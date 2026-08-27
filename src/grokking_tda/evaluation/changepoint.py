"""A second transition detector, so the timing result does not rest on one definition.

The midpoint-crossing proxy in ``transitions.py`` is deliberately assumption-light, which
also makes it crude: it reads a single crossing and says nothing about whether the series
actually changed regime. This module segments the series instead, by exact optimal
partitioning under a squared-error cost with a linear penalty per changepoint — the same
optimum the PELT algorithm computes, reached here by dynamic programming because a
snapshot series is a few hundred points and pruning buys nothing at that size.

Agreement between the two detectors is a robustness result; disagreement is a finding
about how sharply defined the transition is, and either way it belongs in the thesis
rather than in a footnote.
"""

from __future__ import annotations

import numpy as np


def _segment_costs(values: np.ndarray) -> np.ndarray:
    """``cost[i, j]`` = squared error of the best constant fit to ``values[i:j]``."""
    n = values.size
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(values**2)])
    cost = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        lengths = np.arange(1, n - i + 1)
        total = prefix[i + 1 : n + 1] - prefix[i]
        total_sq = prefix_sq[i + 1 : n + 1] - prefix_sq[i]
        cost[i, i + 1 :] = total_sq - (total**2) / lengths
    return cost


def changepoints(values, penalty: float | None = None, max_changes: int = 4) -> list[int]:
    """Indices at which the series changes level, as an exact optimal partition.

    ``penalty`` defaults to the BIC-like ``sigma^2 * log n`` estimated from the series'
    successive differences, which is robust to the level shifts being detected in a way
    that the raw variance is not.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return []
    if penalty is None:
        # Median absolute successive difference -> a scale unaffected by the shifts.
        mad = np.median(np.abs(np.diff(x))) / 0.6745 if n > 1 else 0.0
        penalty = max((mad**2) * np.log(n), 1e-12)

    cost = _segment_costs(x)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    previous = np.zeros(n + 1, dtype=int)
    for end in range(1, n + 1):
        starts = np.arange(end)
        candidates = best[starts] + cost[starts, end] + penalty
        k = int(np.argmin(candidates))
        best[end], previous[end] = candidates[k], k

    cuts: list[int] = []
    at = n
    while at > 0:
        start = previous[at]
        if start > 0:
            cuts.append(int(start))
        at = start
    cuts.reverse()
    return cuts[:max_changes] if max_changes else cuts


def changepoint_step(steps, values, direction: str = "rising") -> int | None:
    """The step of the largest level shift in the declared direction, or ``None``.

    Reported beside ``transition_step`` for every observable: two detectors that disagree
    are telling you the transition is not sharply located, which is itself worth knowing.
    """
    s = np.asarray(steps)
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 4:
        return None
    s, x = s[mask], x[mask]
    if direction == "falling":
        x = -x
    elif direction not in {"rising", "auto"}:
        raise ValueError(f"unknown direction {direction!r}")

    cuts = changepoints(x)
    if not cuts:
        return None
    bounds = [0, *cuts, x.size]
    jumps = []
    for i in range(len(bounds) - 2):
        before = x[bounds[i] : bounds[i + 1]].mean()
        after = x[bounds[i + 1] : bounds[i + 2]].mean()
        jumps.append((after - before, cuts[i]))
    rises = [j for j in jumps if j[0] > 0]
    if not rises:
        return None
    return int(s[max(rises)[1]])
