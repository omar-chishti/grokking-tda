"""A second transition detector, by exact optimal partition — the optimum PELT computes."""

from __future__ import annotations

import numpy as np

from grokking_tda.evaluation.transitions import orient


def _segment_costs(values: np.ndarray) -> np.ndarray:
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


def changepoints(values, penalty: float | None = None) -> list[int]:
    """The exact optimal partition; the BIC-like penalty is the only control on how many cuts."""
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
    return cuts


def changepoint_step(
    steps, values, direction: str = "rising", compare: str = "trough"
) -> int | None:
    """The largest level shift after the trough. ``compare="global"`` keeps the unrepaired form."""
    s = np.asarray(steps)
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 4:
        return None
    s, x = s[mask], x[mask]
    x = orient(x, direction)
    if compare == "trough":
        trough = int(np.argmin(x))
        s, x = s[trough:], x[trough:]
    elif compare != "global":
        raise ValueError(f"unknown compare {compare!r}; choices: trough, global")

    cuts = changepoints(x)
    if not cuts:
        return None
    # Scored before-against-after, not between adjacent segments: how finely a noisy series
    # fragments must not decide where its one transition is
    rises = [(float(np.median(x[c:]) - np.median(x[:c])), c) for c in cuts]
    size, at = max(rises)
    return int(s[at]) if size > 0 else None
