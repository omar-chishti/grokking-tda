"""Snapshot-step schedules.

Grokking happens over a wide range of step counts, so we space heavy snapshots
*logarithmically* by default: dense early (to catch memorisation) and dense enough
to bracket a late transition without storing every step. The endpoints (0 and the
final step) are always included.
"""

from __future__ import annotations

import numpy as np


def snapshot_steps(
    total_steps: int,
    n_snapshots: int,
    schedule: str = "log",
    dense_from: int = 0,
    dense_to: int = 0,
) -> list[int]:
    """Return a sorted, de-duplicated list of steps at which to take snapshots.

    A ``dense_from``/``dense_to`` window spends half the budget linearly inside it
    and the rest on the base schedule. Log spacing alone puts most snapshots in the
    first fraction of a percent of training while leaving the transition — where the
    signed-lag analysis needs resolution finer than the lag it is measuring — sampled
    only every several hundred steps.
    """
    if n_snapshots < 2:
        return [0, total_steps]

    dense = np.empty(0)
    if 0 <= dense_from < dense_to:
        n_dense = n_snapshots // 2
        dense = np.linspace(dense_from, min(dense_to, total_steps), n_dense)
        n_snapshots -= n_dense

    if schedule == "linear":
        pts = np.linspace(0, total_steps, n_snapshots)
    elif schedule == "log":
        # log-space over [1, total_steps], then prepend 0.
        log_pts = np.logspace(0, np.log10(max(total_steps, 1)), n_snapshots - 1)
        pts = np.concatenate([[0.0], log_pts])
    else:
        raise ValueError(f"unknown schedule {schedule!r}; choices: log, linear")

    steps = sorted({int(round(x)) for x in np.concatenate([pts, dense])} | {0, total_steps})
    return [s for s in steps if 0 <= s <= total_steps]
