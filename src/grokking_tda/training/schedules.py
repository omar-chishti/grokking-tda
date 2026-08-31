"""Snapshot-step schedules: logarithmic by default, with an optional dense window."""

from __future__ import annotations

import numpy as np


def snapshot_steps(
    total_steps: int,
    n_snapshots: int,
    schedule: str = "log",
    dense_from: int = 0,
    dense_to: int = 0,
) -> list[int]:
    """Snapshot steps, log-spaced, with an optional dense window taking half the budget."""
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
        log_pts = np.logspace(0, np.log10(max(total_steps, 1)), n_snapshots - 1)
        pts = np.concatenate([[0.0], log_pts])
    else:
        raise ValueError(f"unknown schedule {schedule!r}; choices: log, linear")

    steps = sorted({int(round(x)) for x in np.concatenate([pts, dense])} | {0, total_steps})
    return [s for s in steps if 0 <= s <= total_steps]
