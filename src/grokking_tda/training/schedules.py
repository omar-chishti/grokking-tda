"""Snapshot-step schedules.

Grokking happens over a wide range of step counts, so we space heavy snapshots
*logarithmically* by default: dense early (to catch memorisation) and dense enough
to bracket a late transition without storing every step. The endpoints (0 and the
final step) are always included.
"""

from __future__ import annotations

import numpy as np


def snapshot_steps(total_steps: int, n_snapshots: int, schedule: str = "log") -> list[int]:
    """Return a sorted, de-duplicated list of steps at which to take snapshots."""
    if n_snapshots < 2:
        return [0, total_steps]
    if schedule == "linear":
        pts = np.linspace(0, total_steps, n_snapshots)
    elif schedule == "log":
        # log-space over [1, total_steps], then prepend 0.
        log_pts = np.logspace(0, np.log10(max(total_steps, 1)), n_snapshots - 1)
        pts = np.concatenate([[0.0], log_pts])
    else:
        raise ValueError(f"unknown schedule {schedule!r}; choices: log, linear")
    steps = sorted({int(round(x)) for x in pts} | {0, total_steps})
    return [s for s in steps if 0 <= s <= total_steps]
