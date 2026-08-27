"""Do the transient training collapses contaminate the analysed checkpoints? (section 7.4)

Thesis section 7.4 records three things and closes none of them: that 97 of 144 runs
reaching training accuracy 0.99 later collapse below 0.5 and recover, that snapshots are
not aligned to avoid those collapses, and that no analysis establishes a post-collapse
checkpoint is representative. The first is a measurement, the second is a design fact,
and the third is a gap that can simply be filled --- the metric grid records training
accuracy throughout, and the snapshot steps are known, so whether any analysed snapshot
actually sits inside a collapse is a lookup rather than an argument.

Where snapshots do land in a collapse, the window ratios are recomputed with those
checkpoints dropped. If the headline numbers do not move, a confessed threat becomes a
closed one.

Usage (from ``Code/``)::

    uv run python -m analysis.instability
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import (
    BASELINE_WINDOW,
    NULL_BASELINE_WINDOW,
    NULL_PLATEAU_FROM,
    PLATEAU_FROM,
    RATIO_OBSERVABLES,
    iter_runs,
)

CONVERGED = 0.99  # training accuracy that counts as having fit the training set
COLLAPSED = 0.5  # ... and the level a collapse falls below


def collapse_intervals(metrics: pd.DataFrame) -> list[tuple[float, float]]:
    """Step intervals, after convergence, in which training accuracy is below the floor.

    An interval runs from the last metric record above the floor to the first one back
    above it, so a snapshot taken anywhere between two collapsed records is caught even
    though the metric grid is coarser than the collapse.
    """
    if metrics.empty or "train_acc" not in metrics:
        return []
    steps = metrics["step"].to_numpy(float)
    acc = metrics["train_acc"].to_numpy(float)
    converged = np.flatnonzero(acc >= CONVERGED)
    if converged.size == 0:
        return []

    below = (acc < COLLAPSED) & (np.arange(acc.size) > converged[0])
    intervals, start = [], None
    for i, flag in enumerate(below):
        if flag and start is None:
            start = steps[max(i - 1, 0)]
        elif not flag and start is not None:
            intervals.append((start, steps[i]))
            start = None
    if start is not None:
        intervals.append((start, steps[-1]))
    return intervals


def contaminated_steps(snapshot_steps, intervals) -> np.ndarray:
    steps = np.asarray(snapshot_steps, dtype=float)
    if not intervals:
        return np.zeros(steps.shape, dtype=bool)
    lo = np.array([a for a, _ in intervals])[:, None]
    hi = np.array([b for _, b in intervals])[:, None]
    return ((steps[None, :] >= lo) & (steps[None, :] <= hi)).any(axis=0)


def ratios(obs: pd.DataFrame, t_g: float | None, keep: np.ndarray) -> dict:
    """Window ratios over the retained snapshots, by the thesis window rule."""
    obs = obs.loc[keep]
    if obs.empty:
        return {}
    steps, end = obs["step"].to_numpy(float), float(obs["step"].max())
    if t_g:
        lo, hi, plat = BASELINE_WINDOW[0] * t_g, BASELINE_WINDOW[1] * t_g, PLATEAU_FROM * t_g
    else:
        lo, hi = NULL_BASELINE_WINDOW[0] * end, NULL_BASELINE_WINDOW[1] * end
        plat = NULL_PLATEAU_FROM * end

    out = {}
    for column in RATIO_OBSERVABLES:
        if column not in obs:
            continue
        base = obs.loc[(steps >= lo) & (steps <= hi), column].to_numpy(float)
        after = obs.loc[steps >= plat, column].to_numpy(float)
        if base.size == 0 or after.size == 0:
            continue
        # A diverged run is all-NaN throughout; the divergence is recorded elsewhere.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            b, a = np.nanmedian(base), np.nanmedian(after)
        out[f"{column}__ratio"] = float(a / b) if b else np.nan
    return out


def main() -> None:
    ap = cli.parser(__doc__)
    args = ap.parse_args()

    rows = []
    for run in iter_runs(args.root):
        metrics_path = args.root / run.name / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        metrics = pd.DataFrame(
            json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()
        )
        intervals = collapse_intervals(metrics)
        obs = run.observables
        hit = contaminated_steps(obs["step"], intervals)
        t_g = run.summary.get("grokking_step")

        row = {
            "run": run.name,
            "t_g": t_g,
            "n_collapses": len(intervals),
            "n_snapshots": len(obs),
            "n_contaminated": int(hit.sum()),
        }
        row.update({f"{k}__all": v for k, v in ratios(obs, t_g, np.ones(len(obs), bool)).items()})
        if hit.any():
            row.update({f"{k}__clean": v for k, v in ratios(obs, t_g, ~hit).items()})
        rows.append(row)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "instability.csv", index=False)

    affected = table[table.n_contaminated > 0]
    print(
        f"{len(table)} runs; {int((table.n_collapses > 0).sum())} collapse at least once; "
        f"{len(affected)} have an analysed snapshot inside a collapse "
        f"({int(table.n_contaminated.sum())} of {int(table.n_snapshots.sum())} snapshots)"
    )
    for column in RATIO_OBSERVABLES:
        both = affected.dropna(subset=[f"{column}__ratio__all", f"{column}__ratio__clean"])
        if both.empty:
            continue
        shift = (both[f"{column}__ratio__clean"] / both[f"{column}__ratio__all"]).abs()
        print(
            f"  {column:36s} n={len(both):3d}  ratio of ratios: "
            f"median {shift.median():.3f}  worst {shift.max():.3f}"
        )
    print(f"\nwritten to {args.out}/instability.csv")


if __name__ == "__main__":
    main()
