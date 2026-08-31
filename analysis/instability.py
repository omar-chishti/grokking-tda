"""Do the transient training collapses reach the analysed checkpoints? (§7.4)"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import RATIO_OBSERVABLES, iter_runs, window_medians

CONVERGED = 0.99  # training accuracy that counts as having fit the training set
COLLAPSED = 0.5  # ... and the level a collapse falls below


def collapse_intervals(metrics: pd.DataFrame) -> list[tuple[float, float]]:
    """Post-convergence intervals below the floor, framed by the records either side."""
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
    obs = obs.loc[keep]
    if obs.empty:
        return {}
    out = {}
    for column in RATIO_OBSERVABLES:
        if column not in obs:
            continue
        # a diverged run is all-NaN throughout, and window_medians returns NaN for it
        base, after = window_medians(obs, column, t_g)
        out[f"{column}__ratio"] = float(after / base) if base else np.nan
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
