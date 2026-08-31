"""Where the topological velocity changes rate, and whether it changes at all (§6.3)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import bootstrap_median_ci
from grokking_tda.evaluation import changepoint_step

VELOCITY = "figures/fig-6-2-velocity.csv"


def step_size(steps: np.ndarray, rate: np.ndarray, cut: float) -> float:
    before, after = rate[steps < cut], rate[steps >= cut]
    if before.size == 0 or after.size == 0:
        return float("nan")
    lo = float(np.nanmedian(before))
    return float(np.nanmedian(after) / lo) if lo else float("nan")


def locate(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (panel, run), sub in frame.groupby(["panel", "run"], sort=False):
        sub = sub.sort_values("step")
        steps = sub["step"].to_numpy(float)
        rate = sub["rate"].to_numpy(float)
        t_g = sub["t_g"].dropna().iloc[0] if sub["t_g"].notna().any() else None
        cut = changepoint_step(steps, rate, direction="rising")
        rows.append(
            {
                "panel": panel,
                "run": run,
                "t_g": t_g,
                "t_change": cut,
                "delta": (cut - t_g) if (cut is not None and t_g) else np.nan,
                "relative": (cut / t_g) if (cut is not None and t_g) else np.nan,
                "step_size": step_size(steps, rate, cut) if cut is not None else np.nan,
                "n_points": len(sub),
            }
        )
    return rows


def main() -> None:
    ap = cli.parser(__doc__, root=False)
    args = ap.parse_args()

    path = args.out / VELOCITY
    if not path.exists():
        raise SystemExit(f"{path} missing — run `python -m analysis.figures.build --only 6.2`")

    table = pd.DataFrame(locate(pd.read_csv(path)))
    table.to_csv(args.out / "velocity_changepoint.csv", index=False)

    summary = {}
    for panel, sub in table.groupby("panel", sort=False):
        found = sub[sub.t_change.notna()]
        entry = {
            "n_runs": len(sub),
            "n_located": len(found),
            "t_change": sorted(int(v) for v in found.t_change),
        }
        for column in ("relative", "delta", "step_size"):
            values = found[column].dropna()
            if values.empty:
                continue
            lo, med, hi = bootstrap_median_ci(values)
            entry[column] = {"median": med, "ci": [lo, hi]}
        summary[panel] = entry

        rel = entry.get("relative", {})
        size = entry.get("step_size", {})
        print(
            f"  {panel:10s} located {len(found)}/{len(sub)}"
            + (
                f"  at {rel['median']:.2f} t_g [{rel['ci'][0]:.2f}, {rel['ci'][1]:.2f}]"
                if rel
                else ""
            )
            + (
                f"  rate x{size['median']:.1f} [{size['ci'][0]:.1f}, {size['ci'][1]:.1f}]"
                if size
                else ""
            )
        )

    (args.out / "velocity_changepoint.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {args.out}/velocity_changepoint.csv and .json")


if __name__ == "__main__":
    main()
