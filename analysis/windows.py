"""How much does the window rule decide the answer? (§4.2, A.2)"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import (
    RATIO_OBSERVABLES,
    is_replicate,
    iter_runs,
    plateau_relaxed,
    window_medians,
)

REPORTED = "0.5-0.9 / 1.2"

WINDOWS: dict[str, tuple[tuple[float, float], float]] = {
    REPORTED: ((0.5, 0.9), 1.2),
    "0.7-0.9 / 1.2": ((0.7, 0.9), 1.2),
    "0.3-0.9 / 1.2": ((0.3, 0.9), 1.2),
    "0.5-0.9 / 1.5": ((0.5, 0.9), 1.5),
    "0.5-0.9 / 1.05": ((0.5, 0.9), 1.05),
}


Window = tuple[tuple[float, float], float]


def ratio(obs: pd.DataFrame, column: str, t_g: float, window: Window) -> float:
    baseline, plateau = window
    base, plat = window_medians(obs, column, t_g, baseline=baseline, plateau_from=plateau)
    if not np.isfinite(base) or base <= 0 or not np.isfinite(plat):
        return float("nan")
    return float(plat / base)


def main() -> None:
    args = cli.parser(__doc__).parse_args()

    rows = []
    for run in iter_runs(args.root):
        t_g = run.summary.get("grokking_step")
        if not t_g or is_replicate(run):
            continue
        obs = run.observables
        row = {"run": run.name, "t_g": float(t_g), "plateau_relaxed": plateau_relaxed(obs, t_g)}
        for name, window in WINDOWS.items():
            for column in RATIO_OBSERVABLES:
                if column in obs:
                    row[f"{column}__{name}"] = ratio(obs, column, float(t_g), window)
        rows.append(row)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "window_sensitivity.csv", index=False)

    relaxed = int(table.plateau_relaxed.sum())
    print(f"{len(table)} grokking runs, re-runs excluded; {relaxed} with a relaxed plateau\n")
    print(f"{'observable':<34}{'window':<18}{'median ratio':>13}{'median |log2 shift|':>21}")
    for column in RATIO_OBSERVABLES:
        reference = table.get(f"{column}__{REPORTED}")
        if reference is None:
            continue
        for name in WINDOWS:
            values = table[f"{column}__{name}"]
            shift = np.abs(np.log2(values / reference)).replace([np.inf, -np.inf], np.nan).dropna()
            print(
                f"{column:<34}{name:<18}{values.median():>13.3f}"
                f"{(shift.median() if len(shift) else float('nan')):>21.3f}"
            )
        print()
    print(f"written to {args.out}/window_sensitivity.csv")


if __name__ == "__main__":
    main()
