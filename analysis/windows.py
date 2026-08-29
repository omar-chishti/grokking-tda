"""How much does the window rule decide the answer? (sections 4.2, A.2)

Section 4.2 fixes one rule for what counts as before and after a transition, and then makes
a claim about it that nothing in the analysis layer measures: that the choice of window
matters for the raw persistence series and much less for the normalised one. The claim is
load-bearing, because it is the fourth argument for the correction of section 3.3.1 and the
only one that does not go through the null band, and it is the kind of claim a reader is
entitled to see tested rather than asserted.

Every ratio is therefore recomputed under four alternatives to the reported rule: a
narrower and a wider baseline, and a later and an earlier plateau bound. What is reported
is the shift each alternative produces, in log2 units so that a doubling and a halving
count the same, taken per run and then summarised over runs. A statistic whose answer
depends on where the windows are placed is not measuring the transition.

Dense re-runs are excluded, as everywhere else in this directory: they repeat existing
conditions on a different snapshot schedule, which is itself a change of window.

Usage (from ``Code/``)::

    uv run python -m analysis.windows
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import RATIO_OBSERVABLES, is_replicate, iter_runs

REPORTED = "0.5-0.9 / 1.2"

# baseline window as a fraction of t_g, and the step at which the plateau begins
WINDOWS: dict[str, tuple[tuple[float, float], float]] = {
    REPORTED: ((0.5, 0.9), 1.2),
    "0.7-0.9 / 1.2": ((0.7, 0.9), 1.2),
    "0.3-0.9 / 1.2": ((0.3, 0.9), 1.2),
    "0.5-0.9 / 1.5": ((0.5, 0.9), 1.5),
    "0.5-0.9 / 1.05": ((0.5, 0.9), 1.05),
}


Window = tuple[tuple[float, float], float]


def ratio(obs: pd.DataFrame, column: str, t_g: float, window: Window) -> float:
    (lo, hi), plateau = window
    steps = obs["step"].to_numpy(float)
    with warnings.catch_warnings():  # an all-NaN window is expected for a diverged run
        warnings.simplefilter("ignore", RuntimeWarning)
        base = np.nanmedian(obs.loc[(steps >= lo * t_g) & (steps <= hi * t_g), column])
        plat = np.nanmedian(obs.loc[steps >= plateau * t_g, column])
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
        row = {"run": run.name, "t_g": float(t_g)}
        for name, window in WINDOWS.items():
            for column in RATIO_OBSERVABLES:
                if column in obs:
                    row[f"{column}__{name}"] = ratio(obs, column, float(t_g), window)
        rows.append(row)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "window_sensitivity.csv", index=False)

    print(f"{len(table)} grokking runs, re-runs excluded\n")
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
