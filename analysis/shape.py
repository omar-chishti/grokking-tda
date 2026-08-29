"""Is there one dominant cycle, and can that be asked without a normaliser? (sections 3.3, 4.3)

The reference paper's qualitative claim is that a single long-lived 1-cycle separates from the rest
of the diagram at grokking, and this thesis reproduces the picture without ever reducing it to a
number: section 4.3 shows the diagrams and describes the separation in prose. Maximum persistence
does not measure it. A diagram whose bars all grow by the same factor has a larger maximum and no
more dominance, which is exactly the failure the scale normalisation of section 3.3.1 exists to fix.

Two statistics ask the question directly, and both are ratios of lifetimes drawn from the same
diagram, so both are **invariant to the scale of the point cloud by construction** --- they need no
normaliser, no null band built on one, and no argument about which denominator is right.

``dominance``   the longest bar over the 90th percentile of the others: how far the leading cycle
                stands above the bulk of the diagram, rather than above the second bar, which on a
                noisy diagram is nearly as long as the first.
``share``       the longest bar's share of total persistence: the same question asked of the whole
                lifetime distribution, and the quantity whose entropy the persistence entropy is.

Bar counts are reported beside them, because both statistics depend on how many bars a diagram has
and the moduli in the bank differ.

Usage (from ``Code/``)::

    uv run python -m analysis.shape
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import (
    BASELINE_WINDOW,
    CONDITION_KEYS,
    NULL_BASELINE_WINDOW,
    NULL_PLATEAU_FROM,
    PLATEAU_FROM,
    is_replicate,
    iter_runs,
    task_modulus,
)

MIN_BARS = 5  # below this the percentile is meaningless


def diagram_shape(path: Path) -> tuple[float, float, int]:
    """Dominance, share and bar count for one degree-one diagram."""
    with np.load(path) as z:
        bars = z["dim1"]
    if bars.size == 0:
        return (np.nan, np.nan, 0)
    life = bars[:, 1] - bars[:, 0]
    life = np.sort(life[np.isfinite(life) & (life > 0)])[::-1]
    if life.size < MIN_BARS:
        return (np.nan, np.nan, int(life.size))
    rest = np.percentile(life[1:], 90)
    return (
        float(life[0] / rest) if rest > 0 else np.nan,
        float(life[0] / life.sum()),
        int(life.size),
    )


def series(run) -> pd.DataFrame:
    """Both statistics at every checkpoint of one run."""
    folder = run.directory / "analysis" / "diagrams"
    rows = []
    for path in sorted(folder.glob("*.npz")):
        step = re.search(r"step_(\d+)_", path.name)
        if step is None:
            continue
        dominance, share, bars = diagram_shape(path)
        rows.append({"step": int(step.group(1)), "dominance": dominance,
                     "share": share, "n_bars": bars})
    return pd.DataFrame(rows)


def windowed(table: pd.DataFrame, t_g: float | None) -> dict[str, float]:
    steps, end = table.step.to_numpy(float), float(table.step.max())
    if t_g:
        lo, hi, plateau = BASELINE_WINDOW[0] * t_g, BASELINE_WINDOW[1] * t_g, PLATEAU_FROM * t_g
    else:
        lo, hi = NULL_BASELINE_WINDOW[0] * end, NULL_BASELINE_WINDOW[1] * end
        plateau = NULL_PLATEAU_FROM * end
    out: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for column in ("dominance", "share", "n_bars"):
            base = table.loc[(steps >= lo) & (steps <= hi), column].to_numpy(float)
            plat = table.loc[steps >= plateau, column].to_numpy(float)
            out[f"{column}__baseline"] = float(np.nanmedian(base)) if base.size else np.nan
            out[f"{column}__plateau"] = float(np.nanmedian(plat)) if plat.size else np.nan
        for column in ("dominance", "share"):
            b, p = out[f"{column}__baseline"], out[f"{column}__plateau"]
            out[f"{column}__ratio"] = p / b if b and np.isfinite(b) and b > 0 else np.nan
    return out


def main() -> None:
    args = cli.parser(__doc__).parse_args()

    rows = []
    for run in iter_runs(args.root):
        if is_replicate(run):
            continue
        table = series(run)
        if table.empty:
            continue
        t_g = run.summary.get("grokking_step")
        cfg = run.config
        row = {"run": run.name, "t_g": t_g, "grokked": bool(t_g)}
        row["model"] = cfg["model"]["name"]
        row["operation"] = cfg["data"]["operation"]
        row["modulus"] = task_modulus(cfg["data"])
        row["train_fraction"] = cfg["data"]["train_fraction"]
        row["label_permutation"] = cfg["data"]["label_permutation"]
        row["loss"] = cfg["train"]["loss"]
        row["optimizer"] = cfg["train"]["optimizer"]["name"]
        row["lr"] = cfg["train"]["optimizer"]["lr"]
        row["weight_decay"] = cfg["train"]["optimizer"]["weight_decay"]
        row.update(windowed(table, float(t_g) if t_g else None))
        rows.append(row)

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "diagram_shape.csv", index=False)

    by_condition = (
        frame.groupby(CONDITION_KEYS, dropna=False)
        .agg(n=("run", "size"), grokked=("grokked", "sum"),
             bars=("n_bars__plateau", "median"),
             dom_base=("dominance__baseline", "median"),
             dom_plat=("dominance__plateau", "median"),
             dom_ratio=("dominance__ratio", "median"),
             share_ratio=("share__ratio", "median"))
        .sort_values("dom_ratio", ascending=False)
    )
    by_condition.to_csv(args.out / "diagram_shape_conditions.csv")

    print(f"{len(frame)} runs, re-runs excluded\n")
    print(f"{'condition':<40}{'n':>3}{'grok':>5}{'bars':>6}{'dominance':>22}{'share ratio':>13}")
    for key, row in by_condition.iterrows():
        fields = dict(zip(CONDITION_KEYS, key, strict=True))
        name = (
            f"{fields['model']} {fields['operation']}{fields['modulus']:g} "
            f"f{fields['train_fraction']:g} wd{fields['weight_decay']:g}"
            + (" PERM" if fields["label_permutation"] else "")
            + ("" if fields["loss"] == "softmax_ce" else " smax")
            + ("" if "ortho" not in fields["optimizer"] else " ortho")
        )
        arrow = f"{row.dom_base:.2f} -> {row.dom_plat:.2f} ({row.dom_ratio:.2f})"
        print(f"{name:<40}{int(row.n):>3}{int(row.grokked):>5}{row.bars:>6.0f}"
              f"{arrow:>22}{row.share_ratio:>13.2f}")
    print(f"\nwritten to {args.out}/diagram_shape_conditions.csv")


if __name__ == "__main__":
    main()
