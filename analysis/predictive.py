"""The completed predictive head-to-head (thesis section 5.4).

``gtda-compare`` scores every feature set once, on the whole table. That is not enough to
report the regression's negative $R^2$ as a property of the observables, because the
design is unbalanced by construction: the weight-decay-$0.1$ configuration has a median
$\\tg$ of $140{,}100$, fifty times the fastest condition's, and folds are split by
configuration, so no model trained without it can predict it. A negative driven by one
extreme cell is a design artefact; a negative that survives its removal is a statement
about the whole family of progress measures.

Three fits, on identical folds and identical features:

``full``
    every grokking run, as the chapter currently reports it.
``holdout``
    the same, with the weight-decay-$0.1$ configuration dropped.
``winsorised``
    the same as ``full``, with $\\log \\tg$ clipped to its 10th and 90th percentiles, so
    the extremes stay in the sample but stop dominating the squared error.

Classification is scored on the full table in every case; the imbalance is a property of
the regression target, not of whether a run groks at all.

Usage (from ``Code/``)::

    uv run python -m analysis.predictive
    uv run python -m analysis.predictive --windows w5000
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis import cli
from grokking_tda.analysis.aggregate import early_window_table
from grokking_tda.evaluation.headtohead import head_to_head
from grokking_tda.evaluation.predictive import PREREGISTERED_WINDOWS

# The configuration section 5.4 names as the candidate driver of the negative.
EXTREME_GROUP = "transformer_add97_f0.3_wd0.1_softmax_ce"
WINSOR = (0.10, 0.90)


def regression_variants(table: pd.DataFrame, window: str) -> pd.DataFrame:
    """The three fits, each on the runs and target its name describes."""
    late = table[table["grokking_step"].notna()].copy()
    late["target"] = np.log10(late["grokking_step"].astype(float))
    if late["group"].nunique() < 3:
        return pd.DataFrame()

    held = late[late["group"] != EXTREME_GROUP]
    lo, hi = late["target"].quantile(WINSOR)
    clipped = late.assign(target=late["target"].clip(lo, hi))

    frames = []
    for name, frame in (("full", late), ("holdout", held), ("winsorised", clipped)):
        if frame["group"].nunique() < 3:
            continue
        scored = head_to_head(frame, task="regression", window=window)
        scored["variant"] = name
        scored["n_runs"] = len(frame)
        scored["n_groups"] = frame["group"].nunique()
        scored["target_range"] = frame["target"].max() - frame["target"].min()
        frames.append(scored)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--windows", default=",".join(f"w{w}" for w in PREREGISTERED_WINDOWS) + ",tc")
    args = ap.parse_args()

    frames = []
    for window in (w.strip() for w in args.windows.split(",") if w.strip()):
        table = early_window_table(args.root, window)
        if table.empty:
            continue

        classified = table.assign(target=table["grokking_step"].notna().astype(float))
        scored = head_to_head(classified, task="classification", window=window)
        scored["variant"] = "full"
        scored["n_runs"] = len(table)
        scored["n_groups"] = table["group"].nunique()
        frames.append(scored)
        frames.append(regression_variants(table, window))

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("no runs with early-window features under that root")
    result = pd.concat(frames, ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "head_to_head.csv", index=False)

    for task in ("classification", "regression"):
        sub = result[result.task == task]
        if sub.empty:
            continue
        print(f"\n{task}, {'AUC' if task == 'classification' else 'R^2'} (mean +- fold sd):")
        pivot = sub.pivot_table(
            index=["window", "variant"], columns="feature_set", values="score_mean"
        )
        print(pivot.round(3).to_string())

    late = result[(result.task == "regression")]
    best = late.loc[late.groupby("variant").score_mean.idxmax()] if not late.empty else late
    summary = {
        "extreme_group_held_out": EXTREME_GROUP,
        "winsor_quantiles": list(WINSOR),
        "best_regression_by_variant": {
            r.variant: {
                "window": r.window,
                "feature_set": r.feature_set,
                "r2": r.score_mean,
                "fold_sd": r.score_std,
                "n_runs": int(r.n_runs),
                "n_groups": int(r.n_groups),
            }
            for r in best.itertuples()
        },
    }
    (args.out / "head_to_head.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {args.out}/head_to_head.csv and head_to_head.json")


if __name__ == "__main__":
    main()
