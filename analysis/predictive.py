"""The completed predictive head-to-head (§5.4): full, extreme condition held out, winsorised."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis import cli
from grokking_tda.analysis.aggregate import early_window_table
from grokking_tda.analysis.identity import condition_key
from grokking_tda.evaluation.headtohead import head_to_head
from grokking_tda.evaluation.predictive import PREREGISTERED_WINDOWS, before_the_event

# the configuration §5.4 names as the candidate driver of the negative, as a condition key
EXTREME_GROUP = condition_key(
    {
        "model": {"name": "transformer"},
        "data": {
            "operation": "add",
            "modulus": 97,
            "train_fraction": 0.3,
            "label_permutation": False,
        },
        "train": {
            "loss": "softmax_ce",
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.1},
        },
    }
)
WINSOR = (0.10, 0.90)


def regression_variants(table: pd.DataFrame, window: str) -> pd.DataFrame:
    late = table[table["grokking_step"].notna()].copy()
    late["target"] = np.log10(late["grokking_step"].astype(float))
    if late["group"].nunique() < 3:
        return pd.DataFrame()

    held = late[late["group"] != EXTREME_GROUP]

    frames = []
    for name, frame, winsor in (
        ("full", late, None),
        ("holdout", held, None),
        ("winsorised", late, WINSOR),
    ):
        if frame["group"].nunique() < 3:
            continue
        scored = head_to_head(frame, task="regression", window=window, winsor=winsor)
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

    frames: list[pd.DataFrame] = []
    tables: dict[str, pd.DataFrame] = {}
    for window in (w.strip() for w in args.windows.split(",") if w.strip()):
        table = early_window_table(args.root, window)
        if table.empty:
            continue
        tables[window] = table

        # both grids: the guarded one is the result, the leaked one is what it corrects
        for guarded, frame in ((True, before_the_event(table)), (False, table)):
            classified = frame.assign(target=frame["grokking_step"].notna().astype(float))
            scored = head_to_head(classified, task="classification", window=window)
            scored["variant"] = "full"
            scored["n_runs"] = len(frame)
            scored["n_groups"] = frame["group"].nunique()
            regressed = regression_variants(frame, window)
            for part in (scored, regressed):
                part["guarded"] = guarded
            frames.extend((scored, regressed))

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("no runs with early-window features under that root")
    result = pd.concat(frames, ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "head_to_head.csv", index=False)

    for task in ("classification", "regression"):
        sub = result[(result.task == task) & result.guarded]
        if sub.empty:
            continue
        print(f"\n{task}, {'AUC' if task == 'classification' else 'R^2'} (mean +- fold sd):")
        pivot = sub.pivot_table(
            index=["window", "variant"], columns="feature_set", values="score_mean"
        )
        print(pivot.round(3).to_string())

    def best_by_variant(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {}
        best = frame.loc[frame.groupby("variant").score_mean.idxmax()]
        return {
            r.variant: {
                "window": r.window,
                "feature_set": r.feature_set,
                "r2": r.score_mean,
                "fold_sd": r.score_std,
                "n_runs": int(r.n_runs),
                "n_groups": int(r.n_groups),
            }
            for r in best.itertuples()
        }

    regression = result[result.task == "regression"]
    summary = {
        "extreme_group_held_out": EXTREME_GROUP,
        "winsor_quantiles": list(WINSOR),
        "best_regression_by_variant": best_by_variant(regression[regression.guarded]),
        "unguarded": best_by_variant(regression[~regression.guarded]),
        "dropped_by_window": {
            window: int(len(table) - len(before_the_event(table)))
            for window, table in tables.items()
        },
    }
    (args.out / "head_to_head.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {args.out}/head_to_head.csv and head_to_head.json")


if __name__ == "__main__":
    main()
