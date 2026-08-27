"""``gtda-compare`` — does topology predict grokking beyond the cheap baselines?

    gtda-compare results/raw --out results/processed/head_to_head.csv

Scores every feature set on identical folds, at every pre-registered early window, for
two targets: whether a run groks at all (AUC) and how late it groks (R^2 on log t_g).
Folds are split by configuration, so a run's own seeds never sit in its training set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grokking_tda.analysis.aggregate import early_window_table
from grokking_tda.evaluation.headtohead import head_to_head
from grokking_tda.evaluation.predictive import PREREGISTERED_WINDOWS
from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="directory tree containing run dirs")
    parser.add_argument("--out", type=Path, default=Path("results/processed/head_to_head.csv"))
    parser.add_argument(
        "--windows",
        default=",".join(f"w{w}" for w in PREREGISTERED_WINDOWS) + ",tc",
        help="comma-separated early-window keys",
    )
    args = parser.parse_args()

    frames = []
    for window in (w.strip() for w in str(args.windows).split(",") if w.strip()):
        table = early_window_table(args.root, window)
        if table.empty:
            logger.warning("no features at window %s", window)
            continue

        grokked = table.assign(target=table["grokking_step"].notna().astype(float))
        frames.append(head_to_head(grokked, task="classification", window=window))

        late = table[table["grokking_step"].notna()].copy()
        if late["group"].nunique() >= 3:
            late["target"] = np.log10(late["grokking_step"].astype(float))
            frames.append(head_to_head(late, task="regression", window=window))

    if not frames:
        raise SystemExit("no runs with early-window features under that root")
    result = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    logger.info("%d comparisons -> %s", len(result), args.out)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
