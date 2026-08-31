"""``gtda-aggregate`` — collect many analysed runs into one tidy table."""

from __future__ import annotations

import argparse
from pathlib import Path

from grokking_tda.analysis.aggregate import aggregate_runs
from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate runs into a robustness table.")
    parser.add_argument("root", type=Path, help="directory tree containing run dirs")
    parser.add_argument("--out", type=Path, default=None, help="output CSV path")
    args = parser.parse_args()

    table = aggregate_runs(args.root)
    out = args.out or (args.root / "robustness.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    logger.info("%d runs -> %s", len(table), out)
    print(f"{len(table)} runs -> {out}")


if __name__ == "__main__":
    main()
