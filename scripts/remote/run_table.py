"""One row per run set: how many seeds grokked, and the median grokking step.

Reads metrics directly rather than the analysis outputs, so it works while a sweep is
still in flight.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys

GROK_THRESHOLD = 0.9


def summarise(root: str) -> None:
    sets = collections.defaultdict(list)
    for run in sorted(os.listdir(root)):
        path = os.path.join(root, run, "metrics.jsonl")
        if not os.path.exists(path):
            continue
        t_g, last, steps = None, 0.0, 0
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                last, steps = record.get("test_acc", 0.0), record["step"]
                if t_g is None and last >= GROK_THRESHOLD:
                    t_g = record["step"]
        sets[re.sub(r"_s\d+$", "", run)].append((t_g, steps, last))

    print(f"{'run set':<46}{'grokked':>9}{'median t_g':>12}{'final test':>12}")
    for name, runs in sorted(sets.items(), key=lambda kv: _sort_key(kv[1])):
        hits = sorted(t for t, _, _ in runs if t)
        median = hits[len(hits) // 2] if hits else None
        final = sum(f for _, _, f in runs) / len(runs)
        print(
            f"{name:<46}{len(hits)}/{len(runs):<7}"
            f"{(str(median) if median else '—'):>12}{final:>12.3f}"
        )


def _sort_key(runs):
    hits = sorted(t for t, _, _ in runs if t)
    return (0, hits[len(hits) // 2]) if hits else (1, 0)


if __name__ == "__main__":
    summarise(sys.argv[1] if len(sys.argv) > 1 else "results/raw")
