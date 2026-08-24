"""One line per run: latest step, accuracies, and whether it has grokked yet.

Called by `gtda-remote status` on the remote side, where the results tree lives.
A separate file rather than an inline `python3 -c`, which would have to survive two
levels of shell quoting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GROK_THRESHOLD = 0.9


def last_metrics(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        if line.strip():
            last = line
    return json.loads(last) if last else None


def grokking_step(run_dir: Path) -> int | None:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("test_acc", 0.0) >= GROK_THRESHOLD:
            return int(record["step"])
    return None


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics = last_metrics(run_dir)
        if metrics is None:
            print(f"  {run_dir.name:<46} (no metrics yet)")
            continue
        t_g = grokking_step(run_dir)
        mark = f"GROKKED @ {t_g}" if t_g else ""
        print(
            f"  {run_dir.name:<46} step {int(metrics['step']):6d}"
            f"  train {metrics['train_acc']:.3f}  test {metrics['test_acc']:.3f}  {mark}"
        )


if __name__ == "__main__":
    main()
