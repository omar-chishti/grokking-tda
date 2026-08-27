"""Aggregate many runs into one tidy table for the robustness / predictive figures.

Walks a directory tree for run dirs (anything holding a ``manifest.json``), joins
each run's config keys with its ``analysis/summary.json`` (when present), and
returns one row per run. Per-observable transitions are flattened to
``t_top__<name>`` / ``delta__<name>`` columns. Everything in the robustness map,
lead-lag forest, and predictive figures reads this table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grokking_tda.artifacts.reader import Run
from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)

_SUMMARY_KEYS = (
    "train_convergence_step",
    "grokking_step",
    "topological_transition_step",
    "lead_lag_steps",
    "observable",
)


def _config_row(run: Run) -> dict:
    cfg = run.config
    data = cfg.get("data", {})
    train = cfg.get("train", {})
    optim = train.get("optimizer", {})
    return {
        "run": run.run_name,
        "path": str(run.dir),
        "seed": cfg.get("seed"),
        "model": cfg.get("model", {}).get("name"),
        "operation": data.get("operation"),
        "modulus": data.get("modulus"),
        "train_fraction": data.get("train_fraction"),
        "label_permutation": data.get("label_permutation", False),
        "loss": train.get("loss"),
        "optimizer": optim.get("name"),
        "weight_decay": optim.get("weight_decay"),
        "steps": train.get("steps"),
    }


def aggregate_runs(root: str | Path) -> pd.DataFrame:
    """One row per run under ``root`` (recursively), config joined with summary."""
    rows: list[dict] = []
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        run_dir = manifest_path.parent
        try:
            run = Run(run_dir)
        except Exception as exc:  # unreadable run: skip, never abort the sweep table
            logger.warning("skipping %s: %s", run_dir, exc)
            continue
        row = _config_row(run)
        summary_path = run_dir / "analysis" / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            row.update({key: summary.get(key) for key in _SUMMARY_KEYS})
            for name, tr in (summary.get("transitions") or {}).items():
                row[f"t_top__{name}"] = tr.get("t_top")
                row[f"delta__{name}"] = tr.get("delta")
        rows.append(row)
    return pd.DataFrame(rows)


def early_window_table(root: str | Path, window: str) -> pd.DataFrame:
    """One row per run of early-window features, ready for the predictive comparison.

    ``group`` is the run name without its seed suffix, so that folds can be split by
    configuration rather than by run: seeds of one configuration are near-duplicates and
    splitting across them would leak.
    """
    rows: list[dict] = []
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        run_dir = manifest_path.parent
        summary_path = run_dir / "analysis" / "summary.json"
        if not summary_path.exists():
            continue
        try:
            run = Run(run_dir)
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            logger.warning("skipping %s: %s", run_dir, exc)
            continue
        features = (summary.get("early_window_features") or {}).get(window)
        if not features:
            continue
        name = run.run_name
        row = _config_row(run)
        row["group"] = name.rsplit("_s", 1)[0]
        row["grokking_step"] = summary.get("grokking_step")
        row["diverged"] = summary.get("diverged", False)
        row.update(features)
        rows.append(row)
    return pd.DataFrame(rows)
