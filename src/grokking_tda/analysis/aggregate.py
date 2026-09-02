"""Aggregate many runs into one tidy table: config keys joined with each summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grokking_tda.analysis.identity import condition_key, is_replicate
from grokking_tda.artifacts.reader import Run
from grokking_tda.evaluation.predictive import window_end_step
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
    """Early-window features per run; ``group`` is the configuration, so folds cannot leak.

    A dense or trajectory re-run is the *same optimisation path* as its main-programme twin —
    same model, task, weight decay and seed, differing only in what was recorded — so it is
    dropped rather than allowed into a second group.
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
        if is_replicate(run.run_name, run.config):
            continue
        row = _config_row(run)
        row["group"] = condition_key(run.config)
        row["grokking_step"] = summary.get("grokking_step")
        row["window_step"] = window_end_step(window, summary.get("train_convergence_step"))
        row["diverged"] = summary.get("diverged", False)
        row.update(features)
        rows.append(row)
    return pd.DataFrame(rows)
