"""``gtda-analyse`` — compute observables over a run's snapshots and locate the transition.

    gtda-analyse results/raw/<run>
    gtda-analyse <run_dir> --representation hidden --split test

Writes into ``<run_dir>/analysis/``:

- ``observables.csv`` — every configured observable per snapshot;
- ``summary.json`` — grokking/train-convergence steps, the headline observable's
  signed lead/lag, ``transitions`` (t_top and delta for *every* observable), and
  ``early_window_features`` at the pre-registered windows (plus ``tc``);
- ``trajectory_distance.csv`` — distance between consecutive snapshots' H1 diagrams
  (the topological velocity of the trajectory), when >= 3 snapshots exist;
- ``diagrams/`` — cached per-snapshot persistence diagrams (reused by plotting).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from grokking_tda.analysis import ObservationContext, run_observables
from grokking_tda.artifacts import Run
from grokking_tda.config.schema import AnalysisCfg
from grokking_tda.evaluation import (
    all_transitions,
    early_window_feature_grid,
    lead_lag,
)
from grokking_tda.evaluation.predictive import PREREGISTERED_WINDOWS
from grokking_tda.tda.distances import trajectory_velocity
from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)


def _analysis_cfg(run: Run, representation: str | None, split: str | None):
    cfg = OmegaConf.structured(AnalysisCfg)
    if "analysis" in run.config:
        cfg = OmegaConf.merge(cfg, run.config["analysis"])
    if representation:
        cfg.representation = representation
    if split:
        cfg.representation_split = split
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a grokking run with TDA + baselines.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--representation", default=None, help="embedding | hidden | logits")
    parser.add_argument("--split", default=None, help="all | train | test (hidden/logit rows)")
    parser.add_argument("--observable", default="h1_max_persistence")
    parser.add_argument("--acc-threshold", type=float, default=0.9)
    parser.add_argument(
        "--windows",
        default=",".join(str(w) for w in PREREGISTERED_WINDOWS),
        help="comma-separated early-window cutoffs in steps (pre-registered; not t_g)",
    )
    args = parser.parse_args()

    run = Run(args.run_dir)
    cfg = _analysis_cfg(run, args.representation, args.split)
    observables = run_observables(run, cfg)

    out_dir = args.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    observables.to_csv(out_dir / "observables.csv", index=False)

    summary = lead_lag(run.metrics, observables, args.observable, args.acc_threshold)
    summary["transitions"] = all_transitions(run.metrics, observables, args.acc_threshold)
    windows = tuple(int(w) for w in str(args.windows).split(",") if w.strip())
    summary["early_window_features"] = early_window_feature_grid(
        observables, windows=windows, train_convergence=summary["train_convergence_step"]
    )

    # Topological velocity of the trajectory (cheap: diagrams are disk-cached above).
    snapshots = run.snapshots()
    if len(snapshots) >= 3:
        diagrams = [ObservationContext(run, s, cfg).diagrams().get(1) for s in snapshots]
        velocity = trajectory_velocity([s.step for s in snapshots], diagrams)
        velocity.to_csv(out_dir / "trajectory_distance.csv", index=False)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info("analysis -> %s", out_dir)
    headline = {
        k: v for k, v in summary.items() if k not in ("early_window_features", "transitions")
    }
    print(json.dumps(headline, indent=2, default=str))


if __name__ == "__main__":
    main()
