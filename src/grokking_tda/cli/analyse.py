"""``gtda-analyse`` — compute observables over a run's snapshots and locate the transition.

    gtda-analyse results/raw/<run>
    gtda-analyse <run_dir> --representation hidden --split test
    gtda-analyse <run_dir> --observables h1_max_persistence,test_acc_novel

Writes into ``<run_dir>/analysis/``:

- ``observables.csv`` — every configured observable per snapshot;
- ``summary.json`` — grokking/train-convergence steps, the headline observable's
  signed lead/lag, ``transitions`` (t_top and delta for *every* observable), and
  ``early_window_features`` at the pre-registered windows (plus ``tc``);
- ``trajectory_distance.csv`` — distance between consecutive snapshots' H1 diagrams
  (the topological velocity of the trajectory), when >= 3 snapshots exist;
- ``ph_dimension.csv`` — PH-dimension in a sliding window along the optimisation
  path, for runs that recorded a dense projected trajectory;
- ``diagrams/`` — cached per-snapshot persistence diagrams (reused by plotting).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from grokking_tda.analysis import ObservationContext, run_observables
from grokking_tda.artifacts import Run
from grokking_tda.config.schema import AnalysisCfg
from grokking_tda.evaluation import (
    all_transitions,
    early_window_feature_grid,
    grokking_step_sensitivity,
    lead_lag,
)
from grokking_tda.evaluation.predictive import PREREGISTERED_WINDOWS
from grokking_tda.evaluation.transitions import transition_step
from grokking_tda.tda.distances import trajectory_velocity
from grokking_tda.tda.phdim import ph_dimension_over_training
from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)


def _analysis_cfg(run: Run, representation: str | None, split: str | None, observables: str | None):
    cfg = OmegaConf.structured(AnalysisCfg)
    if "analysis" in run.config:
        cfg = OmegaConf.merge(cfg, run.config["analysis"])
    if representation:
        cfg.representation = representation
    if split:
        cfg.representation_split = split
    if observables == "default":
        # The run's manifest froze whichever observables existed when it trained;
        # re-analysis should be free to use every one the code now provides.
        cfg.observables = list(AnalysisCfg().observables)
    elif observables:
        cfg.observables = [name.strip() for name in observables.split(",") if name.strip()]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a grokking run with TDA + baselines.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--representation", default=None, help="embedding | hidden | logits")
    parser.add_argument("--split", default=None, help="all | train | test (hidden/logit rows)")
    parser.add_argument("--observable", default="h1_max_persistence")
    parser.add_argument(
        "--observables",
        default="default",
        help="comma-separated observable names, or 'default' for every one the code provides "
        "(the run's manifest only records those that existed when it trained)",
    )
    parser.add_argument("--acc-threshold", type=float, default=0.9)
    parser.add_argument(
        "--windows",
        default=",".join(str(w) for w in PREREGISTERED_WINDOWS),
        help="comma-separated early-window cutoffs in steps (pre-registered; not t_g)",
    )
    args = parser.parse_args()

    run = Run(args.run_dir)
    cfg = _analysis_cfg(run, args.representation, args.split, args.observables)
    observables = run_observables(run, cfg)

    out_dir = args.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    observables.to_csv(out_dir / "observables.csv", index=False)

    summary = lead_lag(run.metrics, observables, args.observable, args.acc_threshold)
    summary["transitions"] = all_transitions(run.metrics, observables, args.acc_threshold)
    summary["grokking_step_sensitivity"] = grokking_step_sensitivity(run.metrics, observables)
    # Divergence under an intervention is a reported result, so it is recorded here
    # rather than inferred later from a run's absence from a table.
    summary["diverged"] = any(
        not np.isfinite(run.metrics[column].to_numpy(dtype=float)).all()
        for column in ("train_loss", "test_loss")
        if column in run.metrics
    )
    windows = tuple(int(w) for w in str(args.windows).split(",") if w.strip())
    summary["early_window_features"] = early_window_feature_grid(
        observables, windows=windows, train_convergence=summary["train_convergence_step"]
    )

    # Topological velocity of the trajectory (cheap: diagrams are disk-cached above).
    # An unreadable snapshot — a truncated transfer, an interrupted run — must cost its
    # own step, not the whole analysis, exactly as in run_observables.
    snapshots = run.snapshots()
    steps, diagrams = [], []
    for snapshot in snapshots:
        try:
            diagrams.append(ObservationContext(run, snapshot, cfg).diagrams().get(1))
            steps.append(snapshot.step)
        except Exception as exc:
            logger.warning("skipping snapshot %d in trajectory velocity: %s", snapshot.step, exc)
    if len(steps) >= 3:
        trajectory_velocity(steps, diagrams).to_csv(
            out_dir / "trajectory_distance.csv", index=False
        )

    # PH-dimension of the optimisation path itself (Birdal et al., NeurIPS 2021).
    # Only runs that recorded a dense projected trajectory can support this: the
    # snapshot schedule gives ~10^2 iterates where the estimator needs ~10^3.
    trajectory = run.trajectory()
    if trajectory is not None:
        traj_steps, traj_points = trajectory
        ph_dim = ph_dimension_over_training(
            traj_steps, traj_points, seed=int(run.config.get("seed", 0))
        )
        ph_dim.to_csv(out_dir / "ph_dimension.csv", index=False)
        # PH-dimension falls as the model generalises, so the transition is a fall.
        t_ph = transition_step(
            ph_dim["step"].to_numpy(), ph_dim["ph_dim"].to_numpy(), direction="falling"
        )
        t_g = summary["grokking_step"]
        summary["ph_dimension"] = {
            "t_ph": t_ph,
            "delta": (t_g - t_ph) if (t_g is not None and t_ph is not None) else None,
            "first": float(ph_dim["ph_dim"].iloc[0]) if len(ph_dim) else None,
            "last": float(ph_dim["ph_dim"].iloc[-1]) if len(ph_dim) else None,
            "min": float(ph_dim["ph_dim"].min()) if len(ph_dim) else None,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info("analysis -> %s", out_dir)
    headline = {
        k: v for k, v in summary.items() if k not in ("early_window_features", "transitions")
    }
    print(json.dumps(headline, indent=2, default=str))


if __name__ == "__main__":
    main()
