"""``gtda-analyse`` — observables over a run's snapshots, and the transition.

``--representation`` requires ``--out``: in place it overwrites the embedding analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grokking_tda.analysis import ObservationContext, run_observables
from grokking_tda.analysis.observable import stored_analysis_cfg
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
    cfg = stored_analysis_cfg(run)
    if representation:
        cfg.representation = representation
    if split:
        cfg.representation_split = split
    if observables == "default":
        # the manifest froze whichever observables existed when the run trained
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
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write here instead of <run_dir>/analysis (required for a second representation)",
    )
    parser.add_argument(
        "--reuse-observables",
        action="store_true",
        help="re-derive the summary from the stored observables.csv rather than recomputing "
        "it; for a change to a detector, which leaves the observables untouched",
    )
    parser.add_argument("--acc-threshold", type=float, default=0.9)
    parser.add_argument(
        "--windows",
        default=",".join(str(w) for w in PREREGISTERED_WINDOWS),
        help="comma-separated early-window cutoffs in steps (pre-registered; not t_g)",
    )
    args = parser.parse_args()

    if args.representation and args.out is None:
        raise SystemExit(
            "--representation needs --out: written in place it destroys the embedding analysis"
        )

    run = Run(args.run_dir)
    cfg = _analysis_cfg(run, args.representation, args.split, args.observables)
    out_dir = args.out or (args.run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_observables:
        observables = pd.read_csv(out_dir / "observables.csv")
    else:
        observables = run_observables(run, cfg)
        observables.to_csv(out_dir / "observables.csv", index=False)

    summary = lead_lag(run.metrics, observables, args.observable, args.acc_threshold)
    summary["transitions"] = all_transitions(run.metrics, observables, args.acc_threshold)
    summary["grokking_step_sensitivity"] = grokking_step_sensitivity(run.metrics, observables)
    # divergence is a reported result, not something to infer from an absent table row
    summary["diverged"] = any(
        not np.isfinite(run.metrics[column].to_numpy(dtype=float)).all()
        for column in ("train_loss", "test_loss")
        if column in run.metrics
    )
    windows = tuple(int(w) for w in str(args.windows).split(",") if w.strip())
    summary["early_window_features"] = early_window_feature_grid(
        observables, windows=windows, train_convergence=summary["train_convergence_step"]
    )

    if args.reuse_observables:
        # safe to stop here: neither the velocity nor the PH-dimension depends on a detector
        stored = json.loads((out_dir / "summary.json").read_text())
        if "ph_dimension" in stored:
            summary["ph_dimension"] = stored["ph_dimension"]
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        logger.info("summary re-derived -> %s", out_dir)
        return

    # topological velocity, cheap because the diagrams are cached above; an unreadable
    # snapshot costs its own step, not the whole analysis
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

    # Birdal et al., NeurIPS 2021; needs the dense projected trajectory, not the snapshots
    trajectory = run.trajectory()
    if trajectory is not None:
        traj_steps, traj_points = trajectory
        ph_dim = ph_dimension_over_training(
            traj_steps, traj_points, seed=int(run.config.get("seed", 0))
        )
        ph_dim.to_csv(out_dir / "ph_dimension.csv", index=False)
        # the dimension falls as the model generalises
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
