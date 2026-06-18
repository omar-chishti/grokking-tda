"""``gtda-plot`` — render vector-first figures from a run's artifacts.

    gtda-plot results/raw/tf_mod97_grok_s0

Produces (in ``<run_dir>/figures/``): training curves, observables-over-time, the
final-snapshot persistence diagram, and a CROCKER plot of the trajectory (>=3 snapshots).
Run ``gtda-analyse`` first for the observables-over-time figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from grokking_tda.analysis.observable import ObservationContext
from grokking_tda.artifacts import Run
from grokking_tda.config.schema import AnalysisCfg
from grokking_tda.evaluation import grokking_step
from grokking_tda.plotting import (
    plot_crocker,
    plot_observables_over_time,
    plot_persistence_diagram,
    plot_training_curves,
)
from grokking_tda.tda.trajectory import crocker_from_diagrams
from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figures for a grokking run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--steps",
        default=None,
        help="comma-separated steps for stage persistence diagrams (nearest snapshot is used)",
    )
    args = parser.parse_args()

    run = Run(args.run_dir)
    fig_dir = args.run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = run.metrics
    transition = grokking_step(metrics)
    if not metrics.empty:
        plot_training_curves(metrics, fig_dir / "training_curves.pdf", transition=transition)

    obs_path = args.run_dir / "analysis" / "observables.csv"
    if obs_path.exists():
        observables = pd.read_csv(obs_path)
        plot_observables_over_time(
            observables, fig_dir / "observables_over_time.pdf", markers={"grok": transition}
        )

    snapshots = run.snapshots()
    if snapshots:
        cfg = OmegaConf.structured(AnalysisCfg)
        if "analysis" in run.config:
            cfg = OmegaConf.merge(cfg, run.config["analysis"])
        ctx = ObservationContext(run, snapshots[-1], cfg)
        plot_persistence_diagram(ctx.diagrams(), fig_dir / "persistence_diagram_final.pdf")

        # Stage diagrams (e.g. before / during / after grokking) on request.
        if args.steps:
            available = [s.step for s in snapshots]
            for wanted in (int(s) for s in str(args.steps).split(",") if s.strip()):
                nearest = min(range(len(available)), key=lambda i, w=wanted: abs(available[i] - w))
                snap = snapshots[nearest]
                stage_ctx = ObservationContext(run, snap, cfg)
                plot_persistence_diagram(
                    stage_ctx.diagrams(),
                    fig_dir / f"persistence_diagram_step_{snap.step:08d}.pdf",
                )

        # CROCKER: topology *of* the trajectory (>=3 snapshots needed to read as evolution).
        if len(snapshots) >= 3:
            try:
                _plot_crocker_figure(run, snapshots, cfg, fig_dir, transition)
            except Exception as exc:
                logger.warning("skipping CROCKER figure: %s", exc)

    logger.info("figures -> %s", fig_dir)


def _plot_crocker_figure(run, snapshots, cfg, fig_dir, transition) -> None:
    # Reuse the per-snapshot diagram cache written by gtda-analyse (same construction).
    diagrams = [ObservationContext(run, s, cfg).diagrams().get(1) for s in snapshots]
    scales, matrix = crocker_from_diagrams(diagrams)
    steps = np.array([s.step for s in snapshots])
    plot_crocker(
        matrix, steps, scales, fig_dir / "crocker_h1.pdf", homology_dim=1, grokking_step=transition
    )


if __name__ == "__main__":
    main()
