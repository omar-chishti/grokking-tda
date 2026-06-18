from __future__ import annotations

from grokking_tda.analysis import run_observables
from grokking_tda.artifacts import Run
from grokking_tda.config.schema import AnalysisCfg


def test_engine_learns_and_writes_artifacts(tiny_run) -> None:
    run_dir, trainer = tiny_run
    # A tiny full-batch run should at least memorise the training set.
    assert trainer.last_metrics["train_acc"] > 0.5
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "snapshots" / "index.json").exists()


def test_observables_run_over_snapshots(tiny_run) -> None:
    run_dir, _ = tiny_run
    run = Run(run_dir)
    cfg = AnalysisCfg(observables=["h1_max_persistence", "weight_norm", "lid"])
    df = run_observables(run, cfg)
    assert {"step", "h1_max_persistence", "weight_norm", "lid"} <= set(df.columns)
    assert len(df) >= 2
    # weight norm is a non-negative scalar
    assert (df["weight_norm"] >= 0).all()
