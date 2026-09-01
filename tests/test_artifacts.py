"""The artifact store's contract: what a run writes, and what may be read back from it.

Training happens once and analysis happens many times, so everything downstream is a
statement about this directory rather than about the training loop. Three guarantees are
pinned here — a run round-trips, a run directory is never quietly reused, and the derived
layers (the diagram cache, the representation splits, the aggregate) read what was written
rather than something adjacent to it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from grokking_tda.analysis.aggregate import aggregate_runs
from grokking_tda.analysis.observable import (
    ObservationContext,
    diagram_cache_digest,
    stored_analysis_cfg,
)
from grokking_tda.analysis.representations import extract_representation_matrix
from grokking_tda.artifacts import ArtifactWriter, Manifest, Run, prepare_run_dir
from grokking_tda.config.schema import AnalysisCfg, DataCfg
from grokking_tda.data import build_data


def test_writer_reader_roundtrip(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path / "run")
    writer.write_manifest(
        Manifest(
            run_name="r",
            config={"model": {"name": "mlp"}, "seed": 0},
            env={},
            task_meta={"modulus": 5},
            created_at="now",
        )
    )
    writer.append_metric({"step": 0, "train_acc": 0.5})
    writer.append_metric({"step": 10, "train_acc": 1.0})
    writer.write_snapshot(0, {"w": torch.zeros(2, 2)}, {"embedding": np.zeros((5, 3))})

    run = Run(tmp_path / "run")
    assert run.run_name == "r"
    assert run.task_meta["modulus"] == 5
    assert list(run.metrics["train_acc"]) == [0.5, 1.0]

    snapshots = run.snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].step == 0
    assert snapshots[0].representation("embedding").shape == (5, 3)
    assert "w" in snapshots[0].load_weights()


def test_prepare_run_dir_refuses_existing_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        prepare_run_dir(run_dir, overwrite=False)


def test_prepare_run_dir_overwrite_removes_stale_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "snapshots" / "step_00000007").mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")
    (run_dir / "metrics.jsonl").write_text('{"step": 0}\n')
    (run_dir / "trajectory.npz").write_bytes(b"")
    prepare_run_dir(run_dir, overwrite=True)
    # asserting on the absence of everything, rather than on the names known today, is what
    # catches the next artefact somebody adds to a run
    assert list(run_dir.iterdir()) == []
    # A fresh directory passes silently.
    prepare_run_dir(run_dir, overwrite=False)


def test_run_records_lifecycle_events(tiny_run) -> None:
    run_dir, _ = tiny_run
    events = Run(run_dir).events
    assert not events.empty
    assert (events["event"] == "train_end").any()


def test_diagrams_are_cached_to_disk_and_reread(tiny_run) -> None:
    run_dir, _ = tiny_run
    run = Run(run_dir)
    snapshot = run.snapshots()[0]
    cfg = AnalysisCfg()
    first = ObservationContext(run, snapshot, cfg).diagrams()
    cache_files = list((run_dir / "analysis" / "diagrams").glob("step_*.npz"))
    assert cache_files
    second = ObservationContext(run, snapshot, cfg).diagrams()
    assert set(first) == set(second)
    for dim in first:
        assert np.allclose(first[dim], second[dim], equal_nan=True)


def test_a_second_representation_caches_beside_the_first_rather_than_over_it(tiny_run) -> None:
    """Two constructions of one run share a cache directory, so every reader of that
    directory has to select on the digest. `analysis/shape.py` globbed it, and pooled a
    residue table with a landmark subsample of the logits the first time both existed."""
    run_dir, _ = tiny_run
    run = Run(run_dir)
    snapshot = run.snapshots()[0]
    embedding = stored_analysis_cfg(run)
    hidden = stored_analysis_cfg(run)
    hidden.representation = "hidden"
    assert diagram_cache_digest(embedding, 0) != diagram_cache_digest(hidden, 0)

    ObservationContext(run, snapshot, embedding).diagrams()
    ObservationContext(run, snapshot, hidden).diagrams()
    diagrams = run_dir / "analysis" / "diagrams"
    cached = list(diagrams.glob(f"*_{diagram_cache_digest(embedding, 0)}.npz"))
    assert len(cached) == 1
    assert len(list(diagrams.glob("*.npz"))) > len(cached)


def test_representation_split_restricts_rows(tiny_run) -> None:
    run_dir, _ = tiny_run
    run = Run(run_dir)
    snapshot = run.snapshots()[-1]
    data = build_data(DataCfg(**run.config["data"]), run.config["seed"])
    n_test = int((~data.train_mask).sum())
    hidden_all = extract_representation_matrix(run, snapshot, "hidden", split="all")
    hidden_test = extract_representation_matrix(run, snapshot, "hidden", split="test")
    assert hidden_all.shape[0] == len(data)
    assert hidden_test.shape[0] == n_test


def test_aggregate_runs_joins_config_and_summary(tiny_run, tmp_path) -> None:
    run_dir, _ = tiny_run
    table = aggregate_runs(run_dir)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["run"] == "tiny"
    assert row["model"] == "mlp"
    assert row["operation"] == "add"

    # A run with a summary gets its transitions flattened into columns.
    fake = tmp_path / "fake_run"
    (fake / "analysis").mkdir(parents=True)
    manifest = {
        "run_name": "fake",
        "config": {
            "seed": 1,
            "model": {"name": "mlp"},
            "data": {"operation": "add", "modulus": 11, "train_fraction": 0.5},
            "train": {"loss": "softmax_ce", "steps": 10, "optimizer": {"name": "adamw"}},
        },
        "env": {},
        "task_meta": {"modulus": 11},
        "created_at": "now",
    }
    (fake / "manifest.json").write_text(json.dumps(manifest))
    summary = {
        "grokking_step": 20,
        "transitions": {"h1_max_persistence": {"t_top": 30, "delta": -10}},
    }
    (fake / "analysis" / "summary.json").write_text(json.dumps(summary))
    table = aggregate_runs(tmp_path)
    row = table[table["run"] == "fake"].iloc[0]
    assert row["grokking_step"] == 20
    assert row["t_top__h1_max_persistence"] == 30
    assert row["delta__h1_max_persistence"] == -10
