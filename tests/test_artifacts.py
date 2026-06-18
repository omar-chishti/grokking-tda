from __future__ import annotations

import numpy as np
import torch

from grokking_tda.artifacts import ArtifactWriter, Manifest, Run


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
