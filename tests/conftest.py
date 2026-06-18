from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """Train a tiny MLP run and return ``(run_dir, trainer)`` for pipeline tests."""
    import torch
    from omegaconf import OmegaConf

    from grokking_tda.artifacts import ArtifactWriter, Manifest
    from grokking_tda.config.schema import DataCfg, ExperimentCfg, ModelCfg, OptimCfg, TrainCfg
    from grokking_tda.data import build_data
    from grokking_tda.models import build_model
    from grokking_tda.training import Trainer

    cfg = OmegaConf.structured(
        ExperimentCfg(
            model=ModelCfg(name="mlp", embedding_dim=32, hidden_dim=64, depth=1),
            data=DataCfg(modulus=11, operation="add", train_fraction=0.6),
            train=TrainCfg(
                steps=300,
                n_snapshots=4,
                metric_every=50,
                capture_representations=False,
                optimizer=OptimCfg(name="adamw", lr=1e-2, weight_decay=0.0),
            ),
            seed=0,
        )
    )
    data = build_data(cfg.data, cfg.seed)
    model = build_model(cfg.model, data.meta)
    run_dir = tmp_path_factory.mktemp("tiny_run")
    writer = ArtifactWriter(run_dir)
    writer.write_manifest(
        Manifest(
            run_name="tiny",
            config=OmegaConf.to_container(cfg, resolve=True),
            env={},
            task_meta=asdict(data.meta),
            created_at="now",
        )
    )
    trainer = Trainer(model, data, cfg, torch.device("cpu"), writer).fit()
    return run_dir, trainer
