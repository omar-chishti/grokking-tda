"""``gtda-train`` — compose a config, train, and emit run artifacts.

Examples::

    gtda-train +experiment=smoke
    gtda-train +experiment=tf_mod97_grok seed=1 train.optimizer.weight_decay=0.5
    gtda-train -m +experiment=tf_mod97_grok seed=0,1,2          # local multirun
    gtda-train -m +experiment=tf_mod97_grok seed=0,1,2 \
        hydra/launcher=submitit_slurm                            # cluster sweep
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

from grokking_tda.artifacts import ArtifactWriter, Manifest, prepare_run_dir
from grokking_tda.config.schema import ExperimentCfg
from grokking_tda.config.store import register_configs
from grokking_tda.data import build_data
from grokking_tda.models import build_model
from grokking_tda.training import Trainer
from grokking_tda.utils.env import collect_env_info
from grokking_tda.utils.logging import get_logger
from grokking_tda.utils.seeding import resolve_device, seed_everything

register_configs()
logger = get_logger(__name__)


# Hydra hands back a DictConfig backed by the ExperimentCfg schema; the casts below say
# so once, rather than threading OmegaConf's union types through the whole call graph.
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    logger.info("run=%s device=%s", cfg.run_name, device)

    data = build_data(cfg.data, cfg.seed)
    model = build_model(cfg.model, data.meta)

    run_dir = Path(cfg.output_root) / cfg.run_name
    # Never silently reuse a directory that already holds a run (mixed-run artifacts).
    prepare_run_dir(run_dir, overwrite=bool(cfg.get("overwrite", False)))
    writer = ArtifactWriter(run_dir)
    writer.write_manifest(
        Manifest(
            run_name=cfg.run_name,
            config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
            env=collect_env_info().to_dict(),
            task_meta=asdict(data.meta),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    )

    try:
        Trainer(model, data, cast(ExperimentCfg, cfg), device, writer).fit()
    except Exception as exc:
        # Leave a traceable outcome in the artifact before propagating.
        writer.append_event({"event": "train_end", "status": "failed", "error": repr(exc)})
        logger.exception("training failed for run=%s", cfg.run_name)
        raise
    logger.info("done -> %s", run_dir)


if __name__ == "__main__":
    main()
