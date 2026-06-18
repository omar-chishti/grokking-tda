"""Typed, composable configuration layer.

``schema`` defines the dataclasses (validated by Hydra/OmegaConf); ``store``
registers preset nodes and config groups with Hydra's ``ConfigStore`` so that
experiments are selected and overridden declaratively from the CLI:

    gtda-train +experiment=tf_mod97_grok train.optimizer.weight_decay=0.5 seed=1
    gtda-train -m +experiment=tf_mod97_grok seed=0,1,2   # multirun sweep
"""

from grokking_tda.config.schema import (
    AnalysisCfg,
    DataCfg,
    ExperimentCfg,
    HomologyCfg,
    ModelCfg,
    OptimCfg,
    PointCloudCfg,
    TrainCfg,
)
from grokking_tda.config.store import register_configs

__all__ = [
    "AnalysisCfg",
    "DataCfg",
    "ExperimentCfg",
    "HomologyCfg",
    "ModelCfg",
    "OptimCfg",
    "PointCloudCfg",
    "TrainCfg",
    "register_configs",
]
