"""Typed, composable configuration: ``schema`` defines it, ``store`` registers it with Hydra."""

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
