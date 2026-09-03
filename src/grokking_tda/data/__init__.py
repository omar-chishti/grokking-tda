"""Datasets/tasks, exposed through a registry so configs select them by name."""

from __future__ import annotations

from grokking_tda.config.schema import DataCfg
from grokking_tda.data.modular import ModularArithmeticData, build_modular_data
from grokking_tda.data.multiop import build_multiop_data
from grokking_tda.data.permutation import build_permutation_data
from grokking_tda.registry import Registry

DATASETS: Registry[ModularArithmeticData] = Registry("dataset")
DATASETS.register("modular_arithmetic")(build_modular_data)
DATASETS.register("permutation_group")(build_permutation_data)
DATASETS.register("modular_multiop")(build_multiop_data)


def build_data(cfg: DataCfg, seed: int) -> ModularArithmeticData:
    return DATASETS.build(cfg.task, cfg, seed)


__all__ = [
    "DATASETS",
    "ModularArithmeticData",
    "build_data",
    "build_modular_data",
    "build_multiop_data",
    "build_permutation_data",
]
