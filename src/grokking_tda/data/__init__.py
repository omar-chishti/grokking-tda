"""Datasets/tasks, exposed through a registry so configs select them by name."""

from __future__ import annotations

from grokking_tda.config.schema import DataCfg
from grokking_tda.data.modular import ModularArithmeticData, build_modular_data
from grokking_tda.registry import Registry

DATASETS: Registry[ModularArithmeticData] = Registry("dataset")
DATASETS.register("modular_arithmetic")(build_modular_data)


def build_data(cfg: DataCfg, seed: int) -> ModularArithmeticData:
    """Build the dataset selected by ``cfg.task`` (deterministic split via ``seed``)."""
    return DATASETS.build(cfg.task, cfg, seed)


__all__ = ["DATASETS", "ModularArithmeticData", "build_data", "build_modular_data"]
