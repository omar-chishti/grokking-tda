"""Hookable models, exposed through a registry.

Every model implements ``forward(tokens) -> logits`` and ``embedding_matrix()``
(the residue-embedding point cloud), and inherits ``run_with_cache`` from
:class:`~grokking_tda.models.hooks.HookedModule`. This uniform interface is what
lets the TDA layer treat architectures interchangeably.
"""

from __future__ import annotations

from torch import nn

from grokking_tda.config.schema import ModelCfg
from grokking_tda.data.modular import TaskMeta
from grokking_tda.models.mlp import GrokkingMLP, build_mlp
from grokking_tda.models.transformer import GrokkingTransformer, build_transformer
from grokking_tda.registry import Registry

MODELS: Registry[nn.Module] = Registry("model")
MODELS.register("transformer")(build_transformer)
MODELS.register("mlp")(build_mlp)


def build_model(cfg: ModelCfg, meta: TaskMeta) -> nn.Module:
    """Build the model selected by ``cfg.name``, sized to the task ``meta``."""
    return MODELS.build(cfg.name, cfg, meta)


__all__ = [
    "MODELS",
    "GrokkingMLP",
    "GrokkingTransformer",
    "build_model",
    "build_mlp",
    "build_transformer",
]

