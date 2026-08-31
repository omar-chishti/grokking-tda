"""A hookable embedding-MLP baseline, with the transformer's ``embedding_matrix`` interface."""

from __future__ import annotations

import torch
from torch import nn

from grokking_tda.data.modular import TaskMeta
from grokking_tda.models.hooks import HookedModule, HookPoint

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class GrokkingMLP(HookedModule):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        depth: int = 2,
        act: str = "relu",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        act_cls = _ACTIVATIONS[act]
        layers: list[nn.Module] = []
        current = embedding_dim * 2  # two operands concatenated
        self.hook_hidden = HookPoint()
        for _ in range(depth):
            layers += [nn.Linear(current, hidden_dim), act_cls()]
            current = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.unembed = nn.Linear(current, num_classes)
        self.modulus = num_classes
        self.hidden_hook = "hook_hidden"  # what AnalysisCfg.representation="hidden" reads

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        operands = tokens[:, :2]  # ignore "=" token
        embedded = self.embed(operands).flatten(start_dim=1)
        hidden = self.hook_hidden(self.mlp(embedded))
        return self.unembed(hidden)

    def embedding_matrix(self) -> torch.Tensor:
        # the p residue rows; the slice drops the "=" token at index p
        return self.embed.weight[: self.modulus].detach()


def build_mlp(model_cfg, meta: TaskMeta) -> GrokkingMLP:
    return GrokkingMLP(
        vocab_size=meta.vocab_size,
        num_classes=meta.num_classes,
        embedding_dim=model_cfg.embedding_dim,
        hidden_dim=model_cfg.hidden_dim,
        depth=model_cfg.depth,
        act=model_cfg.act,
    )
