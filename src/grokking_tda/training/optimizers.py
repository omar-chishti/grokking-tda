"""Optimizers, including OrthoGrad (Prieto et al.): it projects out the logit-scaling direction."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import SGD, AdamW, Optimizer

from grokking_tda.config.schema import OptimCfg


class OrthoGrad(Optimizer):
    def __init__(self, base_optimizer: Optimizer, eps: float = 1e-30) -> None:
        self.base = base_optimizer
        self.eps = eps
        # shared with the base optimizer, so checkpointing works
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                w = p.view(-1)
                g = p.grad.view(-1)
                proj = torch.dot(w, g) / (torch.dot(w, w) + self.eps)
                g_orth = g - proj * w
                g_orth.mul_(g.norm() / (g_orth.norm() + self.eps))
                p.grad.copy_(g_orth.view_as(p.grad))
        return self.base.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, state_dict):
        self.base.load_state_dict(state_dict)


def _build_base(name: str, params: Iterable[torch.nn.Parameter], cfg: OptimCfg) -> Optimizer:
    if name == "adamw":
        return AdamW(
            params,
            lr=cfg.lr,
            betas=(cfg.betas[0], cfg.betas[1]),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    if name == "sgd":
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown base optimizer {name!r}; choices: adamw, sgd")


def build_optimizer(params: Iterable[torch.nn.Parameter], cfg: OptimCfg) -> Optimizer:
    if cfg.name.startswith("orthograd_"):
        return OrthoGrad(_build_base(cfg.name.removeprefix("orthograd_"), params, cfg))
    return _build_base(cfg.name, params, cfg)
