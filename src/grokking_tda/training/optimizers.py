"""Optimizers, including the OrthoGrad (⊥Grad) intervention from Prieto et al.

OrthoGrad projects each parameter's gradient onto the subspace orthogonal to the
parameter itself (then rescales to preserve gradient norm), which removes the
"naive loss minimisation" direction that merely scales logits. It wraps any base
optimizer, so ``orthograd_adamw`` / ``orthograd_sgd`` are available alongside the
plain optimizers.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import SGD, AdamW, Optimizer

from grokking_tda.config.schema import OptimCfg


class OrthoGrad(Optimizer):
    """Wrap a base optimizer, orthogonalising gradients w.r.t. weights before each step."""

    def __init__(self, base_optimizer: Optimizer, eps: float = 1e-30) -> None:
        self.base = base_optimizer
        self.eps = eps
        # Share state with the base optimizer so checkpointing/inspection just works.
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
    """Build the optimizer named by ``cfg.name`` (optionally OrthoGrad-wrapped)."""
    if cfg.name.startswith("orthograd_"):
        return OrthoGrad(_build_base(cfg.name.removeprefix("orthograd_"), params, cfg))
    return _build_base(cfg.name, params, cfg)
