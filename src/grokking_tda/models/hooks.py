"""A minimal activation-capture system, in the spirit of TransformerLens HookPoints.

A ``HookPoint`` is an identity module placed at a named location in the forward
pass. Normally it is a no-op; inside ``run_with_cache`` it records its input into a
cache dict. This lets the analysis layer extract internal representations *by name*
("hook_resid_final", "hook_mlp_out", ...) without the model or the engine knowing
anything about TDA.

    logits, cache = model.run_with_cache(tokens, names=["hook_resid_final"])
    point_cloud = cache["hook_resid_final"]
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
from torch import nn


class HookPoint(nn.Module):
    """Identity module that optionally records its input into an attached store."""

    def __init__(self) -> None:
        super().__init__()
        self.name: str | None = None
        self._store: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._store is not None and self.name is not None:
            self._store[self.name] = x.detach()
        return x


class HookedModule(nn.Module):
    """Mixin giving a module ``run_with_cache`` over its named ``HookPoint``s."""

    def _named_hook_points(self) -> Iterator[tuple[str, HookPoint]]:
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                # Use the full dotted module path as the hook name. Top-level hooks keep
                # their bare attribute name ("hook_resid_final"); per-block hooks become
                # unambiguous ("blocks.0.hook_attn_out" vs "blocks.1.hook_attn_out"),
                # which a last-component name would collide on for n_layers >= 2.
                module.name = name if module.name is None else module.name
                yield module.name, module

    @contextmanager
    def _capturing(self, store: dict[str, torch.Tensor], names: Iterable[str] | None):
        wanted = set(names) if names is not None else None
        active: list[HookPoint] = []
        for hook_name, hp in self._named_hook_points():
            if wanted is None or hook_name in wanted:
                hp._store = store
                active.append(hp)
        try:
            yield
        finally:
            for hp in active:
                hp._store = None

    def run_with_cache(
        self,
        *args,
        names: Iterable[str] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass that also returns a dict of captured activations."""
        store: dict[str, torch.Tensor] = {}
        with self._capturing(store, names):
            out = self(*args, **kwargs)
        return out, store
