"""Activation capture, in the spirit of TransformerLens HookPoints."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
from torch import nn


class HookPoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name: str | None = None
        self._store: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._store is not None and self.name is not None:
            self._store[self.name] = x.detach()
        return x


class HookedModule(nn.Module):
    def _named_hook_points(self) -> Iterator[tuple[str, HookPoint]]:
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                # the full dotted path: a last-component name collides across blocks
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
        store: dict[str, torch.Tensor] = {}
        with self._capturing(store, names):
            out = self(*args, **kwargs)
        return out, store
