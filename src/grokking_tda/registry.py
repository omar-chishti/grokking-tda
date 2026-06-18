"""A tiny, typed registry used to make the framework declarative and extensible.

Every pluggable component family — models, datasets, optimizers, losses,
observables — owns a ``Registry``. Components register themselves by name and
configs select them by that name, so adding a new component never requires
touching the engine or the call sites:

    from grokking_tda.registry import Registry

    MODELS: Registry[nn.Module] = Registry("model")

    @MODELS.register("transformer")
    def build_transformer(cfg) -> nn.Module:
        ...

    model = MODELS.build("transformer", cfg)   # resolved at runtime from config

This mirrors how larger research stacks (e.g. fvcore/detectron2, timm) keep a
small composable core while allowing an open set of components.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps string keys to factory callables for one component family."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that registers a factory under ``key``."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if key in self._factories:
                raise KeyError(f"{self.name!r} registry already has a factory named {key!r}")
            self._factories[key] = factory
            return factory

        return decorator

    def build(self, key: str, *args, **kwargs) -> T:
        """Instantiate the component registered under ``key``."""
        try:
            factory = self._factories[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown {self.name} {key!r}; available: {sorted(self._factories)}"
            ) from exc
        return factory(*args, **kwargs)

    def keys(self) -> Iterable[str]:
        return self._factories.keys()

    def __contains__(self, key: object) -> bool:
        return key in self._factories
