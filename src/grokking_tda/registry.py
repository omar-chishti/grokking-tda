"""A tiny typed registry, so component families stay declarative and configs select by name."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if key in self._factories:
                raise KeyError(f"{self.name!r} registry already has a factory named {key!r}")
            self._factories[key] = factory
            return factory

        return decorator

    def build(self, key: str, *args, **kwargs) -> T:
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
