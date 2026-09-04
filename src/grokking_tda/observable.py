"""The Observable contract: a named ``(context) -> float``, with the direction it moves declared.

Kept out of ``analysis/`` deliberately. Importing anything from that package registers the
built-in observables, which imports ``tda`` and ``baselines``, which would in turn have to import
the package they are being registered into. This module depends on nothing but the registry, so
the producers of observables and the consumer of their directions can both reach it.
"""

from __future__ import annotations

from grokking_tda.registry import Registry

OBSERVABLES: Registry[float] = Registry("observable")

# Declared at registration: inferring it from first and last value misreads anything
# non-monotone, and a wrong direction enters the lead-lag results silently
OBSERVABLE_DIRECTION: dict[str, str] = {}


def register_observable(name: str, *, direction: str = "rising"):
    """Register an ``(ctx) -> float`` observable, declaring which way it moves at the transition."""
    if direction not in {"rising", "falling", "auto"}:
        raise ValueError(f"unknown direction {direction!r}")
    OBSERVABLE_DIRECTION[name] = direction
    return OBSERVABLES.register(name)


def ensure_builtins() -> None:
    """Register the observables this package ships, if nothing has yet.

    Registration is a side effect of importing the modules that define them, so a caller who
    reaches a direction without having imported them reads an empty table and gets ``auto`` —
    which resolves the direction from the data being measured, the one thing the declaration
    exists to prevent. Every entry point that reads the table calls this first. The imports sit
    inside the function because they point back at packages that import this one.
    """
    if OBSERVABLE_DIRECTION:
        return
    import grokking_tda.analysis.task_metrics  # noqa: F401
    import grokking_tda.baselines  # noqa: F401
    import grokking_tda.tda.observables  # noqa: F401
