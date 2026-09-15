"""The Observable contract: a named ``(context) -> float``, with the direction it moves declared.

Kept out of ``analysis/`` so that the producers (``tda``, ``baselines``) and the consumer of
directions can all import it without a cycle.
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

    Registration happens on import, and a caller that reads a direction first would get ``auto``,
    the direction inferred from the data, which declaring it exists to prevent. The imports are
    local because those packages import this one.
    """
    if OBSERVABLE_DIRECTION:
        return
    import grokking_tda.analysis.task_metrics  # noqa: F401
    import grokking_tda.baselines  # noqa: F401
    import grokking_tda.tda.observables  # noqa: F401
