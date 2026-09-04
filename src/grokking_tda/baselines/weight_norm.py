"""Global weight norm — the simplest competing explanation (Omnigrok / LU mechanism)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grokking_tda.observable import register_observable

if TYPE_CHECKING:  # the context is needed to describe an observable, never to register one
    from grokking_tda.analysis.context import ObservationContext
from grokking_tda.utils.precision import tensor_norm


@register_observable("weight_norm", direction="falling")
def weight_norm(ctx: ObservationContext) -> float:
    return tensor_norm(ctx.weights().values())
