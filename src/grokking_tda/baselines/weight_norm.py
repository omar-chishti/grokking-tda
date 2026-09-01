"""Global weight norm — the simplest competing explanation (Omnigrok / LU mechanism)."""

from __future__ import annotations

from grokking_tda.analysis.observable import ObservationContext, register_observable
from grokking_tda.utils.precision import tensor_norm


@register_observable("weight_norm", direction="falling")
def weight_norm(ctx: ObservationContext) -> float:
    return tensor_norm(ctx.weights().values())
