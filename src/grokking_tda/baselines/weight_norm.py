"""Global weight norm — the simplest competing explanation (Omnigrok / LU mechanism)."""

from __future__ import annotations

import torch

from grokking_tda.analysis.observable import ObservationContext, register_observable


@register_observable("weight_norm", direction="falling")
def weight_norm(ctx: ObservationContext) -> float:
    total = 0.0
    for tensor in ctx.weights().values():
        if torch.is_floating_point(tensor):
            total += float((tensor.double() ** 2).sum())
    return total**0.5
