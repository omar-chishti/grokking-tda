"""Global weight norm — the simplest competing explanation (Omnigrok / LU mechanism).

Computed from the snapshot's weights so it is available to the analysis layer on the
same footing as every other observable (it is also logged live during training).
"""

from __future__ import annotations

import torch

from grokking_tda.analysis.observable import ObservationContext, register_observable


@register_observable("weight_norm")
def weight_norm(ctx: ObservationContext) -> float:
    """L2 norm of all floating-point parameters."""
    total = 0.0
    for tensor in ctx.weights().values():
        if torch.is_floating_point(tensor):
            total += float((tensor.double() ** 2).sum())
    return total**0.5
