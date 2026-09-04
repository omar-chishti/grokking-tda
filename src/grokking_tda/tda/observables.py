"""Topological observables, registered raw and divided by the cloud's connectivity scale."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grokking_tda.observable import register_observable
from grokking_tda.tda.summaries import (
    connectivity_scale,
    max_persistence,
    n_features,
    persistence_entropy,
    total_persistence,
)

if TYPE_CHECKING:  # the context is needed to describe an observable, never to register one
    from grokking_tda.analysis.context import ObservationContext


def _normalised(ctx: ObservationContext, summary) -> float:
    scale = connectivity_scale(ctx.diagrams())
    return summary(ctx.diagrams().get(1)) / scale if scale > 0 else float("nan")


@register_observable("h1_max_persistence", direction="rising")
def h1_max_persistence(ctx: ObservationContext) -> float:
    """Longest H1 (loop) lifetime — the headline signature from Tang et al."""
    return max_persistence(ctx.diagrams().get(1))


@register_observable("h1_total_persistence", direction="rising")
def h1_total_persistence(ctx: ObservationContext) -> float:
    return total_persistence(ctx.diagrams().get(1))


@register_observable("h0_total_persistence", direction="auto")
def h0_total_persistence(ctx: ObservationContext) -> float:
    return total_persistence(ctx.diagrams().get(0))


@register_observable("h1_num_features", direction="rising")
def h1_num_features(ctx: ObservationContext) -> float:
    return float(n_features(ctx.diagrams().get(1), min_persistence=1e-6))


@register_observable("h1_persistence_entropy", direction="auto")
def h1_persistence_entropy(ctx: ObservationContext) -> float:
    return persistence_entropy(ctx.diagrams().get(1))


@register_observable("pointcloud_scale", direction="auto")
def pointcloud_scale(ctx: ObservationContext) -> float:
    return connectivity_scale(ctx.diagrams())


@register_observable("h1_max_persistence_normalised", direction="rising")
def h1_max_persistence_normalised(ctx: ObservationContext) -> float:
    return _normalised(ctx, max_persistence)


@register_observable("h1_total_persistence_normalised", direction="rising")
def h1_total_persistence_normalised(ctx: ObservationContext) -> float:
    return _normalised(ctx, total_persistence)
