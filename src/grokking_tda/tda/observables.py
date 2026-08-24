"""Topological observables (registered into the global observable registry).

Persistence is measured in the units of the point cloud, and the embedding's overall
scale drifts across training under weight decay — so a rise in raw H1 persistence can
reflect the cloud growing rather than its shape changing. Every summary is therefore
registered twice: raw, as the reference paper reports it, and divided by the cloud's
own connectivity scale. ``pointcloud_scale`` exposes that scale directly, so the size
of the confound is visible rather than assumed away.
"""

from __future__ import annotations

from grokking_tda.analysis.observable import ObservationContext, register_observable
from grokking_tda.tda.summaries import (
    finite_bars,
    max_persistence,
    n_features,
    persistence_entropy,
    total_persistence,
)


def connectivity_scale(ctx: ObservationContext) -> float:
    """The filtration value at which the cloud becomes connected (largest H0 death).

    A natural scale for the cloud that costs nothing — the H0 diagram is already
    computed — and is exactly linear in the cloud's size, which is the property a
    normaliser needs.
    """
    bars = finite_bars(ctx.diagrams().get(0))
    return float(bars[:, 1].max()) if bars.size else 0.0


def _normalised(ctx: ObservationContext, summary) -> float:
    scale = connectivity_scale(ctx)
    return summary(ctx.diagrams().get(1)) / scale if scale > 0 else float("nan")


@register_observable("h1_max_persistence", direction="rising")
def h1_max_persistence(ctx: ObservationContext) -> float:
    """Longest H1 (loop) lifetime — the headline signature from Tang et al."""
    return max_persistence(ctx.diagrams().get(1))


@register_observable("h1_total_persistence", direction="rising")
def h1_total_persistence(ctx: ObservationContext) -> float:
    """Summed H1 lifetimes — total amount of loop structure."""
    return total_persistence(ctx.diagrams().get(1))


@register_observable("h0_total_persistence", direction="auto")
def h0_total_persistence(ctx: ObservationContext) -> float:
    """Summed H0 lifetimes — a proxy for connected-component fragmentation."""
    return total_persistence(ctx.diagrams().get(0))


@register_observable("h1_num_features", direction="rising")
def h1_num_features(ctx: ObservationContext) -> float:
    """Count of H1 bars above a small lifetime threshold (loop multiplicity)."""
    return float(n_features(ctx.diagrams().get(1), min_persistence=1e-6))


@register_observable("h1_persistence_entropy", direction="auto")
def h1_persistence_entropy(ctx: ObservationContext) -> float:
    """Entropy of the H1 lifetime distribution — low when one loop dominates."""
    return persistence_entropy(ctx.diagrams().get(1))


@register_observable("pointcloud_scale", direction="auto")
def pointcloud_scale(ctx: ObservationContext) -> float:
    """The cloud's connectivity scale — the confound the normalised summaries divide out."""
    return connectivity_scale(ctx)


@register_observable("h1_max_persistence_normalised", direction="rising")
def h1_max_persistence_normalised(ctx: ObservationContext) -> float:
    """Longest H1 lifetime relative to the cloud's own scale (shape, not size)."""
    return _normalised(ctx, max_persistence)


@register_observable("h1_total_persistence_normalised", direction="rising")
def h1_total_persistence_normalised(ctx: ObservationContext) -> float:
    """Summed H1 lifetimes relative to the cloud's own scale."""
    return _normalised(ctx, total_persistence)
