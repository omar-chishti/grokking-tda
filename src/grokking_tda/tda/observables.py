"""Topological observables (registered into the global observable registry)."""

from __future__ import annotations

from grokking_tda.analysis.observable import ObservationContext, register_observable
from grokking_tda.tda.summaries import (
    max_persistence,
    n_features,
    persistence_entropy,
    total_persistence,
)


@register_observable("h1_max_persistence")
def h1_max_persistence(ctx: ObservationContext) -> float:
    """Longest H1 (loop) lifetime — the headline signature from Tang et al."""
    return max_persistence(ctx.diagrams().get(1))


@register_observable("h1_total_persistence")
def h1_total_persistence(ctx: ObservationContext) -> float:
    """Summed H1 lifetimes — total amount of loop structure."""
    return total_persistence(ctx.diagrams().get(1))


@register_observable("h0_total_persistence")
def h0_total_persistence(ctx: ObservationContext) -> float:
    """Summed H0 lifetimes — a proxy for connected-component fragmentation."""
    return total_persistence(ctx.diagrams().get(0))


@register_observable("h1_num_features")
def h1_num_features(ctx: ObservationContext) -> float:
    """Count of H1 bars above a small lifetime threshold (loop multiplicity)."""
    return float(n_features(ctx.diagrams().get(1), min_persistence=1e-6))


@register_observable("h1_persistence_entropy")
def h1_persistence_entropy(ctx: ObservationContext) -> float:
    """Entropy of the H1 lifetime distribution — low when one loop dominates."""
    return persistence_entropy(ctx.diagrams().get(1))
