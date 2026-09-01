"""Fourier concentration of the residue-embedding matrix (Nanda et al.).

The residue-axis transform is basis-bound: mul and div arrange their circle by discrete log,
so it is blind to a loop persistent homology still sees. The ``_group`` variant reorders by
powers of a primitive root, and is the fair baseline there.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from grokking_tda.analysis.observable import ObservationContext, register_observable


def _concentration(embedding: np.ndarray, top_k: int) -> float:
    w = np.asarray(embedding, dtype=np.float64)
    w = w - w.mean(axis=0, keepdims=True)  # drop DC by centring along residues
    spectrum = np.fft.rfft(w, axis=0)
    power = (np.abs(spectrum) ** 2).sum(axis=1)
    if power.size > 0:
        power[0] = 0.0  # ignore residual DC
    total = power.sum()
    if total <= 0:
        return 0.0
    k = min(top_k, power.size)
    top = np.partition(power, -k)[-k:].sum()
    return float(top / total)


@cache
def _primitive_root(p: int) -> int | None:
    if p < 3:
        return None
    # trial division; p is small here
    factors: set[int] = set()
    m, d = p - 1, 2
    while d * d <= m:
        while m % d == 0:
            factors.add(d)
            m //= d
        d += 1
    if m > 1:
        factors.add(m)
    for g in range(2, p):
        if pow(g, p - 1, p) != 1:  # Fermat fails => p is not prime
            return None
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    return None


def _discrete_log_order(p: int) -> list[int] | None:
    g = _primitive_root(p)
    if g is None:
        return None
    return [pow(g, k, p) for k in range(p - 1)]


def _group_concentration(ctx: ObservationContext, top_k: int) -> float:
    order = _discrete_log_order(ctx.modulus)
    if order is None:
        return float("nan")
    return _concentration(ctx.embedding_matrix()[order], top_k=top_k)


@register_observable("fourier_concentration", direction="rising")
def fourier_concentration(ctx: ObservationContext) -> float:
    return _concentration(ctx.embedding_matrix(), top_k=5)


@register_observable("fourier_concentration_group", direction="rising")
def fourier_concentration_group(ctx: ObservationContext) -> float:
    """Concentration after discrete-log reordering: the fair baseline for mul and div."""
    return _group_concentration(ctx, top_k=5)


# Concentration rises monotonically in k, so the strongest competitor is chosen downstream
FOURIER_K_SWEEP = (1, 2, 3, 5, 10, 20)

for _k in FOURIER_K_SWEEP:
    register_observable(f"fourier_concentration_k{_k}", direction="rising")(
        lambda ctx, k=_k: _concentration(ctx.embedding_matrix(), top_k=k)
    )
    register_observable(f"fourier_concentration_group_k{_k}", direction="rising")(
        lambda ctx, k=_k: _group_concentration(ctx, top_k=k)
    )
