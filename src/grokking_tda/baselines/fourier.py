"""Fourier concentration of the residue-embedding matrix.

In modular addition the generalising embedding becomes approximately a circle —
i.e. dominated by a few Fourier modes along the residue axis. This scalar measures
that: the fraction of spectral power carried by the ``top_k`` non-DC frequencies.
It rises as the representation becomes periodic, and is the diagnostic a topological
H1 signal must be shown to beat (or to lead in time). See Nanda et al.

The residue-axis DFT is **basis-bound**: for multiplication/division the grokked
circle is arranged by *discrete log*, so it is invisible to this measure while
persistent homology (permutation-invariant) still sees the loop. The
``fourier_concentration_group`` variant reorders rows by powers of a primitive root
(equivalently, measures concentration over multiplicative characters), giving the
*fair* baseline for those operations — without it, the PH-vs-Fourier comparison on
mul/div is rigged in topology's favour.
"""

from __future__ import annotations

import numpy as np

from grokking_tda.analysis.observable import ObservationContext, register_observable


def _concentration(embedding: np.ndarray, top_k: int) -> float:
    w = np.asarray(embedding, dtype=np.float64)
    w = w - w.mean(axis=0, keepdims=True)  # drop DC by centring along residues
    spectrum = np.fft.rfft(w, axis=0)  # DFT along the residue axis
    power = (np.abs(spectrum) ** 2).sum(axis=1)  # power per frequency (summed over dims)
    if power.size > 0:
        power[0] = 0.0  # ignore residual DC
    total = power.sum()
    if total <= 0:
        return 0.0
    k = min(top_k, power.size)
    top = np.partition(power, -k)[-k:].sum()
    return float(top / total)


def _primitive_root(p: int) -> int | None:
    """Smallest primitive root mod prime ``p`` (None if ``p`` is not prime / p < 3)."""
    if p < 3:
        return None
    # Factor p-1 by trial division (p is small in this benchmark).
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
    """Residues 1..p-1 ordered as powers of the smallest primitive root."""
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
    """Power fraction in the top-5 Fourier modes of the embedding."""
    return _concentration(ctx.embedding_matrix(), top_k=5)


@register_observable("fourier_concentration_group", direction="rising")
def fourier_concentration_group(ctx: ObservationContext) -> float:
    """Top-5 concentration after discrete-log reordering (multiplicative characters).

    The fair Fourier baseline for mul/div: a circle arranged by discrete log is flat
    under the residue-axis DFT but periodic in this ordering. Residue 0 is excluded
    (it sits outside the multiplicative group). NaN if the modulus is not prime.
    """
    return _group_concentration(ctx, top_k=5)


# A claim that topology beats Fourier must not turn on an arbitrary k, so the whole
# family is recorded and the comparison is made against whichever member tracks the
# transition best. Concentration rises monotonically with k, so the strongest
# competitor cannot be picked per snapshot — it is chosen downstream, on the series.
FOURIER_K_SWEEP = (1, 2, 3, 5, 10, 20)

for _k in FOURIER_K_SWEEP:
    register_observable(f"fourier_concentration_k{_k}", direction="rising")(
        lambda ctx, k=_k: _concentration(ctx.embedding_matrix(), top_k=k)
    )
    register_observable(f"fourier_concentration_group_k{_k}", direction="rising")(
        lambda ctx, k=_k: _group_concentration(ctx, top_k=k)
    )
