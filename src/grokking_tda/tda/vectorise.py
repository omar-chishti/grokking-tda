"""Landscapes and images: a diagram as a vector rather than as one number.

Every topological observable in ``observables.py`` reduces a diagram to a scalar, and a verdict
built on those is a verdict about the reduction as much as about the topology. Landscapes and
images are the standard alternative — stable vectorisations with the same Lipschitz guarantee the
bottleneck distance gives the diagram itself — and they turn the comparison against a Fourier
spectrum into one between two vectors rather than two numbers.

Both are functions of a diagram alone. The sampling grid is the caller's, because a landscape
sampled on its own diagram's range is not comparable to the next checkpoint's.
"""

from __future__ import annotations

import numpy as np

from grokking_tda.tda.summaries import finite_bars


def diagram_extent(diagrams: list[np.ndarray | None]) -> tuple[float, float]:
    """The birth-death range spanning a collection, for use as a shared sampling grid."""
    bars = [b for b in map(finite_bars, diagrams) if b.size]
    if not bars:
        return (0.0, 1.0)
    lo = min(float(b[:, 0].min()) for b in bars)
    hi = max(float(b[:, 1].max()) for b in bars)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def landscape(
    diagram: np.ndarray | None,
    *,
    layers: int = 5,
    resolution: int = 20,
    extent: tuple[float, float],
) -> np.ndarray:
    """The first ``layers`` persistence landscape functions, sampled at ``resolution`` points.

    Bubenik's construction: each bar contributes a tent, and the k-th landscape is the k-th
    largest tent at each sample point. Returns shape ``(layers, resolution)``.
    """
    grid = np.linspace(*extent, resolution)
    bars = finite_bars(diagram)
    if bars.size == 0:
        return np.zeros((layers, resolution))
    births, deaths = bars[:, 0][:, None], bars[:, 1][:, None]
    tents = np.maximum(np.minimum(grid - births, deaths - grid), 0.0)
    ordered = -np.sort(-tents, axis=0)[:layers]
    return np.vstack([ordered, np.zeros((layers - len(ordered), resolution))])


def persistence_image(
    diagram: np.ndarray | None,
    *,
    grid: int = 8,
    sigma: float | None = None,
    extent: tuple[float, float],
    weight: str = "linear",
) -> np.ndarray:
    """Gaussian-kernel image on the birth-persistence grid; shape ``(grid, grid)``.

    Bars are weighted by their own persistence, so the near-diagonal noise that dominates a
    Rips diagram by count contributes almost nothing by mass.
    """
    lo, hi = extent
    span = hi - lo
    sigma = span / grid if sigma is None else sigma
    births = np.linspace(lo, hi, grid)
    lives = np.linspace(0.0, span, grid)
    bars = finite_bars(diagram)
    if bars.size == 0:
        return np.zeros((grid, grid))
    birth, life = bars[:, 0], bars[:, 1] - bars[:, 0]
    mass = life if weight == "linear" else np.ones_like(life)
    to_birth = births[None, :, None] - birth[:, None, None]
    to_life = lives[None, None, :] - life[:, None, None]
    kernel = np.exp(-(to_birth**2 + to_life**2) / (2.0 * sigma**2))
    return np.tensordot(mass, kernel, axes=(0, 0)) / (2.0 * np.pi * sigma**2)
