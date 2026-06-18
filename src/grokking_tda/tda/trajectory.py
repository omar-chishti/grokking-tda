"""Trajectory topology — topology *of* the training trajectory, not just *over* it.

The reference paper computes a scalar per snapshot and plots it over time. The
project brief asks for the topology of the trajectory as an evolving object. The
CROCKER plot (Contour Realization Of Computed k-dimensional hole Evolution) is a
concrete, well-founded first instrument: a ``(time x scale)`` matrix of Betti
numbers, i.e. how many H_k features are alive at each filtration scale at each
training step. Vineyards / zigzag persistence are natural follow-ups.

This is the headline original extension; kept deliberately small and dependency-light
(it reuses the same ripser backend) so it can graduate from scaffold to result.
"""

from __future__ import annotations

import numpy as np

from grokking_tda.config.schema import HomologyCfg
from grokking_tda.tda.homology import compute_persistence


def betti_at_scales(diagram: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Number of bars in ``diagram`` alive at each scale (birth <= s < death)."""
    scales = np.asarray(scales)
    if diagram is None or diagram.size == 0:
        return np.zeros(scales.shape, dtype=int)
    births = diagram[:, 0][:, None]
    deaths = diagram[:, 1][:, None]
    alive = (births <= scales[None, :]) & (scales[None, :] < deaths)
    return alive.sum(axis=0).astype(int)


def crocker_matrix(
    point_clouds: list[np.ndarray],
    scales: np.ndarray,
    *,
    homology_dim: int = 1,
    metric: str = "euclidean",
) -> np.ndarray:
    """Build the ``(n_snapshots, n_scales)`` Betti-``homology_dim`` CROCKER matrix."""
    cfg = HomologyCfg(maxdim=max(homology_dim, 1))
    rows = []
    for cloud in point_clouds:
        diagrams = compute_persistence(cloud, cfg, metric=metric)
        rows.append(betti_at_scales(diagrams.get(homology_dim), scales))
    return np.vstack(rows) if rows else np.zeros((0, scales.size), dtype=int)


def crocker_from_diagrams(
    diagrams: list[np.ndarray | None], *, n_scales: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    """CROCKER matrix from precomputed diagrams, scale grid framed to their own bars.

    The scale window is ``[min birth, max death]`` of the finite bars across snapshots.
    This both keeps an H1 grid from being swamped by large H0 deaths and zooms onto the
    band where loops actually live — bars in high-dimensional embedding clouds are
    short-lived relative to their birth/death scale, so a ``[0, max death]`` grid
    renders them as invisible slivers. Returns ``(scales, matrix)``.

    Taking diagrams (rather than clouds) lets callers reuse the per-snapshot diagram
    cache instead of recomputing persistence.
    """
    finite = [d[np.isfinite(d[:, 1])] for d in diagrams if d is not None and d.size]
    finite = [f for f in finite if f.size]
    lo = min((f[:, 0].min() for f in finite), default=0.0)
    hi = max((f[:, 1].max() for f in finite), default=1.0)
    pad = 0.02 * (hi - lo) if hi > lo else max(hi, 1.0) * 0.02
    scales = np.linspace(max(0.0, float(lo) - pad), float(hi) + pad, n_scales)
    rows = [betti_at_scales(d, scales) for d in diagrams]
    matrix = np.vstack(rows) if rows else np.zeros((0, n_scales), dtype=int)
    return scales, matrix


def auto_crocker(
    point_clouds: list[np.ndarray],
    *,
    homology_dim: int = 1,
    metric: str = "euclidean",
    n_scales: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """CROCKER matrix computed from point clouds (see ``crocker_from_diagrams``)."""
    cfg = HomologyCfg(maxdim=max(homology_dim, 1))
    diagrams = [
        compute_persistence(c, cfg, metric=metric).get(homology_dim) for c in point_clouds
    ]
    return crocker_from_diagrams(diagrams, n_scales=n_scales)
