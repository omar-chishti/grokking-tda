"""Topology *of* the training trajectory: the CROCKER matrix of Betti numbers over time."""

from __future__ import annotations

import numpy as np

from grokking_tda.config.schema import HomologyCfg
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.summaries import finite_bars


def betti_at_scales(diagram: np.ndarray | None, scales: np.ndarray) -> np.ndarray:
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
    cfg = HomologyCfg(maxdim=max(homology_dim, 1))
    rows = []
    for cloud in point_clouds:
        diagrams = compute_persistence(cloud, cfg, metric=metric)
        rows.append(betti_at_scales(diagrams.get(homology_dim), scales))
    return np.vstack(rows) if rows else np.zeros((0, scales.size), dtype=int)


def crocker_from_diagrams(
    diagrams: list[np.ndarray | None], *, n_scales: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    """CROCKER matrix framed to the finite bars' own range, so loops are not rendered as slivers."""
    finite = [bars for bars in map(finite_bars, diagrams) if bars.size]
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
    cfg = HomologyCfg(maxdim=max(homology_dim, 1))
    diagrams = [
        compute_persistence(c, cfg, metric=metric).get(homology_dim) for c in point_clouds
    ]
    return crocker_from_diagrams(diagrams, n_scales=n_scales)
