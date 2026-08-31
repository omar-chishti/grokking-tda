"""Scale-free Betti profiles: lifetimes over the diameter, dominance as a gap in the barcode."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from grokking_tda.config.schema import HomologyCfg
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.summaries import finite_lifetimes

# A synthetic circle's loop spans 0.84 of the diameter; a grokked embedding's spans about 0.08.
# These constants separate clean shapes, so against a network cloud the counts are conservative
# and the meaningful comparison is null-relative.
GAP_FACTOR = 2.0  # a dominant class is at least this many times the next one down
FLOOR = 0.05  # ... and at least this fraction of the cloud's diameter


def dominant_count(
    lifetimes: np.ndarray, *, gap_factor: float = GAP_FACTOR, floor: float = FLOOR
) -> int:
    ordered = np.sort(np.asarray(lifetimes, dtype=float))[::-1]
    ordered = ordered[ordered > floor]
    if ordered.size == 0:
        return 0
    if ordered.size == 1:
        return 1
    ratios = ordered[:-1] / np.maximum(ordered[1:], 1e-12)
    best = int(np.argmax(ratios))
    return best + 1 if ratios[best] >= gap_factor else 0


def betti_profile(
    points: np.ndarray,
    *,
    maxdim: int = 2,
    gap_factor: float = GAP_FACTOR,
    floor: float = FLOOR,
) -> dict[str, float]:
    cloud = np.asarray(points, dtype=np.float64)
    diameter = float(pdist(cloud).max()) if len(cloud) > 1 else 0.0
    diagrams = compute_persistence(cloud, HomologyCfg(maxdim=maxdim))

    out: dict[str, float] = {"diameter": diameter, "n_points": float(len(cloud))}
    for dim in range(1, maxdim + 1):
        lifetimes = finite_lifetimes(diagrams.get(dim))
        if diameter <= 0 or not lifetimes.size:
            out[f"betti_{dim}"], out[f"life_{dim}"] = 0.0, 0.0
            continue
        normalised = lifetimes / diameter
        out[f"betti_{dim}"] = float(dominant_count(normalised, gap_factor=gap_factor, floor=floor))
        out[f"life_{dim}"] = float(normalised.max())
    return out
