"""Scale-free Betti profiles, for the torus prediction of the joint-input representation.

If each operand of a modular sum is carried on its own circle, the representation over
all $p^2$ inputs lies near a product of two circles — a 2-torus, with two dominant $H_1$
classes and one dominant $H_2$ class — which must collapse to a single circle at the
layer where the network commits to $a + b$. Testing that needs degree-2 homology and a
count of *dominant* features, neither of which the per-snapshot observables provide.

Two decisions make the count trustworthy, and both were calibrated against shapes whose
homology is known (``tests/test_betti.py``).

Lifetimes carry the units of the cloud, and those units change by two orders of magnitude
across training, so every lifetime is divided by the cloud's diameter. The connectivity
scale used elsewhere in this package is the wrong denominator here: on a densely sampled
shape it is the point spacing, which makes every bar look dominant.

Dominance is then a *gap* in the sorted barcode rather than a fixed cut. No single
threshold can work: a sphere's longest $H_1$ noise bar (0.18 of its diameter) is longer
than a torus's genuine $H_2$ class (0.15), so a cut that keeps the torus keeps the
sphere's noise too. What distinguishes them is that the torus's classes stand clear of
what follows them and the sphere's do not.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from grokking_tda.config.schema import HomologyCfg
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.summaries import finite_lifetimes

GAP_FACTOR = 2.0  # a dominant class is at least this many times the next one down
FLOOR = 0.05  # ... and at least this fraction of the cloud's diameter

# Calibration caveat, measured rather than assumed. On a well-sampled synthetic circle the
# loop spans 0.84 of the diameter; on a grokked 128-dimensional embedding of 113 residues it
# spans about 0.08, even where Fourier concentration reaches 0.76 and the representation is
# certainly circular. These constants therefore separate clean shapes, not network clouds:
# against real data the counts are conservative and a null-relative comparison (a run's own
# random initialisation, or `tda.significance.random_init_null`) is the meaningful test.


def dominant_count(
    lifetimes: np.ndarray, *, gap_factor: float = GAP_FACTOR, floor: float = FLOOR
) -> int:
    """How many leading bars stand clear of the rest, given lifetimes already normalised."""
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
    """Dominant-feature counts and diameter-normalised lifetimes, per homology degree."""
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
