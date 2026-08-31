"""Local intrinsic dimension of the point cloud, by the TwoNN estimator (Facco et al., 2017)."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from grokking_tda.analysis.observable import ObservationContext, register_observable


def two_nn_dimension(points: np.ndarray) -> float:
    x = np.asarray(points, dtype=np.float64)
    n = x.shape[0]
    if n < 3:
        return float("nan")
    distances, _ = NearestNeighbors(n_neighbors=3).fit(x).kneighbors(x)
    r1, r2 = distances[:, 1], distances[:, 2]
    valid = r1 > 0
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]  # a tied pair carries no information about the dimension
    if mu.size == 0:
        return float("nan")
    return float(mu.size / np.log(mu).sum())


@register_observable("lid", direction="falling")
def lid(ctx: ObservationContext) -> float:
    return two_nn_dimension(ctx.point_cloud())
