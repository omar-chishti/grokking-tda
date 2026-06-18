"""Local intrinsic dimension of the representation point cloud (TwoNN estimator).

TwoNN (Facco et al., 2017): for each point, ``mu = r2 / r1`` (ratio of its two
nearest-neighbour distances); the intrinsic dimension is ``N / sum(log mu)``. This
is the geometric baseline against which topology is compared.
"""

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
    mu = mu[mu > 1.0]
    if mu.size == 0:
        return float("nan")
    return float(mu.size / np.log(mu).sum())


@register_observable("lid")
def lid(ctx: ObservationContext) -> float:
    """TwoNN local intrinsic dimension of the analysed point cloud."""
    return two_nn_dimension(ctx.point_cloud())
