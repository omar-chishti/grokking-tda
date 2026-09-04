"""Local intrinsic dimension of the point cloud, by the TwoNN estimator (Facco et al., 2017)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

from grokking_tda.observable import register_observable

if TYPE_CHECKING:  # the context is needed to describe an observable, never to register one
    from grokking_tda.analysis.context import ObservationContext


def two_nn_dimension(points: np.ndarray) -> float:
    """Closed-form maximum likelihood, ``d = N / sum log mu``.

    Facco et al. additionally discard the largest decile of ``mu`` before fitting, because
    the estimator is sensitive to the upper tail. That discard is not implemented here, and
    every LID figure in the thesis is read without it.
    """
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
