"""Partial information decomposition of what topology and Fourier say about grokking.

The redundancy question — does the topological observable carry information about the
transition beyond the Fourier baseline — is a question about how information is
distributed among sources, and a predictive comparison can only approach it obliquely.
Williams and Beer's decomposition answers it directly: the mutual information two sources
jointly carry about a target splits into redundant, unique and synergistic atoms, and the
*unique* information carried by topology is the quantity of interest.

The atoms are not fixed by Shannon information alone, so a redundancy function has to be
supplied. This uses the minimum-mutual-information redundancy on Gaussian variables — the
estimator Luppi et al. adopt — under which redundancy is ``min_i I(S_i; T)``. That choice
is a genuine limitation and belongs in the text beside any number produced here: MMI is
the most common choice and the least generous to synergy, so a synergy estimate from it is
conservative.

Gaussianity is an assumption about these observables, not a fact. Rank-transform the
inputs (``normal_scores=True``, the default) and it becomes an assumption about the
*copula* instead, which is far weaker and is what makes the estimate defensible here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def _to_normal_scores(x: np.ndarray) -> np.ndarray:
    """Rank-transform each column to standard normal marginals."""
    ranks = np.apply_along_axis(rankdata, 0, x)
    return norm.ppf(ranks / (x.shape[0] + 1.0))


def _gaussian_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information (nats) between two jointly Gaussian blocks."""
    x = np.atleast_2d(x.T).T
    y = np.atleast_2d(y.T).T
    joint = np.hstack([x, y])
    sign_x, log_x = np.linalg.slogdet(np.cov(x, rowvar=False).reshape(x.shape[1], -1))
    sign_y, log_y = np.linalg.slogdet(np.cov(y, rowvar=False).reshape(y.shape[1], -1))
    sign_j, log_j = np.linalg.slogdet(np.cov(joint, rowvar=False))
    if min(sign_x, sign_y, sign_j) <= 0:
        return float("nan")
    return float(max(0.5 * (log_x + log_y - log_j), 0.0))


def gaussian_pid(
    source_a, source_b, target, *, normal_scores: bool = True
) -> dict[str, float]:
    """MMI decomposition of ``I({A,B}; T)`` into redundant, unique and synergistic atoms.

    Returns nats. ``unique_a`` is the headline: information about the transition that the
    topological source carries and the baseline does not.
    """
    a = np.asarray(source_a, dtype=float).reshape(-1, 1)
    b = np.asarray(source_b, dtype=float).reshape(-1, 1)
    t = np.asarray(target, dtype=float).reshape(-1, 1)
    data = np.hstack([a, b, t])
    keep = np.isfinite(data).all(axis=1)
    if keep.sum() < 8:
        return dict.fromkeys(
            ("redundant", "unique_a", "unique_b", "synergistic", "total", "n"), float("nan")
        )
    data = data[keep]
    if normal_scores:
        data = _to_normal_scores(data)
    a, b, t = data[:, :1], data[:, 1:2], data[:, 2:3]

    mi_a, mi_b = _gaussian_mi(a, t), _gaussian_mi(b, t)
    mi_joint = _gaussian_mi(np.hstack([a, b]), t)
    redundant = min(mi_a, mi_b)  # MMI redundancy
    unique_a, unique_b = mi_a - redundant, mi_b - redundant
    synergistic = mi_joint - redundant - unique_a - unique_b
    return {
        "redundant": redundant,
        "unique_a": unique_a,
        "unique_b": unique_b,
        "synergistic": synergistic,
        "total": mi_joint,
        "n": float(keep.sum()),
    }


def _specific_information(joint: np.ndarray, axis: int) -> np.ndarray:
    """``I(T = t; S)`` for each target value: the surprise a source resolves about it.

    ``joint`` is a ``(n_source_a, n_source_b, n_target)`` probability table; ``axis`` names
    which source to marginalise onto.
    """
    p_st = joint.sum(axis=1 - axis)  # (n_source, n_target)
    p_s, p_t = p_st.sum(axis=1), p_st.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # sum_s p(s|t) [ log 1/p(s) - log 1/p(s|t) ], the pointwise form Williams and Beer use
        p_s_given_t = np.where(p_t > 0, p_st / p_t, 0.0)
        log_conditional = np.log(np.where(p_s_given_t > 0, p_s_given_t, 1.0))
        terms = p_s_given_t * (log_conditional - np.log(p_s)[:, None])
    return np.nansum(np.where(np.isfinite(terms), terms, 0.0), axis=0)


def williams_beer_pid(source_a, source_b, target, *, bins: int = 4) -> dict[str, float]:
    """Williams-Beer decomposition on quantile-binned variables (nats).

    The reason this exists beside ``gaussian_pid``: Barrett showed that for jointly
    Gaussian variables with univariate sources and target, essentially every proposed
    redundancy function collapses to the minimum-mutual-information one, under which the
    weaker source's unique atom is zero by construction. That is a property of the
    Gaussian model, not of the data, and it cannot be escaped by choosing a different
    redundancy function *within* that model.

    ``I_min`` is a redundancy over target values rather than over their average, so it can
    credit a weaker source with unique information: a source that resolves a *different
    part* of the target's range is not dominated even when its average mutual information
    is smaller. Binning is what buys that, and it costs resolution, so both estimates
    belong in any report and neither replaces the other.
    """
    data = np.column_stack(
        [np.asarray(v, dtype=float) for v in (source_a, source_b, target)]
    )
    data = data[np.isfinite(data).all(axis=1)]
    if data.shape[0] < 4 * bins:
        return dict.fromkeys(
            ("redundant", "unique_a", "unique_b", "synergistic", "total", "n"), float("nan")
        )

    coded = np.column_stack(
        [
            np.clip(
                np.searchsorted(np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1]), col),
                0,
                bins - 1,
            )
            for col in data.T
        ]
    )
    joint = np.zeros((bins, bins, bins))
    np.add.at(joint, (coded[:, 0], coded[:, 1], coded[:, 2]), 1.0)
    joint /= joint.sum()

    p_t = joint.sum(axis=(0, 1))
    spec_a = _specific_information(joint, axis=0)
    spec_b = _specific_information(joint, axis=1)
    redundant = float(np.sum(p_t * np.minimum(spec_a, spec_b)))
    mi_a, mi_b = float(np.sum(p_t * spec_a)), float(np.sum(p_t * spec_b))

    pair = joint.reshape(bins * bins, bins)
    p_pair, p_target = pair.sum(axis=1), pair.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pair / np.outer(p_pair, p_target)
        mi_joint = float(np.nansum(pair * np.log(np.where(ratio > 0, ratio, 1.0))))

    unique_a, unique_b = mi_a - redundant, mi_b - redundant
    return {
        "redundant": redundant,
        "unique_a": unique_a,
        "unique_b": unique_b,
        "synergistic": mi_joint - redundant - unique_a - unique_b,
        "total": mi_joint,
        "mi_a": mi_a,
        "mi_b": mi_b,
        "n": float(data.shape[0]),
        "bins": float(bins),
    }
