"""Guards for the predictive comparison — the thesis's redundancy verdict rests on it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grokking_tda.evaluation.headtohead import (
    FEATURE_SETS,
    feature_columns,
    nested_scores,
)


def test_feature_columns_does_not_overmatch_a_longer_name() -> None:
    """`h1_max_persistence` must not swallow `h1_max_persistence_normalised`."""
    available = [
        "h1_max_persistence__mean",
        "h1_max_persistence__trend",
        "h1_max_persistence_normalised__mean",
        "h1_max_persistence_normalised__trend",
    ]
    picked = feature_columns(("h1_max_persistence",), available)
    assert picked == ["h1_max_persistence__mean", "h1_max_persistence__trend"]


def test_feature_sets_are_nested_so_the_increment_is_attributable() -> None:
    assert set(FEATURE_SETS["baselines"]) < set(FEATURE_SETS["baselines+topology"])
    added = set(FEATURE_SETS["baselines+topology"]) - set(FEATURE_SETS["baselines"])
    assert added == set(FEATURE_SETS["topology"])


def _grouped_frame(rng, signal: bool, n_groups: int = 12, per_group: int = 5, positives=None):
    """One configuration per group, `per_group` seeds each, as the real table is."""
    group_effect = rng.normal(size=n_groups)
    if positives is not None:  # force the rarer class down to a chosen number of groups
        group_effect = np.array([1.0] * positives + [-1.0] * (n_groups - positives))
    groups, target, feature = [], [], []
    for g in range(n_groups):
        for _ in range(per_group):
            groups.append(f"cfg{g}")
            target.append(float(group_effect[g] > 0))
            feature.append(group_effect[g] + rng.normal(scale=0.3) if signal else rng.normal())
    frame = pd.DataFrame({"x__mean": feature, "x__trend": rng.normal(size=len(feature))})
    return frame, np.array(target), np.array(groups)


def test_nested_scores_recovers_a_real_signal() -> None:
    rng = np.random.default_rng(0)
    features, target, groups = _grouped_frame(rng, signal=True)
    scores = nested_scores(features, target, groups, task="classification")
    assert scores and np.isfinite(scores).all()
    assert np.mean(scores) > 0.8


def test_nested_scores_stays_at_chance_when_features_are_noise() -> None:
    """The target is fixed by the configuration, so a run's siblings would give it away
    if folds were split by run. Grouped folds must leave noise features at chance."""
    rng = np.random.default_rng(1)
    features, target, groups = _grouped_frame(rng, signal=False)
    scores = nested_scores(features, target, groups, task="classification")
    assert scores and np.isfinite(scores).all()
    assert abs(np.mean(scores) - 0.5) < 0.25


def test_a_thin_minority_class_does_not_tune_on_nan(recwarn) -> None:
    """Two configurations of the rarer class cannot fill every inner fold, so the grid
    would score `NaN` throughout and pick its first candidate. Fit at a fixed penalty
    instead, and leave no non-finite scores behind."""
    rng = np.random.default_rng(2)
    features, target, groups = _grouped_frame(rng, signal=True, positives=2)
    scores = nested_scores(features, target, groups, task="classification")
    assert scores and np.isfinite(scores).all()
    assert not [w for w in recwarn if "non-finite" in str(w.message)]


def test_winsorising_bounds_come_from_the_training_fold_only() -> None:
    """A robustness check with a leak in it is weaker evidence than it appears: clipping the
    whole sample first lets each test fold's target be shaped by its own contents."""
    rng = np.random.default_rng(3)
    features, target, groups = _grouped_frame(rng, signal=True)
    target = target + rng.normal(scale=0.2, size=len(target))
    plain = nested_scores(features, target, groups, task="regression")
    clipped = nested_scores(features, target.copy(), groups, task="regression", winsor=(0.1, 0.9))
    assert np.isfinite(plain).all() and np.isfinite(clipped).all()
    # the caller's array must survive: an in-place clip would silently winsorise every
    # later feature set against a target that had already been clipped once
    assert target.max() > np.quantile(target, 0.9)
