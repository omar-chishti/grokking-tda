"""Guards for the predictive comparison — the thesis's redundancy verdict rests on it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grokking_tda.evaluation.headtohead import (
    FEATURE_SETS,
    _n_splits,
    _splitter,
    feature_columns,
    nested_scores,
)
from grokking_tda.evaluation.predictive import before_the_event, window_end_step


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


def test_winsorising_does_not_carry_between_outer_folds(monkeypatch) -> None:
    """Each fold's bounds must be the quantiles of its own unclipped training rows. Reading them
    from a running array ratchets them inward fold by fold, and clips a test fold by bounds that
    an earlier fold's training rows fixed --- the leak the winsorising exists to avoid."""
    rng = np.random.default_rng(4)
    n_groups, per = 12, 4
    groups = np.repeat([f"cfg{g}" for g in range(n_groups)], per)
    features = pd.DataFrame(
        {"x__mean": rng.normal(size=len(groups)), "x__trend": rng.normal(size=len(groups))}
    )
    target = features["x__mean"].to_numpy() * 2.0 + rng.normal(scale=0.2, size=len(groups))
    target[[0, 5]] += 30.0  # the heavy tails winsorising is for
    target[[9, 17]] -= 25.0

    winsor = (0.05, 0.95)
    outer = _splitter("regression", _n_splits("regression", target, groups, 5))
    folds = list(outer.split(features, target, groups))
    expected = [np.quantile(target[train], winsor) for train, _ in folds]

    seen = []
    original = np.quantile

    def record(values, q, *args, **kwargs):
        out = original(values, q, *args, **kwargs)
        if q is winsor:  # the caller's own tuple, so this is the winsorising call and not another
            seen.append(out)
        return out

    monkeypatch.setattr(np, "quantile", record)
    nested_scores(features, target, groups, task="regression", winsor=winsor)

    assert len(seen) == len(expected)
    for got, want in zip(seen, expected, strict=True):
        assert np.allclose(got, want)


def test_a_constant_test_fold_is_skipped_rather_than_scored() -> None:
    """Clipping can leave every run in one grouped fold on the same bound. `r2_score` then
    divides by a variance of zero and returns a number of order 1e29, which is how one reached
    a committed table."""
    rng = np.random.default_rng(5)
    n_groups, per = 10, 4
    groups = np.repeat([f"cfg{g}" for g in range(n_groups)], per)
    features = pd.DataFrame(
        {"x__mean": rng.normal(size=len(groups)), "x__trend": rng.normal(size=len(groups))}
    )
    # a target concentrated on one value with a tail at each end: clipping at the tenth and
    # ninetieth centiles leaves nothing but the value itself
    target = np.full(len(groups), 5.0)
    target[groups == "cfg0"] = 0.0
    target[groups == "cfg9"] = 99.0

    assert nested_scores(features, target, groups, task="regression", winsor=(0.1, 0.9)) == []

    unclipped = nested_scores(features, target, groups, task="regression")
    assert unclipped and np.isfinite(unclipped).all()


def test_window_end_step_is_the_run_s_own_convergence_for_tc() -> None:
    assert window_end_step("w5000", None) == 5000
    assert window_end_step("tc", 1200) == 1200
    assert window_end_step("tc", None) is None


def test_the_window_may_not_close_after_the_step_it_predicts() -> None:
    """A run that groks at 500 has its w5000 features measured after the event, so its fit
    reads the label off its own window. Twenty-one of eighty-three did, and they were the
    whole of the positive R^2 in the published grid."""
    table = pd.DataFrame(
        {
            "run": ["early", "late", "never"],
            "grokking_step": [500.0, 40000.0, np.nan],
            "window_step": [5000, 5000, 5000],
        }
    )
    kept = before_the_event(table)
    # the non-grokker stays: it is the negative class, not a leak
    assert list(kept["run"]) == ["late", "never"]
