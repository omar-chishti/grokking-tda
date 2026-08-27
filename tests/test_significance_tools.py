"""Guards for the three statistical tools the thesis protocol promises."""

from __future__ import annotations

import numpy as np

from grokking_tda.evaluation.changepoint import changepoint_step, changepoints
from grokking_tda.evaluation.multiplicity import benjamini_hochberg, benjamini_yekutieli
from grokking_tda.evaluation.pid import gaussian_pid, williams_beer_pid


def test_bh_rejects_nothing_when_everything_is_null() -> None:
    rng = np.random.default_rng(0)
    rejected, adjusted = benjamini_hochberg(rng.uniform(size=200), q=0.1)
    assert rejected.sum() <= 2  # at q=0.1 a handful at most, by construction
    assert np.all(adjusted[np.isfinite(adjusted)] <= 1.0)


def test_bh_finds_a_real_signal_among_nulls() -> None:
    rng = np.random.default_rng(1)
    p = np.concatenate([np.full(10, 1e-6), rng.uniform(size=90)])
    rejected, _ = benjamini_hochberg(p, q=0.1)
    assert rejected[:10].all()


def test_bh_adjusted_values_are_monotone_in_the_raw_ones() -> None:
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.9])
    _, adjusted = benjamini_hochberg(p, q=0.1)
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_bh_carries_untestable_cells_through_without_rejecting_them() -> None:
    """A cell that could not be tested is not a discovery."""
    p = np.array([1e-8, np.nan, 0.4])
    rejected, adjusted = benjamini_hochberg(p, q=0.1)
    assert rejected[0] and not rejected[1]
    assert np.isnan(adjusted[1])


def test_changepoint_locates_a_level_shift() -> None:
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(0, 0.05, 60), rng.normal(1.0, 0.05, 60)])
    cuts = changepoints(x)
    assert cuts and abs(cuts[0] - 60) <= 3


def test_changepoint_reports_nothing_on_noise() -> None:
    rng = np.random.default_rng(3)
    assert changepoints(rng.normal(0, 1.0, 200)) == []


def test_changepoint_step_returns_the_step_of_the_rise() -> None:
    steps = np.arange(0, 1200, 10)
    values = np.where(steps < 600, 0.02, 0.5)
    assert abs(changepoint_step(steps, values, direction="rising") - 600) <= 20


def test_pid_calls_identical_sources_redundant() -> None:
    rng = np.random.default_rng(4)
    t = rng.normal(size=400)
    a = t + rng.normal(0, 0.4, 400)
    atoms = gaussian_pid(a, a.copy(), t)
    assert atoms["redundant"] > 0.2
    assert abs(atoms["unique_a"]) < 1e-6 and abs(atoms["unique_b"]) < 1e-6


def test_pid_gives_a_source_unique_information_when_it_has_some() -> None:
    """B is pure noise, so everything A knows is unique to it."""
    rng = np.random.default_rng(5)
    t = rng.normal(size=600)
    a = t + rng.normal(0, 0.3, 600)
    b = rng.normal(size=600)
    atoms = gaussian_pid(a, b, t)
    assert atoms["unique_a"] > atoms["unique_b"]
    assert atoms["unique_a"] > 0.2
    assert all(atoms[k] >= -1e-6 for k in ("redundant", "unique_a", "unique_b", "synergistic"))


def test_by_is_never_more_permissive_than_bh() -> None:
    """BY buys validity under arbitrary dependence by spending power; it cannot buy both."""
    rng = np.random.default_rng(6)
    p = np.concatenate([rng.uniform(0, 0.02, 20), rng.uniform(size=80)])
    bh_rejected, bh_adjusted = benjamini_hochberg(p, q=0.1)
    by_rejected, by_adjusted = benjamini_yekutieli(p, q=0.1)
    assert by_rejected.sum() <= bh_rejected.sum()
    assert np.all(by_adjusted >= bh_adjusted - 1e-12)


def test_by_still_finds_an_unambiguous_signal() -> None:
    p = np.concatenate([np.full(10, 1e-9), np.full(90, 0.5)])
    rejected, _ = benjamini_yekutieli(p, q=0.1)
    assert rejected[:10].all() and not rejected[10:].any()


def test_williams_beer_credits_a_source_that_resolves_a_different_part_of_the_target() -> None:
    """The case MMI cannot express: A resolves the low half of T, B the high half."""
    rng = np.random.default_rng(7)
    t = rng.normal(size=4000)
    a = np.where(t < 0, t, 0.0) + rng.normal(0, 0.3, 4000)
    b = np.where(t > 0, t, 0.0) + rng.normal(0, 0.3, 4000)
    mmi = gaussian_pid(a, b, t)
    atoms = williams_beer_pid(a, b, t, bins=5)
    assert min(mmi["unique_a"], mmi["unique_b"]) < 1e-9  # one source is crushed
    assert min(atoms["unique_a"], atoms["unique_b"]) > 0.05  # ... and here neither is


def test_williams_beer_gives_a_dominated_source_nothing() -> None:
    """It must not manufacture unique information: B is A with less noise."""
    rng = np.random.default_rng(8)
    t = rng.normal(size=4000)
    a = t + rng.normal(0, 1.5, 4000)
    b = t + rng.normal(0, 0.4, 4000)
    atoms = williams_beer_pid(a, b, t, bins=5)
    assert atoms["unique_a"] < 1e-9 < atoms["unique_b"]
