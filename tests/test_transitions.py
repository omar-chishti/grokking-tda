"""The operational definitions of the transition, which every timing claim rests on.

A detector that answers where there is nothing to find is worse than one that answers
nothing: an invented step enters the lead-lag comparison silently and is indistinguishable
there from a real one. These pin the cases the definitions were written against — a series
that only decays, one that opens on an initialisation transient, and the two detectors
disagreeing about which way an observable moves before they disagree about where it turns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grokking_tda.evaluation import (
    all_transitions,
    changepoint_step,
    early_window_feature_grid,
    grokking_step_sensitivity,
    lead_lag,
    transition_step,
)


def test_transition_step_handles_falling_series() -> None:
    steps = np.array([0, 1, 2, 3])
    falling = np.array([1.0, 1.0, 0.0, 0.0])
    assert transition_step(steps, falling, direction="auto") == 2
    assert transition_step(steps, falling, direction="falling") == 2


def test_transition_step_reports_nothing_for_a_series_that_only_decays() -> None:
    """Raw H1 decays across training in the strongly-decayed regime; the midpoint
    proxy used to answer with a step near zero, which reads as a huge lead."""
    steps = np.arange(0, 100, 10)
    decaying = np.linspace(1.0, 0.0, steps.size)
    assert transition_step(steps, decaying, direction="rising") is None


def test_transition_step_measures_the_rise_from_its_trough() -> None:
    """A high initial value must not satisfy the crossing before the rise happens."""
    steps = np.array([0, 10, 20, 30, 40])
    values = np.array([0.9, 0.1, 0.2, 0.7, 1.1])  # starts high, dips, then rises
    # Measured from the trough the midpoint is 0.6, first reached at step 30; the
    # initial 0.9 would otherwise satisfy it at step 0, before any rise occurred.
    assert transition_step(steps, values, direction="rising") == 30


def test_transition_step_survives_an_initialisation_transient() -> None:
    """Raw H1 starts high at random init, collapses, then rises across the transition.

    The initial value is the global maximum, so requiring the global peak to follow
    the trough would discard a real rise; the comparison is against the highest
    value *after* the trough.
    """
    steps = np.array([0, 10, 20, 30, 40, 50])
    values = np.array([0.82, 0.10, 0.02, 0.03, 0.06, 0.09])
    assert transition_step(steps, values, direction="rising") == 40


def test_both_detectors_resolve_an_undeclared_direction_the_same_way() -> None:
    """``auto`` is not a default the two can hold different opinions about. Read as rising,
    a falling series has no level shift up and the changepoint reports nothing, so the two
    would disagree on 144 of the bank's 165 runs on the observables that declare ``auto``."""
    steps = np.arange(0, 200, 10)
    falling = np.where(steps < 100, 4.0, 1.0)
    assert transition_step(steps, falling, direction="auto") == 100
    assert changepoint_step(steps, falling, direction="auto") == 100


def test_the_changepoint_measures_its_shift_from_the_trough_too() -> None:
    """The normalised maximum drifts slowly down through memorisation before it rises, and
    a level shift located inside that drift is the drift. Both detectors are anchored on the
    trough, or the comparison between them is between a repaired detector and an unrepaired
    one rather than between two definitions of a transition."""
    steps = np.arange(0, 400, 10)
    drift = np.linspace(1.0, 0.8, steps.size)
    drift[steps >= 300] = 1.1
    trough = steps[int(drift.argmin())]
    assert changepoint_step(steps, drift) == 300
    assert changepoint_step(steps, drift) >= trough


def test_all_transitions_covers_every_observable() -> None:
    metrics = pd.DataFrame(
        {"step": [0, 10, 20, 30], "test_acc": [0.0, 0.0, 0.95, 1.0], "train_acc": [1.0] * 4}
    )
    observables = pd.DataFrame(
        {
            "step": [0, 10, 20, 30],
            "h1_max_persistence": [0.0, 0.0, 0.0, 1.0],  # rises
            "lid": [8.0, 8.0, 2.0, 2.0],  # falls
        }
    )
    out = all_transitions(metrics, observables)
    assert set(out) == {"h1_max_persistence", "lid"}
    assert out["h1_max_persistence"]["t_top"] == 30
    assert out["lid"]["t_top"] == 20
    assert out["h1_max_persistence"]["delta"] == -10


def test_lead_lag_reports_signed_delta() -> None:
    metrics = pd.DataFrame(
        {"step": [0, 10, 20, 30], "test_acc": [0.0, 0.0, 0.95, 1.0], "train_acc": [1.0] * 4}
    )
    observables = pd.DataFrame(
        {"step": [0, 10, 20, 30], "h1_max_persistence": [0.0, 0.0, 0.0, 1.0]}
    )
    out = lead_lag(metrics, observables, "h1_max_persistence", acc_threshold=0.9)
    assert out["grokking_step"] == 20
    assert out["topological_transition_step"] == 30
    assert out["lead_lag_steps"] == -10  # topology lags here


def test_grokking_step_sensitivity_spans_the_reported_definitions() -> None:
    """The midpoint of the raw curve is biased early by the commutativity plateau;
    the leak-free midpoint is what shows how large that bias is."""
    step = np.arange(0, 1100, 100)
    raw = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.85, 0.93, 0.99, 1.0])
    novel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.79, 0.90, 0.99, 1.0])
    metrics = pd.DataFrame({"step": step, "test_acc": raw, "train_acc": np.ones_like(raw)})
    observables = pd.DataFrame({"step": step, "test_acc_novel": novel})

    out = grokking_step_sensitivity(metrics, observables)
    assert out["threshold_0.8"] == 700
    assert out["threshold_0.9"] == 800
    assert out["threshold_0.95"] == 900
    # The raw midpoint (halfway from 0.3 to 1.0 => 0.65) fires before the 0.9 threshold.
    assert out["midpoint"] < out["threshold_0.9"]
    assert out["midpoint_novel"] is not None


def test_early_window_grid_uses_preregistered_windows_only() -> None:
    observables = pd.DataFrame({"step": [0, 100, 1000, 10_000], "h1_max_persistence": range(4)})
    grid = early_window_feature_grid(observables, windows=(500, 1000), train_convergence=100)
    assert set(grid) == {"w500", "w1000", "tc"}
    assert all("h1_max_persistence__mean" in feats for feats in grid.values())
