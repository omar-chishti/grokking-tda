"""The window rule, the verdict rule and the replicate flag — the layer the thesis quotes.

Every number in Chapters 4-7 comes out of ``analysis/bank.py``, so the rules it encodes are
worth pinning: the windows are anchored on ``t_g`` and fall back when there is none, the
verdict is stated against the null's observed extremes rather than a quantile, and re-runs of
an existing condition are excluded from the condition table rather than pooled into it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis.bank import (
    _window,
    bootstrap_median_ci,
    circularity_column,
    condition_table,
    null_band,
    task_modulus,
    verdicts,
)


@pytest.fixture
def series() -> pd.DataFrame:
    """A step function: 1.0 up to step 1000, 4.0 after."""
    steps = np.arange(0, 2001, 50)
    return pd.DataFrame({"step": steps, "x": np.where(steps < 1000, 1.0, 4.0)})


def test_window_is_anchored_on_t_g(series):
    base, plateau = _window(series, "x", 1000.0)
    assert (base, plateau) == (1.0, 4.0)


def test_window_without_t_g_splits_the_run(series):
    # 0.3-0.6 of 2000 is 600-1200, which straddles the step; the plateau is the final fifth.
    base, plateau = _window(series, "x", None)
    assert plateau == 4.0
    assert 1.0 <= base <= 4.0


def test_bootstrap_interval_degenerates_at_two_seeds():
    """With two values the percentile interval is exactly their range, and the verdict rule
    reduces to ``both seeds clear the null``. Three conditions in the bank carry two seeds."""
    lo, med, hi = bootstrap_median_ci([1.0, 3.0])
    assert (lo, hi) == (1.0, 3.0)
    assert med == 2.0


def test_bootstrap_interval_at_five_seeds_reaches_the_sample_minimum():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, med, hi = bootstrap_median_ci(values)
    assert lo == min(values) and hi == max(values) and med == 3.0


def _bank(ratios: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for name, values in ratios.items():
        for i, value in enumerate(values):
            rows.append({
                "run": f"{name}_s{i}", "model": "mlp", "operation": name, "modulus": 97,
                "train_fraction": 0.3, "label_permutation": name == "permuted",
                "loss": "softmax_ce", "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1.0,
                "grokked": name not in {"permuted", "poly"}, "t_g": 100.0,
                "replicate": False, "x__ratio": value, "circularity": 0.5,
                "scale_collapse": 10.0, "test_acc__final": 1.0,
            })
    return pd.DataFrame(rows)


def test_verdict_needs_the_whole_interval_past_the_null_maximum():
    bank = _bank({"permuted": [0.9, 1.0, 1.1], "poly": [0.95, 1.05],
                  "clear": [2.0, 2.1, 2.2], "touching": [1.1, 2.0, 2.1]})
    band = null_band(bank, "x__ratio")
    assert (band["observed_min"], band["observed_max"]) == (0.9, 1.1)

    table = verdicts(bank, condition_table(bank, observables=["x"]), observables=["x"])
    by_operation = table.set_index("operation")["x__verdict"]
    assert by_operation["clear"] == "above"
    # Its lower bound sits on the null's maximum rather than past it.
    assert by_operation["touching"] == "inside"


def test_replicates_are_excluded_from_the_condition_table():
    bank = _bank({"clear": [2.0, 2.1, 2.2]})
    bank.loc[bank.run == "clear_s2", "replicate"] = True
    pooled = condition_table(bank, observables=["x"])
    assert int(pooled.n_runs.iloc[0]) == 2


def test_task_modulus_recomputes_the_group_order():
    """An early batch of S_5 runs carries the default modulus in its config."""
    assert task_modulus({"task": "permutation_group", "n_symbols": 5, "modulus": 97}) == 120
    assert task_modulus({"operation": "add", "modulus": 113}) == 113


def test_no_spectral_circularity_for_the_permutation_group():
    columns = ["fourier_concentration_k5", "fourier_concentration_group_k5"]
    assert circularity_column("add", columns) == "fourier_concentration_k5"
    assert circularity_column("mul", columns) == "fourier_concentration_group_k5"
    assert circularity_column("compose", columns) is None


def test_unrepaired_detector_fires_at_step_zero_on_a_decaying_series():
    """The artefact section 5.6 reports: a series that opens high crosses the midpoint of its
    own global extremes immediately, so the lag is the whole of ``t_g``."""
    from grokking_tda.evaluation import transition_step

    steps = np.arange(0, 1001, 100)
    decaying = np.linspace(5.0, 1.0, steps.size)
    assert transition_step(steps, decaying, compare="global") == 0
    assert transition_step(steps, decaying) is None
