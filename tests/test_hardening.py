"""Tests for the reliability/observability hardening pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from grokking_tda.artifacts import Run
from grokking_tda.evaluation import lead_lag
from grokking_tda.tda import auto_crocker
from grokking_tda.training import snapshot_steps
from grokking_tda.utils.precision import supported_float_dtype


def test_highprec_dtype_kept_on_cpu() -> None:
    assert supported_float_dtype(torch.float64, torch.device("cpu")) is torch.float64
    assert supported_float_dtype(torch.float32, torch.device("cpu")) is torch.float32


def test_snapshot_steps_log_brackets_endpoints() -> None:
    steps = snapshot_steps(1000, 10, "log")
    assert steps[0] == 0 and steps[-1] == 1000
    assert steps == sorted(set(steps))  # sorted and de-duplicated
    assert all(0 <= s <= 1000 for s in steps)


def test_auto_crocker_scales_to_the_plotted_dimension() -> None:
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    clouds = [np.c_[np.cos(t), np.sin(t)] * r for r in (0.5, 1.0, 1.5)]
    scales, matrix = auto_crocker(clouds, homology_dim=1, n_scales=30)
    assert matrix.shape == (3, 30)
    assert scales[0] >= 0.0 and scales[-1] > scales[0]  # framed, ascending grid
    assert matrix.max() >= 1  # the loop is alive at some scale (grid is not swamped)


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


def test_run_records_lifecycle_events(tiny_run) -> None:
    run_dir, _ = tiny_run
    events = Run(run_dir).events
    assert not events.empty
    assert (events["event"] == "train_end").any()
