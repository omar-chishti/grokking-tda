"""Tests for the 2026-06-12 implementation pass.

Covers: run-dir overwrite guard, new tasks (div/poly) and the label-permutation
control, point-cloud options (drop_first, maxmin), persistence entropy, direction-
aware transitions, leak-free predictive windows, full-path hook names, the diagram
disk cache, representation splits, diagram distances / trajectory velocity, the
significance machinery, and run aggregation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from grokking_tda.analysis.aggregate import aggregate_runs
from grokking_tda.analysis.observable import ObservationContext
from grokking_tda.analysis.representations import extract_representation_matrix
from grokking_tda.artifacts import Run, prepare_run_dir
from grokking_tda.config.schema import AnalysisCfg, DataCfg, HomologyCfg, PointCloudCfg
from grokking_tda.data import build_data
from grokking_tda.evaluation import all_transitions, early_window_feature_grid, transition_step
from grokking_tda.models.transformer import GrokkingTransformer
from grokking_tda.tda import (
    bootstrap_summary_ci,
    build_point_cloud,
    compute_persistence,
    diagram_distance,
    persistence_entropy,
    random_init_null,
    trajectory_velocity,
)


def _circle(n: int = 60, radius: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(t), np.sin(t)] * radius


def _target_for(data, lhs: int, rhs: int) -> int:
    idx = ((data.inputs[:, 0] == lhs) & (data.inputs[:, 1] == rhs)).nonzero()[0]
    return int(data.targets[idx])


# --- run-dir overwrite guard --------------------------------------------------


def test_prepare_run_dir_refuses_existing_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        prepare_run_dir(run_dir, overwrite=False)


def test_prepare_run_dir_overwrite_removes_stale_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "snapshots" / "step_00000007").mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")
    (run_dir / "metrics.jsonl").write_text('{"step": 0}\n')
    prepare_run_dir(run_dir, overwrite=True)
    assert not (run_dir / "manifest.json").exists()
    assert not (run_dir / "snapshots").exists()
    # A fresh directory passes silently.
    prepare_run_dir(run_dir, overwrite=False)


# --- tasks ----------------------------------------------------------------------


def test_div_task_excludes_zero_and_uses_modular_inverse() -> None:
    data = build_data(DataCfg(modulus=7, operation="div", train_fraction=0.5), seed=0)
    assert len(data) == 7 * 6  # b = 0 has no inverse
    assert int((data.inputs[:, 1] == 0).sum()) == 0
    assert _target_for(data, 3, 4) == (3 * pow(4, 5, 7)) % 7  # 3 * 4^{-1} = 3*2 = 6


def test_poly_task_labels() -> None:
    data = build_data(DataCfg(modulus=7, operation="poly", train_fraction=0.5), seed=0)
    assert _target_for(data, 2, 3) == (2**3 + 2 * 3) % 7  # == 0


def test_label_permutation_is_deterministic_and_destroys_the_rule() -> None:
    base = DataCfg(modulus=11, operation="add", train_fraction=0.5)
    orig = build_data(base, seed=0)
    perm_cfg = DataCfg(modulus=11, operation="add", train_fraction=0.5, label_permutation=True)
    perm_a = build_data(perm_cfg, seed=0)
    perm_b = build_data(perm_cfg, seed=0)
    assert torch.equal(perm_a.targets, perm_b.targets)  # seeded, reproducible
    assert not torch.equal(perm_a.targets, orig.targets)  # rule destroyed
    assert torch.equal(  # same label marginals (a permutation, not relabelling)
        torch.sort(perm_a.targets).values, torch.sort(orig.targets).values
    )


# --- point-cloud construction -----------------------------------------------------


def test_drop_first_excludes_residue_zero() -> None:
    x = np.arange(20.0).reshape(10, 2)
    cloud = build_point_cloud(x, PointCloudCfg(normalize="none", drop_first=True))
    assert cloud.shape == (9, 2)


def test_maxmin_landmarks_cover_outliers() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(size=(30, 2)) * 0.1, [[100.0, 0.0]]])
    cfg = PointCloudCfg(normalize="none", max_points=5, subsample="maxmin")
    cloud = build_point_cloud(x, cfg, seed=0)
    assert cloud.shape == (5, 2)
    assert np.linalg.norm(cloud, axis=1).max() > 50  # the far point is always kept


# --- summaries / transitions / windows ---------------------------------------------


def test_persistence_entropy() -> None:
    two_equal = np.array([[0.0, 1.0], [0.0, 1.0]])
    one_bar = np.array([[0.0, 1.0]])
    assert abs(persistence_entropy(two_equal) - np.log(2)) < 1e-12
    assert persistence_entropy(one_bar) == 0.0
    assert persistence_entropy(np.empty((0, 2))) == 0.0


def test_transition_step_handles_falling_series() -> None:
    steps = np.array([0, 1, 2, 3])
    falling = np.array([1.0, 1.0, 0.0, 0.0])
    assert transition_step(steps, falling, direction="auto") == 2
    assert transition_step(steps, falling, direction="falling") == 2


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


def test_early_window_grid_uses_preregistered_windows_only() -> None:
    observables = pd.DataFrame({"step": [0, 100, 1000, 10_000], "h1_max_persistence": range(4)})
    grid = early_window_feature_grid(observables, windows=(500, 1000), train_convergence=100)
    assert set(grid) == {"w500", "w1000", "tc"}
    assert all("h1_max_persistence__mean" in feats for feats in grid.values())


# --- hooks --------------------------------------------------------------------


def test_hook_names_are_unique_across_blocks() -> None:
    model = GrokkingTransformer(vocab_size=12, num_classes=11, seq_len=3, n_layers=2)
    tokens = torch.zeros((2, 3), dtype=torch.long)
    _, cache = model.run_with_cache(tokens)
    assert "blocks.0.hook_attn_out" in cache
    assert "blocks.1.hook_attn_out" in cache
    assert "hook_resid_final" in cache  # top-level hooks keep their bare name


# --- diagram disk cache + representation splits (on the shared tiny run) ------------


def test_diagrams_are_cached_to_disk_and_reread(tiny_run) -> None:
    run_dir, _ = tiny_run
    run = Run(run_dir)
    snapshot = run.snapshots()[0]
    cfg = AnalysisCfg()
    first = ObservationContext(run, snapshot, cfg).diagrams()
    cache_files = list((run_dir / "analysis" / "diagrams").glob("step_*.npz"))
    assert cache_files
    second = ObservationContext(run, snapshot, cfg).diagrams()
    assert set(first) == set(second)
    for dim in first:
        assert np.allclose(first[dim], second[dim], equal_nan=True)


def test_representation_split_restricts_rows(tiny_run) -> None:
    run_dir, _ = tiny_run
    run = Run(run_dir)
    snapshot = run.snapshots()[-1]
    data = build_data(DataCfg(**run.config["data"]), run.config["seed"])
    n_test = int((~data.train_mask).sum())
    hidden_all = extract_representation_matrix(run, snapshot, "hidden", split="all")
    hidden_test = extract_representation_matrix(run, snapshot, "hidden", split="test")
    assert hidden_all.shape[0] == len(data)
    assert hidden_test.shape[0] == n_test


# --- distances / velocity ----------------------------------------------------------


def test_diagram_distance_separates_circles_of_different_radius() -> None:
    h = HomologyCfg(maxdim=1)
    d1 = compute_persistence(_circle(40, 1.0), h)[1]
    d2 = compute_persistence(_circle(40, 1.5), h)[1]
    assert diagram_distance(d1, d1, metric="bottleneck") < 1e-9
    assert diagram_distance(d1, d2, metric="bottleneck") > 0.1
    # An empty diagram is diagonal-only: distance is half the longest lifetime.
    lifetime = (d1[:, 1] - d1[:, 0]).max()
    assert abs(diagram_distance(None, d1, metric="bottleneck") - lifetime / 2) < 1e-9


def test_trajectory_velocity_shape() -> None:
    h = HomologyCfg(maxdim=1)
    diagrams = [compute_persistence(_circle(40, r), h)[1] for r in (1.0, 1.0, 1.5)]
    velocity = trajectory_velocity([0, 10, 20], diagrams, metric="bottleneck")
    assert list(velocity["step"]) == [10, 20]
    assert velocity["distance"].iloc[0] < velocity["distance"].iloc[1]


# --- significance -------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_circle_signal() -> None:
    out = bootstrap_summary_ci(_circle(60), n_boot=20, seed=0)
    ci = out["max"]
    assert ci["lo"] <= ci["median"] <= ci["hi"]
    assert ci["median"] > 0.5  # the loop survives 80% subsampling
    assert len(out["samples"]) == 20


def test_random_init_null_runs_on_a_real_manifest(tiny_run) -> None:
    run_dir, _ = tiny_run
    null = random_init_null(Run(run_dir), n_samples=2, seed=0)
    assert len(null) == 2
    assert np.isfinite(null["max"]).all()
    assert (null["max"] >= 0).all()


# --- aggregation ---------------------------------------------------------------------


def test_aggregate_runs_joins_config_and_summary(tiny_run, tmp_path) -> None:
    run_dir, _ = tiny_run
    table = aggregate_runs(run_dir)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["run"] == "tiny"
    assert row["model"] == "mlp"
    assert row["operation"] == "add"

    # A run with a summary gets its transitions flattened into columns.
    fake = tmp_path / "fake_run"
    (fake / "analysis").mkdir(parents=True)
    manifest = {
        "run_name": "fake",
        "config": {
            "seed": 1,
            "model": {"name": "mlp"},
            "data": {"operation": "add", "modulus": 11, "train_fraction": 0.5},
            "train": {"loss": "softmax_ce", "steps": 10, "optimizer": {"name": "adamw"}},
        },
        "env": {},
        "task_meta": {"modulus": 11},
        "created_at": "now",
    }
    (fake / "manifest.json").write_text(json.dumps(manifest))
    summary = {
        "grokking_step": 20,
        "transitions": {"h1_max_persistence": {"t_top": 30, "delta": -10}},
    }
    (fake / "analysis" / "summary.json").write_text(json.dumps(summary))
    table = aggregate_runs(tmp_path)
    row = table[table["run"] == "fake"].iloc[0]
    assert row["grokking_step"] == 20
    assert row["t_top__h1_max_persistence"] == 30
    assert row["delta__h1_max_persistence"] == -10


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


def test_grokking_step_sensitivity_spans_the_reported_definitions() -> None:
    """The midpoint of the raw curve is biased early by the commutativity plateau;
    the leak-free midpoint is what shows how large that bias is."""
    from grokking_tda.evaluation import grokking_step_sensitivity

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
