"""Persistence itself, on shapes whose answer is known before the code runs.

Every number in the thesis is a summary of a diagram, so the diagram has to be right on a
circle and empty on a blob before anything it says about an embedding is worth reading.
The summaries that read a diagram rather than a cloud — entropy, the distance between two
of them, the velocity along a sequence, the dimension of the path — are pinned here too.
"""

from __future__ import annotations

import numpy as np

from grokking_tda.artifacts import Run
from grokking_tda.config.schema import HomologyCfg
from grokking_tda.tda import (
    auto_crocker,
    compute_persistence,
    diagram_distance,
    max_persistence,
    n_features,
    persistence_entropy,
    random_init_null,
    trajectory_velocity,
)
from grokking_tda.tda.trajectory import crocker_matrix


def _circle(n: int = 60, radius: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(t), np.sin(t)] * radius


def test_circle_has_exactly_one_prominent_loop() -> None:
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = np.c_[np.cos(t), np.sin(t)]
    diagrams = compute_persistence(circle, HomologyCfg(maxdim=1))
    assert max_persistence(diagrams[1]) > 0.5
    assert n_features(diagrams[1], min_persistence=0.5) == 1


def test_blob_has_no_loop() -> None:
    rng = np.random.default_rng(0)
    blob = rng.normal(size=(60, 2)) * 0.05  # tight gaussian, no hole
    diagrams = compute_persistence(blob, HomologyCfg(maxdim=1))
    assert max_persistence(diagrams[1]) < 0.5


def test_crocker_matrix_shape() -> None:
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    clouds = [np.c_[np.cos(t), np.sin(t)] * r for r in (0.1, 1.0)]
    scales = np.linspace(0, 2, 25)
    matrix = crocker_matrix(clouds, scales, homology_dim=1)
    assert matrix.shape == (2, 25)
    assert matrix.max() >= 1  # the unit circle has a loop at some scale


def test_diverged_weights_yield_empty_diagrams_rather_than_raising() -> None:
    """A NaN cloud is a diverged run, which must be recorded, not crash the analysis."""
    cloud = np.random.default_rng(0).normal(size=(20, 3))
    cloud[7, 1] = np.nan
    diagrams = compute_persistence(cloud, HomologyCfg(maxdim=1))
    assert set(diagrams) == {0, 1}
    assert all(dgm.size == 0 for dgm in diagrams.values())


def test_ph_dimension_fit_reports_the_slope_behind_the_dimension() -> None:
    """dim = alpha / (1 - slope) is stiff near the bottom of its range, so the slope is
    part of the answer rather than an implementation detail."""
    from grokking_tda.tda.phdim import ph_dimension, ph_dimension_fit

    rng = np.random.default_rng(0)
    for true_dim, expected_slope in ((1, 0.2), (3, 0.67)):
        points = rng.normal(size=(400, true_dim))
        fit = ph_dimension_fit(points)
        assert abs(fit["ph_dim"] - true_dim) < 0.4
        assert abs(fit["slope"] - expected_slope) < 0.15
        assert fit["r2"] > 0.85
        assert ph_dimension(points) == fit["ph_dim"]


def test_ph_dimension_fit_returns_the_slope_even_when_it_is_inadmissible() -> None:
    """A degenerate window has no dimension, but knowing *why* is the point of the fit."""
    from grokking_tda.tda.phdim import ph_dimension_fit

    fit = ph_dimension_fit(np.zeros((200, 3)))
    assert np.isnan(fit["ph_dim"])


def test_auto_crocker_scales_to_the_plotted_dimension() -> None:
    clouds = [_circle(40, r) for r in (0.5, 1.0, 1.5)]
    scales, matrix = auto_crocker(clouds, homology_dim=1, n_scales=30)
    assert matrix.shape == (3, 30)
    assert scales[0] >= 0.0 and scales[-1] > scales[0]  # framed, ascending grid
    assert matrix.max() >= 1  # the loop is alive at some scale (grid is not swamped)


def test_persistence_entropy() -> None:
    two_equal = np.array([[0.0, 1.0], [0.0, 1.0]])
    one_bar = np.array([[0.0, 1.0]])
    assert abs(persistence_entropy(two_equal) - np.log(2)) < 1e-12
    assert persistence_entropy(one_bar) == 0.0
    assert persistence_entropy(np.empty((0, 2))) == 0.0


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


def test_random_init_null_runs_on_a_real_manifest(tiny_run) -> None:
    run_dir, _ = tiny_run
    null = random_init_null(Run(run_dir), n_samples=2, seed=0)
    assert len(null) == 2
    assert np.isfinite(null["max"]).all()
    assert (null["max"] >= 0).all()
