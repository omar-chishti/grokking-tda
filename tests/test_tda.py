from __future__ import annotations

import numpy as np

from grokking_tda.config.schema import HomologyCfg
from grokking_tda.tda import compute_persistence, max_persistence, n_features
from grokking_tda.tda.trajectory import crocker_matrix


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
