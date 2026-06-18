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
