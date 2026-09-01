"""Calibration of the dominant-feature threshold against shapes with known homology.

The torus prediction in the thesis is only as good as the counter that tests it, so the
counter is checked on a circle, a torus and a sphere before it is pointed at a network.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from grokking_tda.config.schema import HomologyCfg
from grokking_tda.tda.betti import betti_profile, dominant_count
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.summaries import finite_lifetimes


def _circle(n=120, seed=0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(t), np.sin(t)]


def _torus(n=26, r=0.42, seed=0):
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu, vv = uu.ravel(), vv.ravel()
    return np.c_[
        (1 + r * np.cos(vv)) * np.cos(uu),
        (1 + r * np.cos(vv)) * np.sin(uu),
        r * np.sin(vv),
    ]


def _sphere(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_circle_has_one_loop_and_no_void() -> None:
    p = betti_profile(_circle(), maxdim=2)
    assert p["betti_1"] == 1
    assert p["betti_2"] == 0


@pytest.mark.slow
def test_torus_has_two_loops_and_one_void() -> None:
    """The structural prediction for the joint-input representation of a modular sum."""
    p = betti_profile(_torus(), maxdim=2)
    assert p["betti_1"] == 2, p
    assert p["betti_2"] == 1, p


@pytest.mark.slow
def test_sphere_has_a_void_but_no_loop() -> None:
    """Distinguishes a torus from a shape that merely has degree-2 homology."""
    p = betti_profile(_sphere(), maxdim=2)
    assert p["betti_1"] == 0, p
    assert p["betti_2"] == 1, p


def test_profile_is_scale_free() -> None:
    """Persistence carries the cloud's units; the profile must not."""
    small = betti_profile(_circle(), maxdim=1)
    large = betti_profile(_circle() * 137.0, maxdim=1)
    assert small["betti_1"] == large["betti_1"] == 1
    # ripser accumulates float error under a large rescale, so this is not exact
    assert abs(small["life_1"] - large["life_1"]) < 1e-6


def test_two_comparable_loops_are_counted_without_a_gap_beneath_them() -> None:
    """A torus is exactly the configuration with no internal gap. Reading dominance as a gap
    alone counted it as nothing, and one noise bar underneath restored the answer --- so the
    cleaner the cloud, the worse the count."""
    assert dominant_count([0.31, 0.23]) == 2
    assert dominant_count([0.31, 0.23, 0.089]) == 2
    assert dominant_count([0.30, 0.30, 0.30]) == 3


def test_bars_barely_over_the_floor_are_not_dominant() -> None:
    """With no gap to read, the floor is the only evidence, and a bar just above it is noise."""
    assert dominant_count([0.06]) == 0
    assert dominant_count([0.07, 0.065, 0.06]) == 0
    assert dominant_count([0.84]) == 1


@pytest.mark.slow
def test_the_torus_count_does_not_depend_on_its_noise_bars() -> None:
    """The coarse torus passes only because two dozen discretisation-noise bars sit just above
    the floor and supply the gap. Sampling finely enough to push them under it costs an hour of
    degree-two homology, so the same configuration is reached by dropping them: the two loops
    alone must still count as two."""
    diagrams = compute_persistence(_torus(), HomologyCfg(maxdim=2))
    lifetimes = finite_lifetimes(diagrams[1])
    scaled = np.sort(lifetimes / pdist(_torus()).max())[::-1]
    assert dominant_count(scaled) == 2
    assert dominant_count(scaled[:2]) == 2
