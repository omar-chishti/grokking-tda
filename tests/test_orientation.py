"""The degree measurement, on circles whose degree is known by construction."""

from __future__ import annotations

import numpy as np
import pytest
from sidequest.orientation import (
    _plane,
    alignment,
    antisymmetry,
    monotonicity,
    overlap,
    planarity,
    winding,
)

P = 41


def _circle(direction: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    """``p`` points once around a circle, embedded in ``dim`` dimensions at a random attitude."""
    theta = direction * 2 * np.pi * np.arange(P) / P
    flat = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    rng = np.random.default_rng(seed)
    frame = np.linalg.qr(rng.normal(size=(dim, dim)))[0][:, :2]
    return flat @ frame.T


@pytest.mark.parametrize("direction", [1, -1])
def test_a_circle_winds_once_in_its_own_plane(direction: int) -> None:
    """Once round, in either direction. The sign belongs to the plane's basis, which the
    decomposition fixes only up to a reflection, so it is not asserted here."""
    loop = _circle(direction)
    assert abs(winding(loop, _plane(loop))) == pytest.approx(1.0, abs=1e-9)


def test_the_product_of_two_windings_is_basis_free() -> None:
    """The sign of a principal axis is arbitrary, so only the product carries meaning. Flipping
    the basis must flip both windings and leave the product alone."""
    forward, backward = _circle(1), _circle(-1)
    basis = _plane(np.vstack([forward, backward]))
    product = winding(forward, basis) * winding(backward, basis)
    flipped = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ basis
    assert winding(forward, flipped) * winding(backward, flipped) == pytest.approx(product)
    assert product < 0, "opposite traversals must give a negative product"


def test_the_same_traversal_gives_a_positive_product() -> None:
    """Same plane, same direction. Two circles in *different* planes have no shared frame, which
    is what ``overlap`` is for and why the product is read only when the overlap is high."""
    a = _circle(1)
    b = np.roll(a, 7, axis=0)
    basis = _plane(np.vstack([a, b]))
    assert winding(a, basis) * winding(b, basis) > 0


def test_one_circle_traversed_both_ways_shares_its_plane() -> None:
    forward = _circle(1)
    assert overlap(_plane(forward), _plane(forward[::-1])) == pytest.approx(1.0, abs=1e-9)


def test_two_circles_in_orthogonal_planes_do_not() -> None:
    rng = np.random.default_rng(0)
    frame = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    theta = 2 * np.pi * np.arange(P) / P
    flat = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    a, b = flat @ frame[:, :2].T, flat @ frame[:, 2:4].T
    assert overlap(_plane(a), _plane(b)) == pytest.approx(0.0, abs=1e-6)


def test_noise_winds_too_which_is_why_the_winding_is_gated() -> None:
    """The trap the measurement is built around: a closed circuit of noise accumulates an integer
    winding just as a circle does. Only monotonicity separates them."""
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(P, 8))
    basis = _plane(noise)
    assert abs(winding(noise, basis)) >= 1.0 - 1e-9, "noise winds; the guard cannot be the winding"
    assert monotonicity(noise, basis) < 0.7
    assert planarity(noise) < 0.6


@pytest.mark.parametrize("direction", [1, -1])
def test_a_circle_is_monotone_and_planar(direction: int) -> None:
    loop = _circle(direction)
    assert monotonicity(loop, _plane(loop)) == pytest.approx(1.0)
    assert planarity(loop) == pytest.approx(1.0)


def test_a_reversed_loop_aligns_under_reflection_and_not_under_rotation() -> None:
    """The claim `a - b = a + (-b)` in its most direct form: one loop is the other re-indexed."""
    forward = _circle(1)
    backward = np.roll(forward[::-1], 5, axis=0)
    reflected, _ = alignment(backward, forward, reverse=True)
    rotated, _ = alignment(backward, forward, reverse=False)
    assert reflected == pytest.approx(0.0, abs=1e-9)
    assert rotated > 0.5


def test_a_shifted_loop_aligns_under_rotation_instead() -> None:
    forward = _circle(1)
    rotated, _ = alignment(np.roll(forward, 9, axis=0), forward, reverse=False)
    assert rotated == pytest.approx(0.0, abs=1e-9)


def test_alignment_separates_a_reversal_from_noise() -> None:
    rng = np.random.default_rng(1)
    forward = _circle(1)
    assert alignment(rng.normal(size=forward.shape), forward, reverse=True)[0] > 0.9


def test_antisymmetry_reads_a_reversal_at_unit_scale() -> None:
    """The statistic the product could not be: a loop at frequency k winds k times, so a product
    of windings is -k^2 and carries the frequency into what should be a degree."""
    rng = np.random.default_rng(0)
    frequencies = rng.integers(1, 20, size=16).astype(float)
    assert antisymmetry(frequencies, -frequencies) == pytest.approx(-1.0)
    assert abs(antisymmetry(frequencies, rng.permutation(frequencies))) < 0.9
