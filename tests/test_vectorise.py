"""Landscapes and images, on diagrams whose vector is known by hand.

The property that matters is the last one: a vectorisation is admissible as a substitute for the
diagram only if it is stable, so a wrong normalisation shows up as a broken Lipschitz bound rather
than as a plausible number.
"""

from __future__ import annotations

import numpy as np

from grokking_tda.tda.vectorise import diagram_extent, landscape, persistence_image

UNIT = (0.0, 1.0)


def test_single_bar_is_one_tent() -> None:
    values = landscape(np.array([[0.0, 1.0]]), extent=UNIT, resolution=21)
    assert values[0].max() == 0.5
    assert np.isclose(np.linspace(0, 1, 21)[values[0].argmax()], 0.5)
    assert not values[1:].any()


def test_duplicate_bars_fill_the_second_layer() -> None:
    doubled = landscape(np.array([[0.0, 1.0], [0.0, 1.0]]), extent=UNIT)
    assert np.array_equal(doubled[0], doubled[1])
    assert not doubled[2:].any()


def test_landscape_is_one_lipschitz_in_the_diagram() -> None:
    """Bubenik's stability bound: sup-norm in, sup-norm out, on a fixed grid."""
    rng = np.random.default_rng(0)
    births = rng.uniform(0, 0.5, size=12)
    diagram = np.c_[births, births + rng.uniform(0.05, 0.5, size=12)]
    for epsilon in (0.01, 0.05):
        perturbed = diagram + rng.uniform(-epsilon, epsilon, size=diagram.shape)
        moved = np.abs(landscape(diagram, extent=UNIT) - landscape(perturbed, extent=UNIT)).max()
        assert moved <= epsilon + 1e-12


def test_empty_diagram_gives_zeros_not_an_error() -> None:
    assert landscape(None, extent=UNIT, layers=3, resolution=7).shape == (3, 7)
    assert not landscape(np.empty((0, 2)), extent=UNIT).any()
    assert not persistence_image(None, extent=UNIT, grid=4).any()


def test_infinite_bars_are_excluded() -> None:
    finite = np.array([[0.0, 1.0]])
    with_essential = np.array([[0.0, 1.0], [0.0, np.inf]])
    assert np.array_equal(landscape(finite, extent=UNIT), landscape(with_essential, extent=UNIT))


def test_image_mass_follows_persistence() -> None:
    short = persistence_image(np.array([[0.0, 0.1]]), extent=UNIT).sum()
    long = persistence_image(np.array([[0.0, 1.0]]), extent=UNIT).sum()
    assert long > 5 * short


def test_extent_spans_the_collection() -> None:
    lo, hi = diagram_extent([np.array([[0.2, 0.6]]), None, np.array([[0.1, 0.9], [0.3, np.inf]])])
    assert (lo, hi) == (0.1, 0.9)
    assert diagram_extent([None]) == (0.0, 1.0)
