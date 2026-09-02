"""The recipe decomposition: which anchor a cell belongs to, and how far it moved."""

from __future__ import annotations

import pytest
from analysis.recipe import ANCHORS, classify


def _config(**moved) -> dict:
    settings = dict(ANCHORS["canonical"], **moved)
    return {
        "train": {
            "batch_size": settings["batch_size"],
            "optimizer": {"lr": settings["lr"], "eps": settings["eps"]},
        },
        "model": {
            "n_layers": settings["n_layers"],
            "act": settings["act"],
            "d_mlp": settings["d_mlp"],
        },
    }


def test_an_anchor_is_named_and_has_moved_nothing() -> None:
    assert classify(_config()) == ("canonical", "—")


def test_one_move_names_the_ingredient() -> None:
    assert classify(_config(n_layers=2)) == ("canonical", "n_layers")


def test_two_moves_are_the_interaction_cell_and_stay_attributable() -> None:
    """The anchors differ in all six factors, so a cell two moves from one is four from the
    other and cannot be claimed by both."""
    assert classify(_config(n_layers=2, lr=3e-3)) == ("canonical", "lr+n_layers")


def test_three_moves_belong_to_no_anchor() -> None:
    with pytest.raises(ValueError):
        classify(_config(n_layers=2, lr=3e-3, act="gelu"))
