"""What counts as a replicate. Getting this wrong pools a re-run into a live condition."""

from __future__ import annotations

from grokking_tda.analysis.identity import REPLICATE_TAGS, condition_key, is_replicate

CONFIG = {
    "model": {"name": "transformer"},
    "data": {"operation": "add", "modulus": 97, "train_fraction": 0.3, "label_permutation": False},
    "train": {
        "loss": "softmax_ce",
        "dense_to": 0,
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1.0},
    },
}


def test_a_main_programme_run_is_not_a_replicate() -> None:
    assert not is_replicate("transformer_add97_f0.3_wd1.0_softmax_ce_s0", CONFIG)


def test_every_tag_marks_a_re_run() -> None:
    for tag in REPLICATE_TAGS:
        assert is_replicate(f"transformer_add97_f0.3_wd1.0{tag}s0", CONFIG), tag


def test_a_stride_one_re_run_stays_out_of_the_condition_table() -> None:
    """It repeats a condition at a finer trajectory stride, so it is the same optimisation path
    recorded differently. Pooled instead, twenty would move an interval the thesis quotes."""
    assert is_replicate("transformer_add113_f0.3_wd0.1_stride1_s0", CONFIG)


def test_a_dense_window_is_caught_by_its_config_when_the_name_does_not_say() -> None:
    config = {**CONFIG, "train": {**CONFIG["train"], "dense_to": 30000}}
    assert is_replicate("transformer_add113_f0.3_wd1.0_softmax_ce_s0", config)


def test_the_condition_key_ignores_the_seed() -> None:
    assert condition_key(CONFIG) == condition_key({**CONFIG, "seed": 4})
