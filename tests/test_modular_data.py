"""The tasks themselves: the label each operation defines, and the split over it.

Everything measured downstream is a statement about a dataset, so an operation that
computes the wrong label produces a perfectly well-behaved topology of the wrong thing.
The permuted-label control is here too, because it is a task and not a setting: it must
destroy the rule while leaving the label marginals alone.
"""

from __future__ import annotations

import torch

from grokking_tda.config.schema import DataCfg
from grokking_tda.data import build_data


def test_dataset_size_meta_and_split() -> None:
    data = build_data(DataCfg(modulus=11, operation="add", train_fraction=0.3), seed=0)
    assert len(data) == 121
    assert data.meta.vocab_size == 12  # p + 1 ("=" token)
    assert data.meta.num_classes == 11
    n_train = int(data.train_mask.sum())
    assert 0 < n_train < 121


def test_split_is_deterministic_in_seed() -> None:
    a = build_data(DataCfg(modulus=11, operation="add", train_fraction=0.3), seed=0)
    b = build_data(DataCfg(modulus=11, operation="add", train_fraction=0.3), seed=0)
    c = build_data(DataCfg(modulus=11, operation="add", train_fraction=0.3), seed=1)
    assert torch.equal(a.train_mask, b.train_mask)
    assert not torch.equal(a.train_mask, c.train_mask)


def _target_for(data, lhs: int, rhs: int) -> int:
    idx = ((data.inputs[:, 0] == lhs) & (data.inputs[:, 1] == rhs)).nonzero()[0]
    return int(data.targets[idx])


def test_operation_labels() -> None:
    add = build_data(DataCfg(modulus=5, operation="add", train_fraction=0.5), seed=0)
    assert _target_for(add, 2, 4) == (2 + 4) % 5  # == 1
    mul = build_data(DataCfg(modulus=7, operation="mul", train_fraction=0.5), seed=0)
    assert _target_for(mul, 3, 4) == (3 * 4) % 7  # == 5
    sub = build_data(DataCfg(modulus=7, operation="sub", train_fraction=0.5), seed=0)
    assert _target_for(sub, 2, 5) == (2 - 5) % 7  # == 4


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
