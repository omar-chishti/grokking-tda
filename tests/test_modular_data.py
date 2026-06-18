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
