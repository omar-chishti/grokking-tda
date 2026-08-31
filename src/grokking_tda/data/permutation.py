"""Composition in S_n: non-abelian, so no circle respects the operation."""

from __future__ import annotations

from itertools import permutations
from math import factorial

import torch

from grokking_tda.config.schema import DataCfg
from grokking_tda.data.modular import ModularArithmeticData, TaskMeta


def composition_table(n_symbols: int) -> torch.Tensor:
    """``table[i, j]`` is the index of ``perm_i`` composed with ``perm_j``, applied right first."""
    elements = list(permutations(range(n_symbols)))
    index = {p: i for i, p in enumerate(elements)}
    order = len(elements)
    table = torch.empty((order, order), dtype=torch.long)
    for i, p in enumerate(elements):
        for j, q in enumerate(elements):
            table[i, j] = index[tuple(p[q[k]] for k in range(n_symbols))]
    return table


def build_permutation_data(cfg: DataCfg, seed: int) -> ModularArithmeticData:
    n = cfg.n_symbols
    if n < 3:
        raise ValueError("S_n is abelian below n = 3; the point of this task is that it is not")
    order = factorial(n)
    if cfg.modulus != order:
        raise ValueError(
            f"modulus records the group order for this task: expected {order} for S_{n}, "
            f"got {cfg.modulus}. It is what run names and TaskMeta report."
        )

    table = composition_table(n)
    a = torch.arange(order).repeat_interleave(order)
    b = torch.arange(order).repeat(order)
    targets = table[a, b]

    n_pairs = a.shape[0]
    equals = torch.full((n_pairs,), order, dtype=torch.long)
    inputs = torch.stack([a, b, equals], dim=1).long()

    if cfg.label_permutation:
        perm_gen = torch.Generator().manual_seed(seed + 9871)
        targets = targets[torch.randperm(n_pairs, generator=perm_gen)]

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_pairs, generator=generator)
    n_train = max(1, min(int(round(cfg.train_fraction * n_pairs)), n_pairs - 1))
    train_mask = torch.zeros(n_pairs, dtype=torch.bool)
    train_mask[perm[:n_train]] = True

    meta = TaskMeta(
        operation=f"s{n}_composition",
        modulus=order,  # the group order; the residue-axis Fourier baseline is meaningless here
        vocab_size=order + 1,
        num_classes=order,
        seq_len=3,
        equals_token=order,
    )
    return ModularArithmeticData(inputs, targets, train_mask, meta)
