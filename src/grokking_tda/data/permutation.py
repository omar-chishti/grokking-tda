"""Composition in the symmetric group S_n — the genuinely non-cyclic task.

Every operation in ``data/modular.py`` lives on a cyclic group: addition directly,
and subtraction, multiplication and division through isomorphisms, so a generalising
embedding is expected to be a circle in every case. That makes them a poor test of
whether persistent homology tracks *generalisation* or merely *circles*.

S_n for n >= 3 is non-abelian, so no arrangement of its elements on a circle respects
the group operation. If H1 still rises when this task groks, the signature is not
circle-specific; if it does not, that is a sharp boundary on what the signature
detects. Either outcome is a result.

The task the thesis originally reserved for this role — the Power et al. polynomial
``a^3 + ab mod p`` — was measured on 2026-08-24 not to grok at all: six runs at
fractions 0.5 and 0.7 and weight decays 0.3 and 1.0 reached 150,000 steps with train
accuracy 1.0 and test accuracy 0.021-0.026 against a chance level of 0.0103. Without a
transition there is nothing for topology to track, so the decisive test moves here.

Elements are indexed by their position in lexicographic order, which keeps the
interface identical to the modular tasks: tokens ``[a, b, =]``, answer at the final
position, ``num_classes`` equal to the group order.
"""

from __future__ import annotations

from itertools import permutations
from math import factorial

import torch

from grokking_tda.config.schema import DataCfg
from grokking_tda.data.modular import ModularArithmeticData, TaskMeta


def composition_table(n_symbols: int) -> torch.Tensor:
    """``table[i, j]`` is the index of ``perm_i`` composed with ``perm_j``."""
    elements = list(permutations(range(n_symbols)))
    index = {p: i for i, p in enumerate(elements)}
    order = len(elements)
    table = torch.empty((order, order), dtype=torch.long)
    for i, p in enumerate(elements):
        for j, q in enumerate(elements):
            table[i, j] = index[tuple(p[q[k]] for k in range(n_symbols))]
    return table


def build_permutation_data(cfg: DataCfg, seed: int) -> ModularArithmeticData:
    """Composition in S_n, with the same tokenisation and split as the modular tasks."""
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
