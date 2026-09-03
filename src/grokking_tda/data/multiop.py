"""One model, several modular operations, with the operator as a token.

Every other task here trains one operation per model, so the operation is a choice of config and
never reaches the network. Tokenising it asks what a single-operation model cannot: whether the
network reuses one circle for `a + b` and `a - b`, traversing it in opposite directions, or builds
two. See `Documentation/SideQuest_Orientation_2026-09-03.md`.

The residues keep indices ``0 .. p-1`` so that ``embedding_matrix()``'s slice still returns exactly
the residue rows; the operators and ``=`` sit above them. That ordering is load-bearing.
"""

from __future__ import annotations

import torch

from grokking_tda.config.schema import DataCfg
from grokking_tda.data.modular import OPERATIONS, ModularArithmeticData, TaskMeta


def operators_of(operation: str) -> tuple[str, ...]:
    """The operations a joint run mixes, from its ``operation`` field.

    Derived rather than stored: ``TaskMeta`` round-trips through the manifest and
    ``Run.rebuild_model`` reconstructs it positionally, so an extra field there is an extra field
    every reader has to know about. Operator ``i`` is token ``p + i`` and ``=`` is ``p + n``.
    """
    return tuple(op.strip() for op in operation.split("+") if op.strip())


def build_multiop_data(cfg: DataCfg, seed: int) -> ModularArithmeticData:
    operators = operators_of(cfg.operation)
    if len(operators) < 2:
        raise ValueError(
            f"multiop needs two or more operations joined by '+', got {cfg.operation!r}; "
            "a single operation is the modular_arithmetic task"
        )
    unknown = [op for op in operators if op not in OPERATIONS]
    if unknown:
        raise ValueError(f"unknown operation(s) {unknown}; choices: {sorted(OPERATIONS)}")
    if "div" in operators:
        raise ValueError("'div' excludes b = 0, so its example set is a different size; not mixed")

    p = cfg.modulus
    if p <= 1:
        raise ValueError("modulus must be > 1")
    if not 0.0 < cfg.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    equals_token = p + len(operators)

    blocks, targets = [], []
    for i, op in enumerate(operators):
        token = torch.full((p * p,), p + i, dtype=torch.long)
        equals = torch.full((p * p,), equals_token, dtype=torch.long)
        blocks.append(torch.stack([a, token, b, equals], dim=1).long())
        targets.append(OPERATIONS[op](a, b, p).long())
    inputs = torch.cat(blocks)
    target = torch.cat(targets)
    n = inputs.shape[0]

    if cfg.label_permutation:
        perm_gen = torch.Generator().manual_seed(seed + 9871)
        target = target[torch.randperm(n, generator=perm_gen)]

    # one split over the union: an operator-specific split would confound "held out" with
    # "held out under this operator", which is a different experiment
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_train = max(1, min(int(round(cfg.train_fraction * n)), n - 1))
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True

    meta = TaskMeta(
        operation="+".join(operators),
        modulus=p,
        vocab_size=p + len(operators) + 1,
        num_classes=p,
        seq_len=4,
        equals_token=equals_token,
    )
    return ModularArithmeticData(inputs, target, train_mask, meta)
