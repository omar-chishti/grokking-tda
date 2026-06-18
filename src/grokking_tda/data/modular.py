"""Modular-arithmetic tasks — the canonical grokking benchmark.

We enumerate *all* ``p*p`` pairs ``(a, b)`` (``p*(p-1)`` for ``div``, which excludes
``b = 0``) and label them with ``op(a, b) mod p``. A fixed fraction (the train
fraction) is held as the training set; the rest is test. Inputs are tokenised as the
length-3 sequence ``[a, b, =]`` with vocabulary ``{0, ..., p-1, "="}`` (the "=" token
is index ``p``); the answer is read from the final sequence position. This single
representation serves both the transformer (attends over the 3 tokens) and the MLP
(uses the two operand tokens).

Operations: ``add``/``sub``/``mul`` are the classic tasks — note that all three are
group-isomorphic to addition on a cyclic group (mul via discrete log), so their
generalising embeddings are circles, merely permuted in residue order. ``div``
(``a * b^{-1} mod p``, prime ``p``) behaves like mul. ``poly`` (``a^3 + a*b mod p``,
after Power et al.) is the genuinely **non-cyclic** task used for the decisive
redundancy test. ``label_permutation`` destroys the rule entirely (Tang's control
and the null-model runs).

Everything is built once as full tensors. Modular arithmetic is small enough for
**full-batch gradient descent**, which is the standard grokking regime, so the
default training "batch" is the entire train split.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from grokking_tda.config.schema import DataCfg


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _inverse_table(p: int) -> torch.Tensor:
    """Multiplicative inverses mod prime ``p`` (Fermat); index 0 is a placeholder."""
    return torch.tensor([0] + [pow(i, p - 2, p) for i in range(1, p)], dtype=torch.long)


# Binary operations on residues, implemented on tensors (torch.remainder is
# non-negative, which is what we want for modular labels).
OPERATIONS = {
    "add": lambda a, b, p: torch.remainder(a + b, p),
    "sub": lambda a, b, p: torch.remainder(a - b, p),
    "mul": lambda a, b, p: torch.remainder(a * b, p),
    "div": lambda a, b, p: torch.remainder(a * _inverse_table(p)[b], p),  # requires prime p
    "poly": lambda a, b, p: torch.remainder(a * a * a + a * b, p),  # non-cyclic (Power et al.)
}


@dataclass(frozen=True)
class TaskMeta:
    """Static facts about the task that downstream layers need."""

    operation: str
    modulus: int
    vocab_size: int  # p + 1 (operands plus the "=" token)
    num_classes: int  # p (the answer is a residue mod p)
    seq_len: int  # 3: [a, b, =]
    equals_token: int  # index p


class ModularArithmeticData:
    """Full-dataset tensors plus a deterministic train/test mask."""

    def __init__(
        self,
        inputs: torch.Tensor,  # (N, seq_len) long
        targets: torch.Tensor,  # (N,) long
        train_mask: torch.Tensor,  # (N,) bool
        meta: TaskMeta,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.train_mask = train_mask
        self.meta = meta
        self._materialise_views()

    def _materialise_views(self) -> None:
        # Cache the split once; full-batch training reads these every step.
        self.train_inputs = self.inputs[self.train_mask]
        self.train_targets = self.targets[self.train_mask]
        self.test_inputs = self.inputs[~self.train_mask]
        self.test_targets = self.targets[~self.train_mask]

    # --- device handling -----------------------------------------------------
    def to(self, device: torch.device) -> ModularArithmeticData:
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)
        self.train_mask = self.train_mask.to(device)
        self._materialise_views()
        return self

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def iter_batches(
        self,
        batch_size: int | None,
        *,
        generator: torch.Generator | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield training minibatches; ``batch_size=None`` yields one full batch."""
        x, y = self.train_inputs, self.train_targets
        n = x.shape[0]
        if batch_size is None or batch_size >= n:
            yield x, y
            return
        perm = torch.randperm(n, generator=generator).to(x.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            yield x[idx], y[idx]


def build_modular_data(cfg: DataCfg, seed: int) -> ModularArithmeticData:
    """Construct the modular-arithmetic dataset described by ``cfg``."""
    if cfg.operation not in OPERATIONS:
        raise ValueError(f"unknown operation {cfg.operation!r}; choices: {sorted(OPERATIONS)}")
    p = cfg.modulus
    if p <= 1:
        raise ValueError("modulus must be > 1")
    if not 0.0 < cfg.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if cfg.operation == "div" and not _is_prime(p):
        raise ValueError("operation 'div' requires a prime modulus (Fermat inverse)")

    a = torch.arange(p).repeat_interleave(p)  # 0,0,...,1,1,...
    b = torch.arange(p).repeat(p)  # 0,1,...,p-1,0,1,...
    if cfg.operation == "div":  # b = 0 has no inverse; the task is defined on b != 0
        keep = b != 0
        a, b = a[keep], b[keep]
    n_pairs = a.shape[0]
    equals = torch.full((n_pairs,), p, dtype=torch.long)
    inputs = torch.stack([a, b, equals], dim=1).long()
    targets = OPERATIONS[cfg.operation](a, b, p).long()

    if cfg.label_permutation:
        # Destroy the rule but keep the label marginals: a seeded global permutation.
        perm_gen = torch.Generator().manual_seed(seed + 9871)
        targets = targets[torch.randperm(n_pairs, generator=perm_gen)]

    # Deterministic split (seed-controlled): grokking is sensitive to this.
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_pairs, generator=generator)
    n_train = int(round(cfg.train_fraction * n_pairs))
    n_train = max(1, min(n_train, n_pairs - 1))  # never empty train/test
    train_mask = torch.zeros(n_pairs, dtype=torch.bool)
    train_mask[perm[:n_train]] = True

    meta = TaskMeta(
        operation=cfg.operation,
        modulus=p,
        vocab_size=p + 1,
        num_classes=p,
        seq_len=3,
        equals_token=p,
    )
    return ModularArithmeticData(inputs, targets, train_mask, meta)
