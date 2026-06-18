"""Deterministic seeding and device selection.

Reproducibility is a first-class requirement for a thesis: every run records its
seed and we make the numerics as deterministic as the backend allows. Grokking is
seed-sensitive, so this is not optional polish.
"""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch RNGs and (optionally) force deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN determinism (no-op off CUDA) and a best-effort global flag.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Not all ops have deterministic implementations; warn rather than crash.
        with contextlib.suppress(Exception):  # pragma: no cover - backend dependent
            torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device string, gracefully degrading cuda -> mps -> cpu.

    ``"auto"`` picks the best available backend (CUDA on the cluster, MPS on a Mac,
    CPU otherwise). An explicit request that is unavailable falls back to CPU.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(requested)
