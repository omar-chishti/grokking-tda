"""Device-aware precision: float64 cross-entropy, degraded to float32 on MPS with a warning."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)
_warned_devices: set[str] = set()


def supported_float_dtype(requested: torch.dtype, device: torch.device | str) -> torch.dtype:
    device_type = torch.device(device).type
    if requested == torch.float64 and device_type == "mps":
        if device_type not in _warned_devices:
            logger.warning("float64 is unsupported on MPS; using float32 for high-precision ops")
            _warned_devices.add(device_type)
        return torch.float32
    return requested


def tensor_norm(tensors: Iterable[torch.Tensor]) -> float:
    """L2 norm over an iterable of tensors, accumulated in float64 on the CPU.

    The training metric and the ``weight_norm`` observable both call this. They summed over
    ``parameters()`` and over a snapshot's ``state_dict()`` respectively, which agree only
    while no model registers a buffer.
    """
    total = torch.zeros((), dtype=torch.float64)  # CPU: MPS rejects float64 even as a cast target
    for tensor in tensors:
        if torch.is_floating_point(tensor):
            total += (tensor.detach().cpu().double() ** 2).sum()
    return float(total.sqrt())
