"""Device-aware precision: float64 cross-entropy, degraded to float32 on MPS with a warning."""

from __future__ import annotations

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
