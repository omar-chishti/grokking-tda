"""The precision a run actually got, as opposed to the one it asked for.

MPS has no float64, so a high-precision cross-entropy silently degrades there. The fallback
is a fact about a run and is recorded in its manifest; this pins the rule behind it.
"""

from __future__ import annotations

import torch

from grokking_tda.utils.precision import supported_float_dtype


def test_high_precision_dtype_is_kept_on_cpu() -> None:
    assert supported_float_dtype(torch.float64, torch.device("cpu")) is torch.float64
    assert supported_float_dtype(torch.float32, torch.device("cpu")) is torch.float32
