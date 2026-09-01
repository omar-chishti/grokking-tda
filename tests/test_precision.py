"""The precision a run actually got, as opposed to the one it asked for.

MPS has no float64, so a high-precision cross-entropy silently degrades there. The fallback
is a fact about a run and is recorded in its manifest; this pins the rule behind it.
"""

from __future__ import annotations

import pytest
import torch

from grokking_tda.utils.precision import supported_float_dtype, tensor_norm


def test_high_precision_dtype_is_kept_on_cpu() -> None:
    assert supported_float_dtype(torch.float64, torch.device("cpu")) is torch.float64
    assert supported_float_dtype(torch.float32, torch.device("cpu")) is torch.float32


def test_tensor_norm_matches_the_training_metric_over_buffers() -> None:
    """The metric summed over ``parameters()`` and the observable over ``state_dict()``.
    They agree only while no model registers a buffer; this is the case that separated them."""
    model = torch.nn.Linear(3, 2)
    model.register_buffer("running", torch.arange(4.0))
    assert tensor_norm(model.parameters()) != pytest.approx(
        tensor_norm(model.state_dict().values())
    )
    assert tensor_norm(model.state_dict().values()) == pytest.approx(
        tensor_norm(list(model.parameters()) + [model.running])
    )
