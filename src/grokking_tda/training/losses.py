"""Loss functions, with the numerics that matter for grokking.

``softmax_ce`` computes cross-entropy in float64 by default: float32 log-softmax
underflows on very confident logits, producing loss spikes and "dodgy gradients"
(Nanda et al.). ``stablemax_ce`` replaces softmax with StableMax (Prieto et al.),
which avoids Softmax Collapse and enables grokking without weight decay — the loss
side of the intervention experiments.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from grokking_tda.utils.precision import supported_float_dtype

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _gather_logprob(logprobs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return logprobs.gather(dim=-1, index=targets[:, None]).squeeze(-1)


def softmax_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    work = supported_float_dtype(dtype, logits.device)
    logprobs = F.log_softmax(logits.to(work), dim=-1)
    return -_gather_logprob(logprobs, targets).mean()


def stablemax_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    # StableMax: s(x) = x + 1 (x >= 0) else 1 / (1 - x); normalise to a distribution.
    dtype = supported_float_dtype(dtype, logits.device)
    x = logits.to(dtype)
    s = torch.where(x >= 0, x + 1.0, 1.0 / (1.0 - x))
    probs = s / s.sum(dim=-1, keepdim=True)
    logprobs = torch.log(probs.clamp_min(torch.finfo(dtype).tiny))
    return -_gather_logprob(logprobs, targets).mean()


_LOSSES: dict[str, Callable[..., torch.Tensor]] = {
    "softmax_ce": softmax_cross_entropy,
    "stablemax_ce": stablemax_cross_entropy,
}


def build_loss(name: str, dtype: str = "float64") -> LossFn:
    """Return a ``(logits, targets) -> scalar`` loss bound to the chosen dtype."""
    if name not in _LOSSES:
        raise ValueError(f"unknown loss {name!r}; choices: {sorted(_LOSSES)}")
    torch_dtype = getattr(torch, dtype)
    fn = _LOSSES[name]
    return lambda logits, targets: fn(logits, targets, dtype=torch_dtype)
