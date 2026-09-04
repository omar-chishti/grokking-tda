"""Task accuracy on the part of the test set a memorised transpose cannot answer."""

from __future__ import annotations

import torch

from grokking_tda.analysis.context import ObservationContext
from grokking_tda.observable import register_observable


def leak_free_test_mask(data) -> torch.Tensor:
    """Test rows no memorised transpose answers.

    Keyed on the label rather than a list of commutative operations, so sub, div and poly need
    no special case: their transposes almost never share a label and nothing is excluded.
    """
    inputs, targets = data.inputs, data.targets
    a, b = inputs[:, 0], inputs[:, 1]
    modulus = data.meta.modulus

    # every trained pair's label, indexed a * p + b; -1 where unseen
    table = torch.full((modulus * modulus,), -1, dtype=targets.dtype, device=targets.device)
    table[(a * modulus + b)[data.train_mask]] = targets[data.train_mask]

    test = ~data.train_mask
    transposed_label = table[(b * modulus + a)[test]]
    return transposed_label != targets[test]


@register_observable("test_acc_novel", direction="rising")
def test_acc_novel(ctx: ObservationContext) -> float:
    data = ctx.dataset()
    novel = leak_free_test_mask(data)
    inputs = data.test_inputs[novel]
    targets = data.test_targets[novel]
    if inputs.shape[0] == 0:
        return float("nan")
    with torch.no_grad():
        predictions = ctx.model()(inputs).argmax(dim=-1)
    return float((predictions == targets).float().mean())


@register_observable("test_acc", direction="rising")
def test_acc(ctx: ObservationContext) -> float:
    data = ctx.dataset()
    with torch.no_grad():
        predictions = ctx.model()(data.test_inputs).argmax(dim=-1)
    return float((predictions == data.test_targets).float().mean())
