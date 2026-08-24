"""Task accuracy measured on the part of the test set that is actually held out.

The modular-arithmetic benchmark splits *ordered* pairs ``(a, b)`` uniformly at
random. Addition and multiplication are commutative, so a test pair carries the
same label as its transpose, and with train fraction ``f`` that transpose is in the
training set with probability ``f``. A model that has merely memorised the training
table therefore answers ``f`` of the test set correctly — which is exactly the
plateau every grokking curve on modular addition shows before the transition.

Measured on this codebase (p=97, f=0.3, step 507 of a run that grokked at 27,600):
98.9% correct where the transpose was memorised, 0.0% where it was not. The plateau
is commutativity, not partial generalisation; subtraction, which shares the split
geometry but is not commutative, sits at 0.5% instead of 30%.

``test_acc_novel`` restricts accuracy to test pairs the model cannot answer from a
memorised transpose. It is the generalisation the benchmark is meant to measure, and
it starts at zero. Report it alongside the raw series, never instead of it: the raw
series is what prior work plots.
"""

from __future__ import annotations

import torch

from grokking_tda.analysis.observable import ObservationContext, register_observable


def leak_free_test_mask(data) -> torch.Tensor:
    """Test rows whose transpose is not in train *with the same label*.

    Keying on the label rather than on a table of commutative operations keeps the
    rule general: for ``sub``, ``div`` and ``poly`` the transpose almost never shares
    a label, so nothing is excluded and the measure reduces to plain test accuracy.
    """
    inputs, targets = data.inputs, data.targets
    a, b = inputs[:, 0], inputs[:, 1]
    modulus = data.meta.modulus

    # Label of every trained pair, indexed by a * p + b; -1 where the pair is unseen.
    table = torch.full((modulus * modulus,), -1, dtype=targets.dtype, device=targets.device)
    table[(a * modulus + b)[data.train_mask]] = targets[data.train_mask]

    test = ~data.train_mask
    transposed_label = table[(b * modulus + a)[test]]
    return transposed_label != targets[test]


@register_observable("test_acc_novel", direction="rising")
def test_acc_novel(ctx: ObservationContext) -> float:
    """Test accuracy on pairs unreachable from a memorised transpose."""
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
    """Plain test accuracy, on the observable grid so it can be compared like for like."""
    data = ctx.dataset()
    with torch.no_grad():
        predictions = ctx.model()(data.test_inputs).argmax(dim=-1)
    return float((predictions == data.test_targets).float().mean())
