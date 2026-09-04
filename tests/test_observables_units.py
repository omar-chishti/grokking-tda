"""Direct unit tests for the scientific diagnostics the thesis compares.

These guard the numbers that go into the report: the Fourier baselines, the LID
estimator, the loss numerics, and the OrthoGrad projection.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.optim import SGD

from grokking_tda.baselines.fourier import _concentration, _discrete_log_order
from grokking_tda.baselines.lid import two_nn_dimension
from grokking_tda.training.losses import softmax_cross_entropy, stablemax_cross_entropy
from grokking_tda.training.optimizers import OrthoGrad


def test_fourier_concentration_detects_a_single_mode() -> None:
    # An embedding that is a pure cosine along the residue axis -> nearly all power in one mode.
    p, d = 97, 16
    residues = np.arange(p)
    emb = np.cos(2 * np.pi * 3 * residues / p)[:, None] * np.ones((1, d))
    assert _concentration(emb, top_k=1) > 0.95


def test_fourier_concentration_low_for_noise() -> None:
    rng = np.random.default_rng(0)
    assert _concentration(rng.normal(size=(97, 16)), top_k=5) < 0.5


def test_group_fourier_sees_the_discrete_log_circle() -> None:
    # A circle arranged by discrete log: flat under the residue-axis DFT, but a pure
    # cosine after primitive-root reordering. This is the fair baseline for mul/div.
    p, d = 11, 4
    order = _discrete_log_order(p)
    assert order is not None and sorted(order) == list(range(1, p))
    emb = np.zeros((p, d))
    for k, residue in enumerate(order):
        emb[residue] = np.cos(2 * np.pi * k / (p - 1))
    group = _concentration(emb[order], top_k=1)
    naive = _concentration(emb, top_k=1)
    assert group > 0.9
    assert naive < group - 0.2


def test_two_nn_dimension_recovers_line_and_plane() -> None:
    rng = np.random.default_rng(0)
    line = np.c_[rng.uniform(size=500), np.zeros(500)]
    plane = rng.uniform(size=(500, 2))
    assert abs(two_nn_dimension(line) - 1.0) < 0.4
    assert abs(two_nn_dimension(plane) - 2.0) < 0.5


def test_stablemax_is_finite_on_extreme_logits() -> None:
    logits = torch.tensor([[1e3, 0.0, -1e3]])
    targets = torch.tensor([0])
    loss = stablemax_cross_entropy(logits, targets)
    assert torch.isfinite(loss)  # float32 softmax CE would underflow here


def test_softmax_ce_matches_torch_reference() -> None:
    logits = torch.randn(8, 5, dtype=torch.float64)
    targets = torch.randint(0, 5, (8,))
    ref = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(softmax_cross_entropy(logits, targets), ref, atol=1e-10)


def test_orthograd_makes_gradients_orthogonal_to_weights() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    opt = OrthoGrad(SGD([p], lr=0.0))  # lr=0: inspect the projected grad, don't move
    p.grad = torch.tensor([1.0, 0.0])
    opt.step()
    assert torch.dot(p.grad, p.detach()).abs() < 1e-6  # orthogonal to weights
    assert abs(p.grad.norm().item() - 1.0) < 1e-6  # norm preserved


def test_every_configured_observable_is_registered() -> None:
    """A typo in the default list would otherwise surface only mid-sweep, as NaN."""
    import grokking_tda.analysis.task_metrics  # noqa: F401
    import grokking_tda.baselines  # noqa: F401
    import grokking_tda.tda.observables  # noqa: F401
    from grokking_tda.config.schema import AnalysisCfg
    from grokking_tda.observable import OBSERVABLES

    for name in AnalysisCfg().observables:
        assert name in OBSERVABLES, name


def test_fourier_concentration_increases_with_k() -> None:
    """Concentration is monotone in k, which is why the strongest competitor must be
    chosen on the series downstream rather than by maximising the scalar."""
    from grokking_tda.baselines.fourier import FOURIER_K_SWEEP, _concentration

    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(97, 16))
    values = [_concentration(embedding, top_k=k) for k in FOURIER_K_SWEEP]
    assert values == sorted(values)
    assert values[-1] <= 1.0 + 1e-12
