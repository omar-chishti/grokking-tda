"""The commutativity correction, the scale-normalised summaries, and declared directions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from grokking_tda.analysis.observable import OBSERVABLE_DIRECTION, register_observable
from grokking_tda.analysis.task_metrics import leak_free_test_mask
from grokking_tda.config.schema import DataCfg
from grokking_tda.data import build_data
from grokking_tda.evaluation.transitions import transition_step


@pytest.mark.parametrize("fraction", [0.2, 0.3, 0.5])
def test_commutative_leakage_matches_train_fraction(fraction):
    """For addition, the leaked share of the test set is the train fraction.

    Averaged over seeds at the modulus the experiments actually use: the split is
    random, so a single small-modulus draw carries several points of sampling noise.
    """
    leaked = [
        (~leak_free_test_mask(build_data(DataCfg("modular_arithmetic", "add", 97, fraction), s)))
        .float()
        .mean()
        .item()
        for s in range(5)
    ]
    assert sum(leaked) / len(leaked) == pytest.approx(fraction, abs=0.02)


def test_non_commutative_operations_leak_nothing():
    """Subtraction shares the split geometry but not the labels, so nothing is excluded."""
    data = build_data(DataCfg("modular_arithmetic", "sub", 31, 0.3), seed=0)
    novel = leak_free_test_mask(data)
    # a - b == b - a only when a == b, which is a vanishing share of the table.
    assert novel.float().mean().item() > 0.97


def test_leak_free_mask_length_matches_test_split():
    data = build_data(DataCfg("modular_arithmetic", "add", 17, 0.4), seed=1)
    assert leak_free_test_mask(data).shape[0] == data.test_inputs.shape[0]


def test_normalised_persistence_is_scale_invariant():
    """Doubling the cloud doubles raw persistence but leaves the normalised value fixed."""
    from grokking_tda.config.schema import HomologyCfg
    from grokking_tda.tda.homology import compute_persistence
    from grokking_tda.tda.summaries import max_persistence

    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])

    def summarise(points):
        diagrams = compute_persistence(points, HomologyCfg(maxdim=1))
        h0 = diagrams[0]
        scale = h0[np.isfinite(h0[:, 1])][:, 1].max()
        return max_persistence(diagrams.get(1)), max_persistence(diagrams.get(1)) / scale

    raw_small, norm_small = summarise(circle)
    raw_big, norm_big = summarise(circle * 3.0)

    assert raw_big == pytest.approx(3.0 * raw_small, rel=1e-6)
    assert norm_big == pytest.approx(norm_small, rel=1e-6)


def test_registered_observables_declare_a_direction():
    import grokking_tda.analysis.task_metrics  # noqa: F401
    import grokking_tda.baselines  # noqa: F401
    import grokking_tda.tda.observables  # noqa: F401

    assert OBSERVABLE_DIRECTION["lid"] == "falling"
    assert OBSERVABLE_DIRECTION["h1_max_persistence"] == "rising"
    assert OBSERVABLE_DIRECTION["test_acc_novel"] == "rising"
    assert set(OBSERVABLE_DIRECTION.values()) <= {"rising", "falling", "auto"}


def test_register_observable_rejects_an_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        register_observable("_bad_direction", direction="sideways")


def test_declared_direction_beats_first_versus_last_inference():
    """A falling series that ends above where it started is still falling."""
    steps = np.arange(0, 100, 10)
    values = np.array([5.0, 5.0, 4.0, 2.0, 0.0, 0.0, 1.0, 3.0, 6.0, 7.0])

    assert transition_step(steps, values, direction="auto") is not None
    falling = transition_step(steps, values, direction="falling")
    rising = transition_step(steps, values, direction="rising")
    assert falling != rising


def test_context_exposes_model_and_dataset(tiny_run):
    """The commutativity observable needs both; they must round-trip from a snapshot."""
    from omegaconf import OmegaConf

    from grokking_tda.analysis.observable import ObservationContext
    from grokking_tda.artifacts import Run
    from grokking_tda.config.schema import AnalysisCfg

    run = Run(tiny_run[0])
    ctx = ObservationContext(run, run.snapshots()[-1], OmegaConf.structured(AnalysisCfg))
    assert ctx.dataset().inputs.shape[0] > 0
    with torch.no_grad():
        assert ctx.model()(ctx.dataset().test_inputs).ndim == 2


def test_task_metric_observables_run_on_a_real_snapshot(tiny_run):
    from omegaconf import OmegaConf

    from grokking_tda.analysis.observable import ObservationContext
    from grokking_tda.analysis.task_metrics import test_acc, test_acc_novel
    from grokking_tda.artifacts import Run
    from grokking_tda.config.schema import AnalysisCfg

    run = Run(tiny_run[0])
    ctx = ObservationContext(run, run.snapshots()[-1], OmegaConf.structured(AnalysisCfg))
    for value in (test_acc(ctx), test_acc_novel(ctx)):
        assert 0.0 <= value <= 1.0
