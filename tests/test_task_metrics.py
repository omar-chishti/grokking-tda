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


def test_ph_dimension_recovers_a_known_dimension():
    """Uniform points in the plane should read as roughly two-dimensional."""
    from grokking_tda.tda.phdim import ph_dimension

    rng = np.random.default_rng(0)
    plane = ph_dimension(rng.random((400, 2)), seed=0)
    line = ph_dimension(rng.random((400, 1)), seed=0)

    assert 1.5 < plane < 3.0
    assert line < plane


def test_ph_dimension_is_nan_on_too_few_points():
    from grokking_tda.tda.phdim import ph_dimension

    assert np.isnan(ph_dimension(np.random.default_rng(0).random((10, 3))))


def test_ph_dimension_over_training_is_labelled_without_look_ahead():
    from grokking_tda.tda.phdim import ph_dimension_over_training

    rng = np.random.default_rng(0)
    steps = np.arange(0, 1000, 5)
    frame = ph_dimension_over_training(steps, rng.random((len(steps), 8)), window=100, stride=50)

    assert not frame.empty
    assert frame["step"].is_monotonic_increasing
    assert frame["step"].iloc[0] == steps[99]


def test_s5_composition_table_is_a_group():
    """Closure, an identity, and an inverse for every element."""
    from grokking_tda.data.permutation import composition_table

    table = composition_table(4)
    order = table.shape[0]
    assert order == 24
    assert int(table.min()) == 0 and int(table.max()) == order - 1

    identity = 0  # lexicographically first permutation is the identity
    assert all(int(table[identity, j]) == j for j in range(order))
    assert all(int(table[i, identity]) == i for i in range(order))
    assert all(identity in {int(table[i, j]) for j in range(order)} for i in range(order))


def test_s5_composition_is_not_abelian():
    from grokking_tda.data.permutation import composition_table

    table = composition_table(4)
    assert not torch.equal(table, table.T)


def test_s5_task_builds_with_the_standard_interface():
    from grokking_tda.config.schema import DataCfg
    from grokking_tda.data import build_data

    data = build_data(DataCfg(task="permutation_group", modulus=120, train_fraction=0.5), seed=0)
    assert data.meta.num_classes == 120
    assert data.meta.vocab_size == 121
    assert data.inputs.shape == (14400, 3)
    assert int(data.train_mask.sum()) == 7200
    assert (data.inputs[:, 2] == data.meta.equals_token).all()


def test_s5_rejects_abelian_symbol_counts():
    from grokking_tda.config.schema import DataCfg
    from grokking_tda.data import build_data

    with pytest.raises(ValueError, match="abelian"):
        build_data(DataCfg(task="permutation_group", n_symbols=2, modulus=2), seed=0)


def test_s5_rejects_a_modulus_that_is_not_the_group_order():
    """The run name is built from modulus, so a wrong one would mislabel the artifacts."""
    from grokking_tda.config.schema import DataCfg
    from grokking_tda.data import build_data

    with pytest.raises(ValueError, match="group order"):
        build_data(DataCfg(task="permutation_group", modulus=97), seed=0)
