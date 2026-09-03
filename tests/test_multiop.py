"""The joint-operator task, and the guarantee that adding it changed nothing that existed."""

from __future__ import annotations

import pytest
import torch

from grokking_tda.config.schema import DataCfg, ModelCfg
from grokking_tda.data import build_data
from grokking_tda.data.modular import OPERATIONS
from grokking_tda.data.multiop import operators_of
from grokking_tda.models.mlp import build_mlp

P = 11


def _data(operation: str = "add+sub", **kw) -> object:
    cfg = DataCfg(task="modular_multiop", operation=operation, modulus=P,
                  train_fraction=0.3, **kw)
    return build_data(cfg, seed=0)


def test_one_block_of_examples_per_operator() -> None:
    d = _data()
    assert len(d) == 2 * P * P
    assert operators_of(d.meta.operation) == ("add", "sub")


def test_residues_keep_the_low_indices() -> None:
    """``embedding_matrix()`` slices ``[:p]``; if the operators sat below the residues it would
    return operator rows and every downstream cloud would be wrong."""
    d = _data()
    assert d.meta.equals_token == P + 2
    assert d.meta.vocab_size == P + 3
    assert int(d.inputs[:, [0, 2]].max()) < P


def test_the_operator_token_selects_the_label() -> None:
    d = _data()
    a, op, b = d.inputs[:, 0], d.inputs[:, 1], d.inputs[:, 2]
    for i, name in enumerate(operators_of(d.meta.operation)):
        sel = op == P + i
        expected = OPERATIONS[name](a[sel], b[sel], P)
        assert torch.equal(d.targets[sel], expected)


def test_the_meta_round_trips_through_the_artifact_reader() -> None:
    """``Run.rebuild_model`` does ``TaskMeta(**stored)``; an extra field there breaks every
    joint run at analysis time, which a smoke run found and this keeps found."""
    from dataclasses import asdict

    from grokking_tda.data.modular import TaskMeta

    d = _data()
    assert type(d.meta) is TaskMeta
    assert TaskMeta(**asdict(d.meta)) == d.meta


def test_a_single_operation_is_rejected() -> None:
    with pytest.raises(ValueError, match="two or more"):
        _data(operation="add")


def test_division_is_rejected_because_its_example_set_differs() -> None:
    with pytest.raises(ValueError, match="div"):
        _data(operation="add+div")


def test_label_permutation_keeps_the_marginals() -> None:
    plain, permuted = _data(), _data(label_permutation=True)
    assert torch.equal(plain.targets.bincount(minlength=P), permuted.targets.bincount(minlength=P))
    assert not torch.equal(plain.targets, permuted.targets)


def test_mlp_at_two_operands_is_unchanged() -> None:
    """The joint task needs a three-token MLP. The default must stay exactly what it was."""
    cfg = ModelCfg(name="mlp")
    binary = build_data(DataCfg(operation="add", modulus=P, train_fraction=0.3), seed=0)
    model = build_mlp(cfg, binary.meta)
    assert model.n_operands == 2

    torch.manual_seed(0)
    reference = torch.nn.Embedding(binary.meta.vocab_size, cfg.embedding_dim)
    model.embed.weight.data.copy_(reference.weight.data)
    out = model(binary.inputs[:8])
    # the "=" column cannot reach a two-operand MLP, so scrambling it changes nothing
    scrambled = binary.inputs[:8].clone()
    scrambled[:, 2] = 0
    assert torch.equal(out, model(scrambled))


def test_mlp_reads_the_operator_when_the_task_has_one() -> None:
    d = _data()
    model = build_mlp(ModelCfg(name="mlp"), d.meta)
    assert model.n_operands == 3
    same_pair = d.inputs[[0, P * P]]          # (0, +, 0) and (0, -, 0): identical but the operator
    assert not torch.equal(same_pair[0], same_pair[1])
    out = model(same_pair)
    assert not torch.allclose(out[0], out[1]), "the operator token is not reaching the network"
