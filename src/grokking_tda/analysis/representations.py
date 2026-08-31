"""A representation matrix from a snapshot, recomputed from weights when uncached."""

from __future__ import annotations

import numpy as np
import torch

from grokking_tda.artifacts.reader import Run, Snapshot

# Deterministic in (data config, seed), so one build per key serves every snapshot.
_DATA_CACHE: dict[tuple, object] = {}


def dataset_for(run: Run):
    from math import factorial

    from grokking_tda.config.schema import DataCfg
    from grokking_tda.data import build_data

    data_cfg = dict(run.config["data"])
    # An early batch of S_5 runs predates the guard rejecting ``modulus != |S_n|`` and carries
    # the default 97, so rebuilding raises and every observable returns NaN. Recomputed, not
    # trusted: the guard stays a real error for new runs.
    if data_cfg.get("task") == "permutation_group":
        data_cfg["modulus"] = factorial(int(data_cfg["n_symbols"]))
    key = (tuple(sorted(data_cfg.items())), int(run.config["seed"]))
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = build_data(DataCfg(**data_cfg), int(run.config["seed"]))
    return _DATA_CACHE[key]


def _split_rows(run: Run, matrix: np.ndarray, split: str) -> np.ndarray:
    if split in (None, "all"):
        return matrix
    if split not in {"train", "test"}:
        raise ValueError(f"unknown representation split {split!r}; choices: all, train, test")
    data = dataset_for(run)
    mask = data.train_mask.cpu().numpy()
    if split == "test":
        mask = ~mask
    if matrix.shape[0] != mask.shape[0]:
        raise ValueError(
            f"cached representation has {matrix.shape[0]} rows but the dataset has "
            f"{mask.shape[0]}; cannot apply split {split!r}"
        )
    return matrix[mask]


def extract_representation_matrix(
    run: Run, snapshot: Snapshot, kind: str, split: str = "all"
) -> np.ndarray:
    """The ``(n, d)`` matrix of the requested kind; ``split`` applies to hidden and logits only."""
    if kind == "embedding":
        matrix = snapshot.representation("embedding")
        if matrix is None:
            raise KeyError("snapshot has no cached 'embedding' representation")
        return matrix

    if kind not in {"hidden", "logits"}:
        raise ValueError(f"unknown representation kind {kind!r}")

    cached = snapshot.representation(kind)
    if cached is not None:
        return _split_rows(run, cached, split)

    data = dataset_for(run)
    if split == "train":
        inputs = data.train_inputs
    elif split == "test":
        inputs = data.test_inputs
    else:
        inputs = data.inputs
    model = run.rebuild_model(snapshot)
    with torch.no_grad():
        logits, cache = model.run_with_cache(inputs, names=[model.hidden_hook])
    tensor = logits if kind == "logits" else cache[model.hidden_hook]
    return tensor.cpu().numpy()
