"""Extract a representation matrix from a snapshot.

``embedding`` is read straight from the snapshot (always cached, cheap). ``hidden``
and ``logits`` are read from the snapshot if cached, otherwise *recomputed* by
rebuilding the model from weights and running the dataset — so we never have to
store large activation tensors, yet can still analyse them losslessly.

``split`` restricts hidden/logit rows to the train or test inputs. Tang et al. build
hidden-state point clouds from the **test set** (at the answer-token position, which
is what the models' ``hidden_hook`` already captures); analysing "all" would leak the
train/test distinction into the topology. The split mask is rebuilt deterministically
from the manifest's data config + seed, so cached full-table representations can be
sliced after the fact.
"""

from __future__ import annotations

import numpy as np
import torch

from grokking_tda.artifacts.reader import Run, Snapshot

# Datasets are deterministic in (data config, seed) and small; cache per process so
# per-snapshot split lookups do not rebuild the tensors every time.
_DATA_CACHE: dict[tuple, object] = {}


def _dataset_for(run: Run):
    from grokking_tda.config.schema import DataCfg
    from grokking_tda.data import build_data

    data_cfg = dict(run.config["data"])
    key = (tuple(sorted(data_cfg.items())), int(run.config["seed"]))
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = build_data(DataCfg(**data_cfg), int(run.config["seed"]))
    return _DATA_CACHE[key]


def _split_rows(run: Run, matrix: np.ndarray, split: str) -> np.ndarray:
    if split in (None, "all"):
        return matrix
    if split not in {"train", "test"}:
        raise ValueError(f"unknown representation split {split!r}; choices: all, train, test")
    data = _dataset_for(run)
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
    """Return the ``(n, d)`` representation matrix of the requested ``kind``.

    ``split`` applies to ``hidden``/``logits`` only; the embedding matrix is the
    ``p x d`` residue table and has no train/test notion.
    """
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

    # Recompute deterministically from weights, on the requested split only.
    data = _dataset_for(run)
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
