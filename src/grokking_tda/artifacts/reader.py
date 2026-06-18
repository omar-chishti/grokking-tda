"""Read run artifacts. This is the *only* interface the analysis layer uses.

``Run`` exposes the manifest, the step-indexed metrics (as a DataFrame), and the
list of ``Snapshot``s. A ``Snapshot`` lazily loads weights and cached
representations, and can rebuild the exact model so any representation (hidden
states, logits) can be recomputed deterministically from weights — keeping
snapshots small without losing information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class Snapshot:
    """One saved training state."""

    step: int
    directory: Path

    def load_weights(self, map_location: str = "cpu") -> dict[str, torch.Tensor]:
        # weights_only=True: a state dict is pure tensors, so refuse arbitrary unpickling.
        return torch.load(
            self.directory / "weights.pt", map_location=map_location, weights_only=True
        )

    def load_representations(self) -> dict[str, np.ndarray]:
        with np.load(self.directory / "representations.npz") as data:
            return {key: data[key] for key in data.files}

    def representation(self, key: str) -> np.ndarray | None:
        path = self.directory / "representations.npz"
        with np.load(path) as data:
            return data[key] if key in data.files else None


class Run:
    """A read-only view over a single run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.dir = Path(run_dir)
        if not (self.dir / "manifest.json").exists():
            raise FileNotFoundError(f"no manifest.json in {self.dir}")
        self.manifest = json.loads((self.dir / "manifest.json").read_text())

    @property
    def run_name(self) -> str:
        return self.manifest["run_name"]

    @property
    def config(self) -> dict:
        return self.manifest["config"]

    @property
    def task_meta(self) -> dict:
        return self.manifest["task_meta"]

    @cached_property
    def metrics(self) -> pd.DataFrame:
        path = self.dir / "metrics.jsonl"
        if not path.exists():
            return pd.DataFrame()
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(records)

    @cached_property
    def events(self) -> pd.DataFrame:
        """Lifecycle/timing events emitted during the run (empty if none)."""
        path = self.dir / "events.jsonl"
        if not path.exists():
            return pd.DataFrame()
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(records)

    def snapshots(self) -> list[Snapshot]:
        index = self.dir / "snapshots" / "index.json"
        steps = json.loads(index.read_text())["steps"] if index.exists() else []
        return [Snapshot(s, self.dir / "snapshots" / f"step_{s:08d}") for s in steps]

    def snapshot(self, step: int) -> Snapshot:
        return Snapshot(step, self.dir / "snapshots" / f"step_{step:08d}")

    def rebuild_model(self, snapshot: Snapshot | None = None) -> nn.Module:
        """Reconstruct the model from the manifest; load ``snapshot`` weights if given.

        Imported lazily to keep the artifact layer free of model dependencies at
        import time.
        """
        from grokking_tda.config.schema import ModelCfg
        from grokking_tda.data.modular import TaskMeta
        from grokking_tda.models import build_model

        model = build_model(ModelCfg(**self.config["model"]), TaskMeta(**self.task_meta))
        if snapshot is not None:
            model.load_state_dict(snapshot.load_weights())
        model.eval()
        return model
