"""Dense recording of the optimisation path, for topology *of* the trajectory.

Snapshots are log-spaced and number in the low hundreds — ample for measuring an
observable over training, far too sparse to treat the trajectory itself as a point
cloud. The persistent-homology dimension of a training path (Birdal et al., NeurIPS
2021) is estimated from a window of consecutive iterates, so it needs thousands.

Storing the full parameter vector that often is wasteful, so each iterate is written
through a fixed random projection. Johnson-Lindenstrauss keeps pairwise distances
close to their true values, and pairwise distances are all persistent homology reads
— so the projected path carries the topology of the real one at a few kilobytes per
recorded step.

The projection is drawn from the run's seed and its shape is recorded, so the same
matrix is reconstructible from the manifest alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from grokking_tda.training.callbacks import Callback


def flat_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


class TrajectoryRecorder(Callback):
    """Record a projected copy of the parameter vector every ``every`` steps."""

    def __init__(self, dim: int, every: int, seed: int) -> None:
        self.dim = dim
        self.every = max(1, every)
        self.seed = seed
        self._projection: torch.Tensor | None = None
        self._steps: list[int] = []
        self._points: list[np.ndarray] = []

    def _project(self, weights: torch.Tensor) -> np.ndarray:
        if self._projection is None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + 7919)
            matrix = torch.randn(weights.numel(), self.dim, generator=generator)
            self._projection = (matrix / self.dim**0.5).to(weights.device)
        return (weights @ self._projection).cpu().numpy()

    def _record(self, trainer) -> None:
        self._steps.append(trainer.step)
        self._points.append(self._project(flat_parameters(trainer.model)))

    def on_train_start(self, trainer) -> None:
        self._record(trainer)

    def on_step_end(self, trainer) -> None:
        if trainer.step % self.every == 0:
            self._record(trainer)

    def on_train_end(self, trainer) -> None:
        path = Path(trainer.writer.run_dir) / "trajectory.npz"
        np.savez_compressed(
            path,
            steps=np.asarray(self._steps),
            points=np.vstack(self._points),
            projection_seed=self.seed + 7919,
        )
