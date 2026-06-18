"""The Observable abstraction and the per-run runner.

An *observable* is any scalar computed from a training snapshot: H1 persistence,
Fourier concentration, weight norm, local intrinsic dimension. Treating them all as
the same kind of object is what makes the central thesis comparison — "what does
topology add over cheaper diagnostics?" — a one-line change in the config's
``observables`` list.

A single :class:`ObservationContext` per snapshot lazily builds (and caches) the
expensive shared artifacts — the point cloud and its persistence diagrams — so many
observables over one snapshot pay that cost once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from grokking_tda.analysis.representations import extract_representation_matrix
from grokking_tda.artifacts.reader import Run, Snapshot
from grokking_tda.registry import Registry
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.pointcloud import build_point_cloud
from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)


class ObservationContext:
    """Shared, lazily-computed inputs for all observables on one snapshot.

    Persistence diagrams are also cached **on disk** (``analysis/diagrams/``), keyed
    by the construction config, because every downstream consumer — observables,
    CROCKER, trajectory velocity, bootstrap — wants the same diagrams and recomputing
    them is the dominant analysis cost.
    """

    def __init__(self, run: Run, snapshot: Snapshot, cfg) -> None:
        self.run = run
        self.snapshot = snapshot
        self.cfg = cfg
        self.modulus = int(run.task_meta["modulus"])
        self.seed = int(run.config.get("seed", 0))
        self._point_cloud: np.ndarray | None = None
        self._diagrams: dict[int, np.ndarray] | None = None
        self._weights: dict | None = None
        self._embedding: np.ndarray | None = None

    def embedding_matrix(self) -> np.ndarray:
        """Always the residue-embedding matrix (the Fourier baseline lives here)."""
        if self._embedding is None:
            self._embedding = extract_representation_matrix(self.run, self.snapshot, "embedding")
        return self._embedding

    def point_cloud(self) -> np.ndarray:
        if self._point_cloud is None:
            matrix = extract_representation_matrix(
                self.run,
                self.snapshot,
                self.cfg.representation,
                getattr(self.cfg, "representation_split", "all"),
            )
            self._point_cloud = build_point_cloud(matrix, self.cfg.pointcloud, seed=self.seed)
        return self._point_cloud

    def _diagram_cache_path(self):
        pc, hm = self.cfg.pointcloud, self.cfg.homology
        key = "|".join(
            str(v)
            for v in (
                self.cfg.representation,
                getattr(self.cfg, "representation_split", "all"),
                pc.normalize,
                pc.metric,
                pc.max_points,
                getattr(pc, "subsample", "random"),
                getattr(pc, "drop_first", False),
                hm.maxdim,
                hm.coeff,
                hm.thresh,
                self.seed,
            )
        )
        digest = hashlib.sha1(key.encode()).hexdigest()[:10]
        return (
            self.run.dir / "analysis" / "diagrams" / f"step_{self.snapshot.step:08d}_{digest}.npz"
        )

    def diagrams(self) -> dict[int, np.ndarray]:
        if self._diagrams is not None:
            return self._diagrams
        cache_enabled = bool(getattr(self.cfg, "cache_diagrams", True))
        path = self._diagram_cache_path()
        if cache_enabled and path.exists():
            with np.load(path) as data:
                self._diagrams = {int(name[3:]): data[name] for name in data.files}
            return self._diagrams
        self._diagrams = compute_persistence(
            self.point_cloud(), self.cfg.homology, self.cfg.pointcloud.metric
        )
        if cache_enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{f"dim{d}": arr for d, arr in self._diagrams.items()})
        return self._diagrams

    def weights(self) -> dict:
        if self._weights is None:
            self._weights = self.snapshot.load_weights()
        return self._weights


# Observables are factories taking a context and returning a scalar.
Observable = Callable[[ObservationContext], float]
OBSERVABLES: Registry[float] = Registry("observable")


def register_observable(name: str):
    """Decorator registering an ``(ctx) -> float`` observable under ``name``."""
    return OBSERVABLES.register(name)


def run_observables(run: Run, cfg) -> pd.DataFrame:
    """Compute every observable in ``cfg.observables`` over every snapshot of ``run``."""
    # Ensure the built-in observables are registered.
    import grokking_tda.baselines  # noqa: F401
    import grokking_tda.tda.observables  # noqa: F401

    rows: list[dict] = []
    for snapshot in run.snapshots():
        ctx = ObservationContext(run, snapshot, cfg)
        row: dict[str, float] = {"step": snapshot.step}
        for name in cfg.observables:
            # One bad snapshot must not abort a long analysis: record NaN and carry on.
            try:
                row[name] = OBSERVABLES.build(name, ctx)
            except Exception as exc:
                logger.warning("observable %r failed at step %d: %s", name, snapshot.step, exc)
                row[name] = float("nan")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
