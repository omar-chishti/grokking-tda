"""The Observable abstraction and the per-run runner, with one cached context per snapshot."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from grokking_tda.analysis.representations import dataset_for, extract_representation_matrix
from grokking_tda.artifacts.reader import Run, Snapshot
from grokking_tda.registry import Registry
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.pointcloud import build_point_cloud
from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)


def diagram_cache_digest(cfg, seed: int) -> str:
    """The construction a cached diagram came from; readers of the directory must select on it."""
    pc, hm = cfg.pointcloud, cfg.homology
    key = "|".join(
        str(v)
        for v in (
            cfg.representation,
            cfg.representation_split,
            pc.normalize,
            pc.metric,
            pc.max_points,
            pc.subsample,
            pc.drop_first,
            hm.maxdim,
            hm.coeff,
            hm.thresh,
            seed,
        )
    )
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def stored_analysis_cfg(run: Run):
    """The analysis recipe a run was written with, laid over the current defaults."""
    from omegaconf import OmegaConf

    from grokking_tda.config.schema import AnalysisCfg

    cfg = OmegaConf.structured(AnalysisCfg)
    if "analysis" in run.config:
        cfg = OmegaConf.merge(cfg, run.config["analysis"])
    return cfg


class ObservationContext:
    def __init__(self, run: Run, snapshot: Snapshot, cfg) -> None:
        self.run = run
        self.snapshot = snapshot
        self.cfg = cfg
        self.modulus = int(run.task_meta["modulus"])
        self.seed = int(run.config.get("seed", 0))
        self._point_cloud: np.ndarray | None = None
        self._point_cloud_error: Exception | None = None
        self._diagrams: dict[int, np.ndarray] | None = None
        self._weights: dict | None = None
        self._embedding: np.ndarray | None = None
        self._model = None
        self._dataset = None

    def embedding_matrix(self) -> np.ndarray:
        if self._embedding is None:
            self._embedding = extract_representation_matrix(self.run, self.snapshot, "embedding")
        return self._embedding

    def point_cloud(self) -> np.ndarray:
        # the failure is memoised as well as the value: a bad snapshot would otherwise be
        # reconstructed once per observable, and warn ~28 times about the same thing
        if self._point_cloud_error is not None:
            raise self._point_cloud_error
        if self._point_cloud is None:
            try:
                matrix = extract_representation_matrix(
                    self.run,
                    self.snapshot,
                    self.cfg.representation,
                    self.cfg.representation_split,
                )
                self._point_cloud = build_point_cloud(matrix, self.cfg.pointcloud, seed=self.seed)
            except Exception as exc:
                self._point_cloud_error = exc
                raise
        return self._point_cloud

    def _diagram_cache_path(self):
        digest = diagram_cache_digest(self.cfg, self.seed)
        return (
            self.run.dir / "analysis" / "diagrams" / f"step_{self.snapshot.step:08d}_{digest}.npz"
        )

    def diagrams(self) -> dict[int, np.ndarray]:
        if self._diagrams is not None:
            return self._diagrams
        cache_enabled = bool(self.cfg.cache_diagrams)
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

    def model(self):
        if self._model is None:
            self._model = self.run.rebuild_model(self.snapshot)
        return self._model

    def dataset(self):
        if self._dataset is None:
            self._dataset = dataset_for(self.run)
        return self._dataset


Observable = Callable[[ObservationContext], float]
OBSERVABLES: Registry[float] = Registry("observable")

# Declared at registration: inferring it from first and last value misreads anything
# non-monotone, and a wrong direction enters the lead-lag results silently
OBSERVABLE_DIRECTION: dict[str, str] = {}


def register_observable(name: str, *, direction: str = "rising"):
    """Register an ``(ctx) -> float`` observable, declaring which way it moves at the transition."""
    if direction not in {"rising", "falling", "auto"}:
        raise ValueError(f"unknown direction {direction!r}")
    OBSERVABLE_DIRECTION[name] = direction
    return OBSERVABLES.register(name)


def run_observables(run: Run, cfg) -> pd.DataFrame:
    import grokking_tda.analysis.task_metrics  # noqa: F401
    import grokking_tda.baselines  # noqa: F401
    import grokking_tda.tda.observables  # noqa: F401

    rows: list[dict] = []
    for snapshot in run.snapshots():
        ctx = ObservationContext(run, snapshot, cfg)
        row: dict[str, float] = {"step": snapshot.step}
        for name in cfg.observables:
            # one bad snapshot must not abort a long analysis
            try:
                row[name] = OBSERVABLES.build(name, ctx)
            except Exception as exc:
                logger.warning("observable %r failed at step %d: %s", name, snapshot.step, exc)
                row[name] = float("nan")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
