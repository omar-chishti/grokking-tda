"""Significance machinery: bootstrap confidence sets and null models.

This is the apparatus the reference paper lacks. Two pieces:

- **Bootstrap confidence sets** (after Fasy et al.): subsample the point cloud,
  recompute persistence, and form percentile intervals on max/total persistence —
  uncertainty attached to every headline number.
- **Null models**: the same summaries on point clouds that *cannot* carry the
  signal — freshly-initialised (untrained) weights here; shuffled-label runs are
  produced by training with ``data.label_permutation=true`` and analysed normally.

Multi-run aggregation and multiple-comparison control live in
``analysis/aggregate.py`` and the figure layer, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from grokking_tda.config.schema import HomologyCfg, PointCloudCfg
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.pointcloud import build_point_cloud
from grokking_tda.tda.summaries import max_persistence, total_persistence


def bootstrap_summary_ci(
    points: np.ndarray,
    *,
    homology: HomologyCfg | None = None,
    metric: str = "euclidean",
    dim: int = 1,
    n_boot: int = 200,
    subsample_fraction: float = 0.8,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile CIs on max/total H_dim persistence under point subsampling.

    Returns ``{"max": {lo, median, hi}, "total": {...}, "samples": DataFrame}``.
    """
    homology = homology or HomologyCfg(maxdim=max(dim, 1))
    x = np.asarray(points, dtype=np.float64)
    n = x.shape[0]
    k = max(3, int(round(subsample_fraction * n)))
    rng = np.random.default_rng(seed)
    maxes, totals = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=min(k, n), replace=False)
        diagram = compute_persistence(x[idx], homology, metric).get(dim)
        maxes.append(max_persistence(diagram))
        totals.append(total_persistence(diagram))

    def _ci(values: list[float]) -> dict[str, float]:
        lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        return {"lo": float(lo), "median": float(np.median(values)), "hi": float(hi)}

    return {
        "max": _ci(maxes),
        "total": _ci(totals),
        "samples": pd.DataFrame({"max": maxes, "total": totals}),
    }


def random_init_null(
    run,
    *,
    n_samples: int = 20,
    seed: int = 0,
    pointcloud: PointCloudCfg | None = None,
    homology: HomologyCfg | None = None,
    metric: str = "euclidean",
    dim: int = 1,
) -> pd.DataFrame:
    """H_dim summaries of freshly-initialised (untrained) models of the run's architecture.

    The distribution of max/total persistence under random init is the band a trained
    snapshot's value must exceed before it can be called a signal.
    """
    from grokking_tda.config.schema import ModelCfg
    from grokking_tda.data.modular import TaskMeta
    from grokking_tda.models import build_model

    pointcloud = pointcloud or PointCloudCfg()
    homology = homology or HomologyCfg(maxdim=max(dim, 1))
    rows = []
    for i in range(n_samples):
        # A local generator: seeding the global RNG here would perturb any caller
        # that draws randomness afterwards.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + i)
            model = build_model(ModelCfg(**run.config["model"]), TaskMeta(**run.task_meta))
        embedding = model.embedding_matrix().cpu().numpy()
        cloud = build_point_cloud(embedding, pointcloud, seed=seed + i)
        diagram = compute_persistence(cloud, homology, metric).get(dim)
        rows.append(
            {
                "sample": i,
                "max": max_persistence(diagram),
                "total": total_persistence(diagram),
            }
        )
    return pd.DataFrame(rows)
