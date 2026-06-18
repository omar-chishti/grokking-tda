"""Topological data analysis: point clouds, persistent homology, summaries."""

from grokking_tda.tda.distances import diagram_distance, trajectory_velocity
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.pointcloud import build_point_cloud
from grokking_tda.tda.significance import bootstrap_summary_ci, random_init_null
from grokking_tda.tda.summaries import (
    max_persistence,
    n_features,
    persistence_entropy,
    total_persistence,
)
from grokking_tda.tda.trajectory import (
    auto_crocker,
    betti_at_scales,
    crocker_from_diagrams,
    crocker_matrix,
)

__all__ = [
    "auto_crocker",
    "betti_at_scales",
    "bootstrap_summary_ci",
    "build_point_cloud",
    "compute_persistence",
    "crocker_from_diagrams",
    "crocker_matrix",
    "diagram_distance",
    "max_persistence",
    "n_features",
    "persistence_entropy",
    "random_init_null",
    "total_persistence",
    "trajectory_velocity",
]

