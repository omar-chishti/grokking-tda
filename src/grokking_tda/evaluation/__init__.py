"""Evaluation: locating the grokking transition and comparing observables to it.

This is where the thesis's quantitative questions live — *when* does generalization
happen, *when* does a topological observable transition, and what is the signed lag
between them (does topology lead?).
"""

from grokking_tda.evaluation.predictive import (
    early_window_feature_grid,
    early_window_features,
)
from grokking_tda.evaluation.transitions import (
    all_transitions,
    grokking_step,
    lead_lag,
    train_convergence_step,
    transition_step,
)

__all__ = [
    "all_transitions",
    "early_window_feature_grid",
    "early_window_features",
    "grokking_step",
    "lead_lag",
    "train_convergence_step",
    "transition_step",
]
