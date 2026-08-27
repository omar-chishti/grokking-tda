"""Evaluation: locating the grokking transition and comparing observables to it.

This is where the thesis's quantitative questions live — *when* does generalization
happen, *when* does a topological observable transition, and what is the signed lag
between them (does topology lead?).
"""

from grokking_tda.evaluation.changepoint import changepoint_step, changepoints
from grokking_tda.evaluation.multiplicity import benjamini_hochberg, benjamini_yekutieli
from grokking_tda.evaluation.pid import gaussian_pid, williams_beer_pid
from grokking_tda.evaluation.predictive import (
    early_window_feature_grid,
    early_window_features,
)
from grokking_tda.evaluation.transitions import (
    all_transitions,
    grokking_step,
    grokking_step_sensitivity,
    lead_lag,
    train_convergence_step,
    transition_step,
)

__all__ = [
    "all_transitions",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "changepoint_step",
    "changepoints",
    "gaussian_pid",
    "early_window_feature_grid",
    "early_window_features",
    "grokking_step",
    "grokking_step_sensitivity",
    "lead_lag",
    "train_convergence_step",
    "williams_beer_pid",
    "transition_step",
]
