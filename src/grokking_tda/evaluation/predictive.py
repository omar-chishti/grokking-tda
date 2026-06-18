"""Early-window features for the predictive-value question.

Can topology (or any observable) predict the grokking step from an *early* window of
training — before generalization is visible? This module extracts per-run early-window
features (level and trend of each observable up to a cutoff step). Fitting a predictor
across many runs is a multi-run analysis layered on top of these features.

**Leakage rule:** the window must never be the run's own grokking step — the window
*length* would then encode the label the predictor is judged on. Windows are
pre-registered absolute step counts (plus, optionally, the train-convergence step
``t_c``, which is observable without test data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Pre-registered prediction windows (steps). Fixed in advance in
# Documentation/Experiment_Design.md §4; do not tune these post hoc.
PREREGISTERED_WINDOWS: tuple[int, ...] = (500, 1000, 2000, 5000)


def _trend(steps: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of ``values`` against (log) step; 0 if degenerate."""
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return 0.0
    x = np.log1p(steps[mask].astype(float))
    y = values[mask].astype(float)
    if np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def early_window_features(
    observables: pd.DataFrame, until_step: int
) -> dict[str, float]:
    """Mean and trend of each observable over snapshots up to ``until_step``."""
    if observables.empty:
        return {}
    window = observables[observables["step"] <= until_step]
    if window.empty:
        window = observables.iloc[:1]
    feats: dict[str, float] = {}
    steps = window["step"].to_numpy()
    for column in observables.columns:
        if column == "step":
            continue
        values = window[column].to_numpy(dtype=float)
        feats[f"{column}__mean"] = float(np.nanmean(values))
        feats[f"{column}__trend"] = _trend(steps, values)
    return feats


def early_window_feature_grid(
    observables: pd.DataFrame,
    windows: tuple[int, ...] = PREREGISTERED_WINDOWS,
    train_convergence: int | None = None,
) -> dict[str, dict[str, float]]:
    """Features at every pre-registered window (keys ``w500``, ..., and ``tc``).

    ``train_convergence`` adds a ``tc``-anchored window — defensible because t_c is
    observable without the test set. The run's grokking step is deliberately not an
    option (see the module docstring).
    """
    grid = {f"w{int(w)}": early_window_features(observables, int(w)) for w in windows}
    if train_convergence is not None:
        grid["tc"] = early_window_features(observables, int(train_convergence))
    return grid
