"""Early-window features. The window is never the run's own t_g, or its length is the label."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed before any run was analysed: a window chosen post hoc measures the chooser
PREREGISTERED_WINDOWS: tuple[int, ...] = (500, 1000, 2000, 5000)


def window_end_step(window: str, train_convergence: int | None) -> int | None:
    """The step a named window closes at. ``tc`` is the run's own train-convergence step."""
    if window == "tc":
        return train_convergence
    return int(window[1:]) if window.startswith("w") else None


def before_the_event(table: pd.DataFrame) -> pd.DataFrame:
    """Drop runs whose window closes at or after their own grokking step.

    The window above is a step count and knows nothing of where a given run's transition falls,
    so for the fastest conditions it closes late and the features are read after the event they
    are asked to predict. Twenty-one of eighty-three grokked runs were in that position at five
    thousand steps, and they carried the whole of the positive R^2 in the grid. Non-grokkers are
    kept: they are the negative class of the classification, not a leak.
    """
    grokked = table["grokking_step"].notna()
    after = table["window_step"].isna() | (table["grokking_step"] > table["window_step"])
    return table[~grokked | after]


def _trend(steps: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return 0.0
    x = np.log1p(steps[mask].astype(float))  # against log step: the snapshot grid is logarithmic
    y = values[mask].astype(float)
    if np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def early_window_features(
    observables: pd.DataFrame, until_step: int
) -> dict[str, float]:
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
    grid = {f"w{int(w)}": early_window_features(observables, int(w)) for w in windows}
    if train_convergence is not None:
        grid["tc"] = early_window_features(observables, int(train_convergence))
    return grid
