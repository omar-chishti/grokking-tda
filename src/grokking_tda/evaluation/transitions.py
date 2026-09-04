"""Operational definitions, fixed in advance: ``t_g`` and ``t_c`` threshold crossings,
``t_top`` a midpoint crossing from the trough, ``delta = t_g - t_top``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from grokking_tda.observable import OBSERVABLE_DIRECTION, ensure_builtins


def _first_crossing(steps: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    hits = np.where(values >= threshold)[0]
    return int(steps[hits[0]]) if hits.size else None


def orient(values: np.ndarray, direction: str) -> np.ndarray:
    """The series as a rising one, so both detectors resolve ``auto`` the same way."""
    if direction == "auto":
        finite = values[np.isfinite(values)]
        direction = "falling" if finite.size >= 2 and finite[-1] < finite[0] else "rising"
    if direction == "falling":
        return -values
    if direction != "rising":
        raise ValueError(f"unknown direction {direction!r}; choices: rising, falling, auto")
    return values


def grokking_step(metrics: pd.DataFrame, acc_threshold: float = 0.9) -> int | None:
    if metrics.empty or "test_acc" not in metrics:
        return None
    m = metrics.sort_values("step")
    return _first_crossing(m["step"].to_numpy(), m["test_acc"].to_numpy(), acc_threshold)


def train_convergence_step(metrics: pd.DataFrame, acc_threshold: float = 0.99) -> int | None:
    if metrics.empty or "train_acc" not in metrics:
        return None
    m = metrics.sort_values("step")
    return _first_crossing(m["step"].to_numpy(), m["train_acc"].to_numpy(), acc_threshold)


def transition_step(
    steps: np.ndarray,
    values: np.ndarray,
    direction: str = "rising",
    compare: str = "trough",
) -> int | None:
    """Midpoint crossing from the trough; ``None`` where the observable never rises."""
    steps = np.asarray(steps)
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return None
    values = orient(values, direction)
    if compare == "global":
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if hi <= lo:
            return None
        return _first_crossing(steps, values, lo + 0.5 * (hi - lo))
    if compare != "trough":
        raise ValueError(f"unknown compare {compare!r}; choices: trough, global")
    # The rise from the trough to the highest value that *follows* it: the global maximum is
    # often the initialisation transient
    trough = int(np.nanargmin(values))
    peak = trough + int(np.nanargmax(values[trough:]))
    if peak == trough:
        return None
    lo, hi = values[trough], values[peak]
    return _first_crossing(steps[trough:], values[trough:], lo + 0.5 * (hi - lo))


def grokking_step_sensitivity(
    metrics: pd.DataFrame, observables: pd.DataFrame | None = None
) -> dict[str, int | None]:
    """``t_g`` under every definition the thesis reports, so the choice is auditable."""
    out: dict[str, int | None] = {
        f"threshold_{t}": grokking_step(metrics, t) for t in (0.8, 0.9, 0.95)
    }
    m = metrics.sort_values("step")
    out["midpoint"] = (
        transition_step(m["step"].to_numpy(), m["test_acc"].to_numpy(dtype=float))
        if "test_acc" in m
        else None
    )
    out["midpoint_novel"] = None
    if observables is not None and "test_acc_novel" in observables:
        o = observables.sort_values("step")
        out["midpoint_novel"] = transition_step(
            o["step"].to_numpy(), o["test_acc_novel"].to_numpy(dtype=float)
        )
    return out


def all_transitions(
    metrics: pd.DataFrame,
    observables: pd.DataFrame,
    acc_threshold: float = 0.9,
) -> dict[str, dict[str, int | None]]:
    """Transition step and signed lag for every observable, under both detectors."""
    # deferred because `changepoint` imports `orient` from this module: a genuine cycle, unlike
    # the direction table, which now comes from a leaf
    from grokking_tda.evaluation.changepoint import changepoint_step

    ensure_builtins()
    t_g = grokking_step(metrics, acc_threshold)
    obs = observables.sort_values("step")
    steps = obs["step"].to_numpy()
    out: dict[str, dict[str, int | None]] = {}
    for column in obs.columns:
        if column == "step":
            continue
        direction = OBSERVABLE_DIRECTION.get(column, "auto")
        series = obs[column].to_numpy(dtype=float)
        t_top = transition_step(steps, series, direction=direction)
        delta = (t_g - t_top) if (t_g is not None and t_top is not None) else None
        t_cp = changepoint_step(steps, series, direction=direction)
        out[column] = {
            "t_top": t_top,
            "delta": delta,
            "t_changepoint": t_cp,
            "delta_changepoint": (t_g - t_cp) if (t_g is not None and t_cp is not None) else None,
        }
    return out


def lead_lag(
    metrics: pd.DataFrame,
    observables: pd.DataFrame,
    observable: str = "h1_max_persistence",
    acc_threshold: float = 0.9,
) -> dict[str, Any]:
    t_g = grokking_step(metrics, acc_threshold)
    t_c = train_convergence_step(metrics)
    t_top = None
    if observable in observables:
        ensure_builtins()
        obs = observables.sort_values("step")
        t_top = transition_step(
            obs["step"].to_numpy(),
            obs[observable].to_numpy(),
            direction=OBSERVABLE_DIRECTION.get(observable, "auto"),
        )
    delta = (t_g - t_top) if (t_g is not None and t_top is not None) else None
    return {
        "train_convergence_step": t_c,
        "grokking_step": t_g,
        "topological_transition_step": t_top,
        "observable": observable,
        "lead_lag_steps": delta,  # > 0: topology leads generalisation
    }
