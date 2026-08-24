"""Operational definitions of the transition, fixed in advance (no post-hoc tuning).

  - grokking step ``t_g``: first step with test accuracy >= threshold (default 0.9)
  - train-convergence ``t_c``: first step with train accuracy >= threshold (0.99)
  - topological transition ``t_top``: midpoint-crossing of a (rising) observable
  - lead/lag ``delta = t_g - t_top``  (delta > 0 means topology leads generalization)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _first_crossing(steps: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    """First step at which a rising series reaches ``threshold`` (or None)."""
    hits = np.where(values >= threshold)[0]
    return int(steps[hits[0]]) if hits.size else None


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
    steps: np.ndarray, values: np.ndarray, direction: str = "rising"
) -> int | None:
    """Midpoint-crossing step of an observable: first reach min + 0.5*(max-min).

    A simple, assumption-light change-point proxy. For a near-monotone rise (the
    H1 signature) it returns the step where the observable is halfway to its peak.
    ``direction="falling"`` negates the series first (LID falls at grokking);
    ``"auto"`` infers the direction from the first and last finite values.
    """
    steps = np.asarray(steps)
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return None
    if direction == "auto":
        finite = values[np.isfinite(values)]
        direction = "falling" if finite.size >= 2 and finite[-1] < finite[0] else "rising"
    if direction == "falling":
        values = -values
    elif direction != "rising":
        raise ValueError(f"unknown direction {direction!r}; choices: rising, falling, auto")
    lo, hi = np.nanmin(values), np.nanmax(values)
    if hi <= lo:
        return None
    return _first_crossing(steps, values, lo + 0.5 * (hi - lo))


def all_transitions(
    metrics: pd.DataFrame,
    observables: pd.DataFrame,
    acc_threshold: float = 0.9,
) -> dict[str, dict[str, int | None]]:
    """Transition step and signed lag for *every* observable column.

    The lead-lag forest plot needs ``t_top`` per observable, not only the headline
    one. Each observable declares which way it moves at registration; only columns
    with no declaration fall back to inferring it from the series.
    """
    from grokking_tda.analysis.observable import OBSERVABLE_DIRECTION

    t_g = grokking_step(metrics, acc_threshold)
    obs = observables.sort_values("step")
    steps = obs["step"].to_numpy()
    out: dict[str, dict[str, int | None]] = {}
    for column in obs.columns:
        if column == "step":
            continue
        direction = OBSERVABLE_DIRECTION.get(column, "auto")
        t_top = transition_step(steps, obs[column].to_numpy(dtype=float), direction=direction)
        delta = (t_g - t_top) if (t_g is not None and t_top is not None) else None
        out[column] = {"t_top": t_top, "delta": delta}
    return out


def lead_lag(
    metrics: pd.DataFrame,
    observables: pd.DataFrame,
    observable: str = "h1_max_persistence",
    acc_threshold: float = 0.9,
) -> dict[str, int | float | None]:
    """Compare the grokking step to a topological observable's transition step."""
    t_g = grokking_step(metrics, acc_threshold)
    t_c = train_convergence_step(metrics)
    t_top = None
    if observable in observables:
        obs = observables.sort_values("step")
        t_top = transition_step(obs["step"].to_numpy(), obs[observable].to_numpy())
    delta = (t_g - t_top) if (t_g is not None and t_top is not None) else None
    return {
        "train_convergence_step": t_c,
        "grokking_step": t_g,
        "topological_transition_step": t_top,
        "observable": observable,
        "lead_lag_steps": delta,  # > 0: topology leads generalization
    }
