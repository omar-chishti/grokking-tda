"""Operational definitions of the transition, fixed in advance (no post-hoc tuning).

  - grokking step ``t_g``: first step with test accuracy >= threshold (default 0.9)
  - train-convergence ``t_c``: first step with train accuracy >= threshold (0.99)
  - topological transition ``t_top``: midpoint-crossing of a rising observable,
    measured from its trough, and undefined where the observable never rises
  - lead/lag ``delta = t_g - t_top``  (delta > 0 means topology leads generalization)
"""

from __future__ import annotations

from typing import Any

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
    """Midpoint-crossing step of a rising observable, measured from its trough.

    A simple, assumption-light change-point proxy: the step at which the observable
    is halfway from its lowest value to its subsequent peak. Returns ``None`` where
    the observable never rises — a series that only decays has no transition to
    report, and inventing one for it corrupts the lead-lag comparison.
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
    # The midpoint proxy crosses *something* whenever max > min, so a series that only
    # decays still yields a step — and one near zero, which reads as a large topological
    # lead. Measure the rise from the trough to the highest value that follows it: the
    # global maximum is often the random-initialisation transient, which is not a
    # transition, and a series with nothing above its trough has none at all.
    trough = int(np.nanargmin(values))
    peak = trough + int(np.nanargmax(values[trough:]))
    if peak == trough:
        return None
    lo, hi = values[trough], values[peak]
    return _first_crossing(steps[trough:], values[trough:], lo + 0.5 * (hi - lo))


def grokking_step_sensitivity(
    metrics: pd.DataFrame, observables: pd.DataFrame | None = None
) -> dict[str, int | None]:
    """``t_g`` under every definition the thesis reports, so the choice is auditable.

    Thresholds are read from ``metrics``, which is logged far more finely than the
    snapshot grid; the leak-free series exists only as an observable, so the two frames
    are used for what each can answer. The midpoint-of-rise alternative is biased early
    on a commutative task — raw test accuracy rests on a plateau of roughly the train
    fraction, so the midpoint is taken between that plateau and one — and the leak-free
    midpoint is reported beside it to show the size of that bias.
    """
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
    """Transition step and signed lag for *every* observable column.

    The lead-lag forest plot needs ``t_top`` per observable, not only the headline
    one. Each observable declares which way it moves at registration; only columns
    with no declaration fall back to inferring it from the series.
    """
    from grokking_tda.analysis.observable import OBSERVABLE_DIRECTION
    from grokking_tda.evaluation.changepoint import changepoint_step

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
        # A second, differently-principled detector: agreement is a robustness result and
        # disagreement says the transition is not sharply located. Both are reported.
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
    """Compare the grokking step to a topological observable's transition step."""
    t_g = grokking_step(metrics, acc_threshold)
    t_c = train_convergence_step(metrics)
    t_top = None
    if observable in observables:
        from grokking_tda.analysis.observable import OBSERVABLE_DIRECTION

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
        "lead_lag_steps": delta,  # > 0: topology leads generalization
    }
