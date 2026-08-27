"""How well does the transition detector recover a step it is shown? (section 5.6)

The lag of section 5.6 is a difference between two detectors with different noise
properties --- $\\tg$ is a threshold crossing on test accuracy, a series that goes from
chance to one, while $\\ttop$ is a midpoint crossing on a noisy persistence ratio. The
difference of two estimators is biased whenever either is, and the size of that bias is
currently unknown, so the seed spread quoted in D5.5 is standing in for an uncertainty
rather than being one.

It can be measured directly. Take a real null series --- a permuted-label run's own
persistence, which carries this pipeline's actual noise and none of its signal --- inject a
logistic step of known location and known amplitude, and ask the detector where it thinks
the step is. Sweep the amplitude against the series' own noise, because the answer is only
interesting relative to what the reference regime actually shows: a normalised ratio rising
about threefold across the transition.

The snapshot grid is logarithmic, so the recovery error is reported in steps *and* as a
fraction of the injected location, and the smallest amplitude at which the detector is
usable at all is reported with it.

Usage (from ``Code/``)::

    uv run python -m analysis.detector
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import HEADLINE_OBSERVABLE, iter_runs
from grokking_tda.evaluation import transition_step

# Amplitudes as a multiple of the baseline: 1.0 is no step, 2.9 is what the reference
# regime shows, and the range brackets every condition in the bank.
AMPLITUDES = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 9.0)
LOCATIONS = (0.2, 0.35, 0.5, 0.65, 0.8)  # as a fraction of the run's length
N_DRAWS = 40
WIDTH = 0.04  # logistic width, as a fraction of the run: a transition, not a jump


def inject(steps: np.ndarray, values: np.ndarray, location: float, amplitude: float):
    """A logistic step of known location and amplitude, multiplied into a null series.

    Multiplicative rather than additive because the observable is a ratio and the thesis
    reads it as one; injecting into the log is what makes "an amplitude of 2.9" mean the
    same thing here as in table 4.2.
    """
    span = steps[-1] - steps[0]
    centre = steps[0] + location * span
    ramp = 1.0 / (1.0 + np.exp(-(steps - centre) / (WIDTH * span)))
    return values * (1.0 + (amplitude - 1.0) * ramp), centre


def null_series(root: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Permuted-label runs' headline observable: this pipeline's noise, without its signal."""
    out = []
    for run in iter_runs(root):
        if not run.config.get("data", {}).get("label_permutation"):
            continue
        obs = run.observables
        if HEADLINE_OBSERVABLE not in obs:
            continue
        values = obs[HEADLINE_OBSERVABLE].to_numpy(float)
        steps = obs["step"].to_numpy(float)
        keep = np.isfinite(values) & (values > 0)
        if keep.sum() > 40:
            out.append((run.name, steps[keep], values[keep]))
    return out


def main() -> None:
    ap = cli.parser(__doc__)
    args = ap.parse_args()

    series = null_series(args.root)
    if not series:
        raise SystemExit("no permuted-label runs with the headline observable")

    rng = np.random.default_rng(0)
    rows = []
    for amplitude in AMPLITUDES:
        for location in LOCATIONS:
            for draw in range(N_DRAWS):
                name, steps, values = series[rng.integers(len(series))]
                injected, centre = inject(steps, values, location, amplitude)
                found = transition_step(steps, injected, direction="rising")
                rows.append(
                    {
                        "amplitude": amplitude,
                        "location": location,
                        "draw": draw,
                        "run": name,
                        "true_step": centre,
                        "found_step": found,
                        "error": (found - centre) if found is not None else np.nan,
                        "relative_error": (found - centre) / centre if found else np.nan,
                    }
                )

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "detector_calibration.csv", index=False)

    summary: dict = {
        "amplitudes": list(AMPLITUDES),
        "locations": list(LOCATIONS),
        "by_amplitude": {},
    }
    print("recovering an injected step, over permuted-label series:")
    print("  amplitude   located   median bias    IQR of bias    median |relative|")
    for amplitude, sub in table.groupby("amplitude"):
        found = sub.dropna(subset=["error"])
        located = len(found) / len(sub)
        if found.empty:
            print(f"  {amplitude:6.2f}x     {located:5.0%}          —")
            continue
        bias = found.error.median()
        iqr = found.error.quantile(0.75) - found.error.quantile(0.25)
        relative = found.relative_error.abs().median()
        summary["by_amplitude"][f"{amplitude:g}"] = {
            "fraction_located": located,
            "median_bias_steps": float(bias),
            "iqr_steps": float(iqr),
            "median_absolute_relative_error": float(relative),
        }
        print(
            f"  {amplitude:6.2f}x     {located:5.0%}     {bias:+10.0f}    {iqr:10.0f}"
            f"        {relative:8.1%}"
        )

    (args.out / "detector_calibration.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {args.out}/detector_calibration.csv and .json")


if __name__ == "__main__":
    main()
