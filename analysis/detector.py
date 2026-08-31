"""How much does the choice of detector decide the timing result? (§3.6, §5.6)"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import HEADLINE_OBSERVABLE, condition_label, iter_runs, load_bank
from grokking_tda.evaluation import transition_step

# as a multiple of the baseline: 1.0 is no step, 2.9 is what the reference regime shows
AMPLITUDES = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 9.0)
LOCATIONS = (0.2, 0.35, 0.5, 0.65, 0.8)  # as a fraction of the run's length
N_DRAWS = 40
WIDTH = 0.04  # logistic width, as a fraction of the run: a transition, not a jump


def inject(steps: np.ndarray, values: np.ndarray, location: float, amplitude: float):
    """A logistic step of known location and amplitude, multiplied in: the observable is a ratio."""
    span = steps[-1] - steps[0]
    centre = steps[0] + location * span
    ramp = 1.0 / (1.0 + np.exp(-(steps - centre) / (WIDTH * span)))
    return values * (1.0 + (amplitude - 1.0) * ramp), centre


def null_series(root: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Permuted-label runs: this pipeline's own noise, carrying none of its signal."""
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


def unrepaired(root: Path) -> dict:
    """What the proxy did before it measured the rise from the trough."""
    n_runs = n_located = n_zero = n_grokking = n_lead = 0
    for run in iter_runs(root):
        obs = run.observables
        if "h1_max_persistence" not in obs:
            continue
        n_runs += 1
        found = transition_step(
            obs["step"].to_numpy(float),
            obs["h1_max_persistence"].to_numpy(float),
            direction="rising",
            compare="global",
        )
        if found is None:
            continue
        n_located += 1
        n_zero += found == 0
        t_g = run.summary.get("grokking_step")
        if t_g is not None:
            n_grokking += 1
            n_lead += (t_g - found) == t_g
    return {
        "n_runs": n_runs,
        "n_located": n_located,
        "n_at_step_zero": n_zero,
        "n_grokking": n_grokking,
        "n_lead_is_tg": n_lead,
    }


def changepoint_agreement(bank: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The timing result under the second detector, from the same ``t_g`` (§3.6, §5.6)."""
    runs = bank[bank.grokked & ~bank.replicate].copy()
    runs["label"] = [condition_label(r) for _, r in runs.iterrows()]
    both = runs.dropna(subset=["t_top", "t_changepoint"])

    rows = []
    for label, sub in runs.groupby("label"):
        proxy = sub.lead_lag_steps.dropna()
        cp = sub.lead_lag_steps__changepoint.dropna()
        rows.append(
            {
                "condition": label,
                "n_seeds": len(sub),
                "n_located_proxy": len(proxy),
                "n_located_changepoint": len(cp),
                "median_lag_proxy": float(proxy.median()) if len(proxy) else float("nan"),
                "median_lag_changepoint": float(cp.median()) if len(cp) else float("nan"),
                "n_lagging_proxy": int((proxy < 0).sum()),
                "n_lagging_changepoint": int((cp < 0).sum()),
            }
        )
    table = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)

    separation = (both.t_top - both.t_changepoint).abs()
    signs = np.sign(both.lead_lag_steps) == np.sign(both.lead_lag_steps__changepoint)
    summary = {
        "observable": HEADLINE_OBSERVABLE,
        "n_grokking_runs": int(len(runs)),
        "n_located_proxy": int(runs.t_top.notna().sum()),
        "n_located_changepoint": int(runs.t_changepoint.notna().sum()),
        "n_located_by_both": int(len(both)),
        "median_separation_steps": float(separation.median()) if len(both) else float("nan"),
        "median_separation_over_t_g": float((separation / both.t_g).median())
        if len(both)
        else float("nan"),
        "sign_agreement": float(signs.mean()) if len(both) else float("nan"),
        "n_lagging_proxy": int((runs.lead_lag_steps < 0).sum()),
        "n_lagging_changepoint": int((runs.lead_lag_steps__changepoint < 0).sum()),
    }
    return table, summary


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

    bank, _ = load_bank(args.root)
    agreement, summary["changepoint"] = changepoint_agreement(bank)
    agreement.to_csv(args.out / "detector_agreement.csv", index=False)
    c = summary["changepoint"]
    print(
        f"\n{c['n_grokking_runs']} grokking runs: the proxy locates a transition on "
        f"{c['n_located_proxy']}, the changepoint on {c['n_located_changepoint']}, both on "
        f"{c['n_located_by_both']}"
    )
    print(
        f"  they place it a median {c['median_separation_steps']:,.0f} steps apart "
        f"({c['median_separation_over_t_g']:.0%} of t_g) and agree on the sign of the lag "
        f"on {c['sign_agreement']:.0%} of runs"
    )
    print(
        f"  topology lags on {c['n_lagging_proxy']} runs under the proxy and "
        f"{c['n_lagging_changepoint']} under the changepoint"
    )

    summary["unrepaired_detector"] = unrepaired(args.root)
    u = summary["unrepaired_detector"]
    print(
        f"\nunrepaired detector on the raw series: located a transition on {u['n_located']} of "
        f"{u['n_runs']} runs, {u['n_at_step_zero']} of them at step 0; on the "
        f"{u['n_grokking']} that grok it reported a lead of exactly t_g on {u['n_lead_is_tg']}"
    )

    (args.out / "detector_calibration.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {args.out}/detector_calibration.csv, .json and detector_agreement.csv")


if __name__ == "__main__":
    main()
