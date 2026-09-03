"""Is the trajectory-dimension negative about the quantity or the construction? (§6.4)"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import bootstrap_median_ci
from grokking_tda.artifacts.reader import Run
from grokking_tda.tda.phdim import ph_dimension_fit

WINDOWS = (100, 200, 400)
STRIDE_FACTORS = (1, 2, 3, 5)  # recorded every 20 steps, so effective strides 20-100
PROJECTIONS = (0, 64, 32)  # 0 keeps the stored 128 dimensions
STRIDE = 50
CALIBRATION_DIMS = (1, 2, 3, 4)


def calibrate(window: int, *, seed: int = 0, repeats: int = 5) -> list[dict]:
    """What the estimator returns on clouds of known dimension, at this window size."""
    rng = np.random.default_rng(seed)
    rows = []
    for dim in CALIBRATION_DIMS:
        for repeat in range(repeats):
            # a random walk, not an i.i.d. blob: the trajectory's own shape
            walk = np.cumsum(rng.normal(size=(window, dim)), axis=0)
            rows.append(
                {"window": window, "true_dim": dim, "repeat": repeat, **ph_dimension_fit(walk)}
            )
    return rows


def series(points: np.ndarray, steps: np.ndarray, window: int, *, seed: int = 0) -> pd.DataFrame:
    rows = []
    for end in range(window, len(steps) + 1, STRIDE):
        rows.append(
            {
                "step": int(steps[end - 1]),
                **ph_dimension_fit(points[end - window : end], seed=seed),
            }
        )
    return pd.DataFrame(rows)


def _median(values: list[dict], key: str) -> float:
    return float(np.median([v[key] for v in values])) if values else float("nan")


def stride_sweep(
    points: np.ndarray, steps: np.ndarray, window: int, *, seed: int = 0
) -> pd.DataFrame:
    """PH-dimension against the *iterate* stride, at a fixed number of points per window.

    Birdal et al. fit on consecutive iterates; these trajectories were recorded every twentieth
    optimiser step, and the stride is the one parameter §A.6 cannot sweep downward, because a
    finer sampling is not recoverable from a coarser recording. What is recoverable is the
    gradient: coarsen 20 to 40, 60 and 100 and see whether the estimate moves. Holding the window
    at a fixed number of *points* rather than steps is what makes a difference attributable to
    the sampling rate instead of to how much of training the window covers.
    """
    rows = []
    for factor in STRIDE_FACTORS:
        thinned, thinned_steps = points[::factor], steps[::factor]
        if len(thinned) < window:
            continue
        values = [
            ph_dimension_fit(thinned[end - window : end], seed=seed)
            for end in range(window, len(thinned) + 1, STRIDE)
        ]
        finite = [v for v in values if np.isfinite(v["ph_dim"])]
        rows.append(
            {
                "stride": factor * 20,
                "n_windows": len(values),
                "n_finite": len(finite),
                "ph_dim_terminal": values[-1]["ph_dim"] if values else float("nan"),
                "ph_dim_median": _median(finite, "ph_dim"),
                "slope_median": _median(finite, "slope"),
                "steps_spanned": int(thinned_steps[window - 1] - thinned_steps[0]),
            }
        )
    return pd.DataFrame(rows)


def project(points: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if dim <= 0 or dim >= points.shape[1]:
        return points
    rng = np.random.default_rng(seed)
    matrix = rng.normal(scale=1.0 / np.sqrt(dim), size=(points.shape[1], dim))
    return points @ matrix


def condition_of(name: str) -> str:
    return name.rsplit("_s", 1)[0]


DRAW_SWEEP = (1, 4, 8)


def draw_sweep(runs: list, window: int = 200, seed: int = 0) -> pd.DataFrame:
    """Terminal dimension with ``E_alpha`` averaged over subsamples, as Birdal et al. do."""
    rows = []
    for run_dir in runs:
        run = Run(run_dir)
        trajectory = run.trajectory()
        if trajectory is None:
            continue
        steps, points = trajectory
        for draws in DRAW_SWEEP:
            for end in range(len(steps), max(window, len(steps) - 5 * STRIDE), -STRIDE):
                fit = ph_dimension_fit(points[end - window : end], n_draws=draws, seed=seed)
                rows.append({"run": run.run_name, "condition": condition_of(run.run_name),
                             "n_draws": draws, "step": int(steps[end - 1]), **fit})
    return pd.DataFrame(rows)


def birdal_correlation(table: pd.DataFrame, bank: pd.DataFrame, window: int, dim: int) -> dict:
    """Terminal dimension against the generalisation gap; runs that never fit are dropped."""
    from scipy import stats

    sub = table[(table.window == window) & (table.projection_dim == dim)]
    terminal = (
        sub.sort_values("step").groupby("run").tail(5).groupby("run").ph_dim.median().dropna()
    )
    merged = bank.set_index("run").join(terminal.rename("ph_dim"), how="inner")
    merged = merged[merged.fits_train_set & merged.ph_dim.notna()]
    if len(merged) < 8:
        return {"n": int(len(merged))}

    gap, dimension = merged.generalisation_gap.to_numpy(), merged.ph_dim.to_numpy()
    rho = stats.spearmanr(gap, dimension)
    pearson = stats.pearsonr(gap, dimension)
    # Tan et al.'s comparator, which they report beating the dimension
    comparator: dict = {}
    if "weight_norm__final" in merged:
        # reported over the whole set and over the decayed subset alone, because the norm of a
        # vector that was never shrunk is not the quantity Tan et al. compare against
        columns = ["generalisation_gap", "weight_norm__final", "weight_decay"]
        norm = merged[[c for c in columns if c in merged]].dropna()
        for label, subset in (("all", norm), ("weight_decayed", norm[norm.weight_decay > 0])):
            if len(subset) < 8:
                continue
            r = stats.spearmanr(subset.generalisation_gap, subset.weight_norm__final)
            comparator[label] = {
                "n": int(len(subset)),
                "spearman_rho": float(r.statistic),
                "spearman_p": float(r.pvalue),
            }
    return {
        "weight_norm_vs_gap": comparator,
        "n": int(len(merged)),
        "window": window,
        "projection_dim": dim,
        "gap_range": [float(gap.min()), float(gap.max())],
        "gap_distinct_values": int(np.unique(np.round(gap, 2)).size),
        "ph_dim_range": [float(dimension.min()), float(dimension.max())],
        "spearman_rho": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "n_excluded_not_fitting": int((~bank.set_index("run").join(
            terminal.rename("ph_dim"), how="inner").fits_train_set).sum()),
    }


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--windows", type=int, nargs="*", default=list(WINDOWS))
    ap.add_argument("--projections", type=int, nargs="*", default=list(PROJECTIONS))
    ap.add_argument("--projection-seeds", type=int, default=2)
    ap.add_argument("--draw-sweep", action="store_true",
                    help="only sweep the number of subsamples per size, and stop")
    args = ap.parse_args()

    runs = sorted(p.parent for p in args.root.glob("*/trajectory.npz"))
    if not runs:
        raise SystemExit(f"no trajectory.npz under {args.root}")

    if args.draw_sweep:
        table = draw_sweep(runs)
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / "phdim_draws.csv", index=False)
        print(f"subsamples per size, terminal dimension over {table.run.nunique()} runs:\n")
        pivot = table.groupby(["condition", "n_draws"]).ph_dim.median().unstack()
        r2 = table.groupby("n_draws").r2.median()
        print(pivot.round(3).to_string())
        print("\nmedian fit r2 by draws: " + "  ".join(f"{k}: {v:.3f}" for k, v in r2.items()))
        print(f"\nwritten to {args.out}/phdim_draws.csv")
        return

    frames = []
    for run_dir in runs:
        run = Run(run_dir)
        trajectory = run.trajectory()
        if trajectory is None:
            continue
        steps, points = trajectory
        summary_path = run_dir / "analysis" / "summary.json"
        t_g = (
            json.loads(summary_path.read_text()).get("grokking_step")
            if summary_path.exists()
            else None
        )

        for window in args.windows:
            for dim in args.projections:
                seeds = range(args.projection_seeds) if dim > 0 else [0]
                for seed in seeds:
                    frame = series(project(points, dim, seed), steps, window, seed=seed)
                    if frame.empty:
                        continue
                    frame.insert(0, "projection_seed", seed)
                    frame.insert(0, "projection_dim", dim or points.shape[1])
                    frame.insert(0, "window", window)
                    frame.insert(0, "t_g", t_g)
                    frame.insert(0, "condition", condition_of(run.run_name))
                    frame.insert(0, "run", run.run_name)
                    frames.append(frame)
        print(f"  {run.run_name}", flush=True)

    table = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "phdim_sensitivity.csv", index=False)

    calibration = pd.DataFrame(
        [row for window in args.windows for row in calibrate(window)]
    )
    calibration.to_csv(args.out / "phdim_calibration.csv", index=False)

    print("\nestimator on random walks of known dimension (its floor at each window):")
    for (window, true_dim), sub in calibration.groupby(["window", "true_dim"]):
        print(
            f"  window {window:4d}  true dim {true_dim}  ->  "
            f"{sub.ph_dim.median():.2f}  (slope {sub.slope.median():.3f})"
        )

    print("\ntrajectory dimension, terminal value by condition:")
    summary: dict = {"windows": args.windows, "projections": args.projections, "conditions": {}}
    for (condition, window, dim), sub in table.groupby(["condition", "window", "projection_dim"]):
        terminal = sub.sort_values("step").groupby("run").tail(5)
        lo, med, hi = bootstrap_median_ci(terminal.ph_dim.dropna())
        entry = {
            "ph_dim": med,
            "ci": [lo, hi],
            "slope": float(terminal.slope.median()),
            "fraction_admissible": float(terminal.ph_dim.notna().mean()),
            "r2": float(terminal.r2.median()),
        }
        summary["conditions"].setdefault(condition, {})[f"w{window}_d{dim}"] = entry
        if window == 200 and dim == 128:
            print(
                f"  {condition:44s} dim {med:.2f} [{lo:.2f}, {hi:.2f}]  "
                f"slope {entry['slope']:.3f}  r2 {entry['r2']:.3f}"
            )

    bank_path = args.out / "bank.csv"
    if bank_path.exists():
        bank = pd.read_csv(bank_path)
        summary["birdal"] = {
            f"w{w}_d{d}": birdal_correlation(table, bank, w, d)
            for w in args.windows
            for d in (128,)
        }
        headline = summary["birdal"].get("w200_d128", {})
        if "spearman_rho" in headline:
            print(
                f"\nBirdal claim across runs (n={headline['n']}, gap "
                f"{headline['gap_range'][0]:.2f}-{headline['gap_range'][1]:.2f} over "
                f"{headline['gap_distinct_values']} distinct values): "
                f"rho={headline['spearman_rho']:+.3f} (p={headline['spearman_p']:.3g}), "
                f"pearson r={headline['pearson_r']:+.3f}"
            )

    (args.out / "phdim_sensitivity.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {args.out}/phdim_sensitivity.csv, .json and phdim_calibration.csv")


if __name__ == "__main__":
    main()
