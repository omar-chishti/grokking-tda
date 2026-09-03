"""Is the trajectory-dimension negative about the quantity or the construction? (§6.4)"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import bootstrap_median_ci, load_bank
from grokking_tda.artifacts.reader import Run
from grokking_tda.tda.phdim import ph_dimension_fit

WINDOWS = (100, 200, 400)
STRIDES = (1, 5, 20, 40, 60, 100)  # in optimiser steps, so a run can only report its own and up
PROJECTIONS = (0, 64, 32)  # 0 keeps the stored 128 dimensions
STRIDE = 50
CALIBRATION_DIMS = (1, 2, 3, 4)
# alpha-stable Levy walks, whose image has Hausdorff dimension alpha: Simsekli et al.'s own
# model class, and the ladder the Gaussian one cannot supply, since a Brownian path has
# dimension min(2, k) in every ambient dimension and three of its four rungs coincide.
ALPHAS = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5, 2.0)
N_ITERATES = 200  # Birdal et al.'s protocol: the final consecutive iterates, and only those


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


def calibrate_alpha(
    window: int,
    *,
    stride: int = 20,
    dim: int = 128,
    repeats: int = 10,
    seed: int = 0,
) -> list[dict]:
    """What the estimator returns on walks whose image dimension is known to be ``alpha``.

    Drawn directly in the projected dimensions rather than drawn in parameter space and then
    projected: a linear combination of alpha-stable variates is alpha-stable with the same index,
    so this *is* the projected walk and not an approximation of one. Sampled at the stride the
    real trajectories are read at, so what is calibrated is the measurement rather than the
    estimator in isolation.
    """
    from scipy.stats import levy_stable

    rows = []
    for alpha in ALPHAS:
        for repeat in range(repeats):
            increments = levy_stable.rvs(
                alpha, 0.0, size=(window * stride, dim), random_state=seed * 1000 + repeat
            )
            walk = np.cumsum(increments, axis=0)[::stride][:window]
            rows.append(
                {
                    "window": window,
                    "stride": stride,
                    "alpha": alpha,
                    "repeat": repeat,
                    **ph_dimension_fit(walk, seed=repeat),
                }
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
    points: np.ndarray, steps: np.ndarray, window: int, *, every: int = 20, seed: int = 0
) -> pd.DataFrame:
    """PH-dimension against the *iterate* stride, at a fixed number of points per window.

    Birdal et al. fit on consecutive iterates. A recording made every ``every`` steps can be
    thinned but not refined, so a run reports the strides at or above its own recording rate and
    R19 exists to supply the rest. Holding the window at a fixed number of *points* rather than
    steps is what makes a difference attributable to the sampling rate instead of to how much of
    training the window covers.
    """
    rows = []
    for stride in STRIDES:
        if stride % every:
            continue
        thinned, thinned_steps = points[:: stride // every], steps[:: stride // every]
        if len(thinned) < window:
            continue
        values = [
            ph_dimension_fit(thinned[end - window : end], seed=seed)
            for end in range(window, len(thinned) + 1, STRIDE)
        ]
        finite = [v for v in values if np.isfinite(v["ph_dim"])]
        rows.append(
            {
                "stride": stride,
                "n_windows": len(values),
                "n_finite": len(finite),
                "ph_dim_terminal": values[-1]["ph_dim"] if values else float("nan"),
                "ph_dim_median": _median(finite, "ph_dim"),
                "slope_median": _median(finite, "slope"),
                "steps_spanned": int(thinned_steps[window - 1] - thinned_steps[0]),
            }
        )
    return pd.DataFrame(rows)


def birdal_protocol(
    points: np.ndarray, steps: np.ndarray, *, n_iterates: int = N_ITERATES, seed: int = 0
) -> dict:
    """The published protocol: train to convergence, then fit on the final *consecutive* iterates.

    Only meaningful on a stride-one recording. On a coarser one the same 200 points span twenty
    times the training, which is the objection this answers rather than a coarser version of it.
    """
    window = points[-n_iterates:]
    return {
        "n_iterates": len(window),
        "steps_spanned": int(steps[-1] - steps[-len(window)]),
        **ph_dimension_fit(window, seed=seed),
    }


def project(points: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if dim <= 0 or dim >= points.shape[1]:
        return points
    rng = np.random.default_rng(seed)
    matrix = rng.normal(scale=1.0 / np.sqrt(dim), size=(points.shape[1], dim))
    return points @ matrix


def recording_every(run: Run) -> int:
    return int(run.config["train"].get("trajectory_every", 20))


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


def _correlate(terminal: pd.Series, bank: pd.DataFrame) -> dict:
    """Terminal dimension against the generalisation gap; runs that never fit are dropped."""
    from scipy import stats

    joined = bank.set_index("run").join(terminal.rename("ph_dim"), how="inner")
    merged = joined[joined.fits_train_set & joined.ph_dim.notna()]
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
        "gap_range": [float(gap.min()), float(gap.max())],
        "gap_distinct_values": int(np.unique(np.round(gap, 2)).size),
        "ph_dim_range": [float(dimension.min()), float(dimension.max())],
        "spearman_rho": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "n_excluded_not_fitting": int((~joined.fits_train_set).sum()),
    }


def birdal_correlation(table: pd.DataFrame, bank: pd.DataFrame, window: int, dim: int) -> dict:
    sub = table[(table.window == window) & (table.projection_dim == dim)]
    terminal = (
        sub.sort_values("step").groupby("run").tail(5).groupby("run").ph_dim.median().dropna()
    )
    return {"window": window, "projection_dim": dim} | _correlate(terminal, bank)


def stride_main(runs: list, args) -> None:
    """The sampling-rate axis §A.6 could not sweep downward until R19 recorded at stride one."""
    window = args.stride_window
    frames, protocol = [], []
    for run_dir in runs:
        run = Run(run_dir)
        trajectory = run.trajectory()
        if trajectory is None:
            continue
        steps, points = trajectory
        every = recording_every(run)
        frame = stride_sweep(points, steps, window, every=every)
        if frame.empty:
            continue
        frame.insert(0, "recorded_every", every)
        frame.insert(0, "condition", condition_of(run.run_name))
        frame.insert(0, "run", run.run_name)
        frames.append(frame)
        if every == 1:
            protocol.append(
                {"run": run.run_name, "condition": condition_of(run.run_name)}
                | birdal_protocol(points, steps)
            )
        print(f"  {run.run_name}  every {every}", flush=True)

    table = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "phdim_stride.csv", index=False)

    summary: dict = {"window": window, "strides": {}, "protocol": {}}
    # a run set outside the thesis tree has no committed bank, so summarise it here rather
    # than write one into `results/processed/thesis/` and move the run count the chapters quote
    bank_path = args.bank or args.out / "bank.csv"
    bank = pd.read_csv(bank_path) if bank_path.exists() else load_bank(args.root)[0]
    if bank is not None:
        for stride, sub in table.groupby("stride"):
            terminal = sub.set_index("run").ph_dim_terminal.dropna()
            summary["strides"][f"s{stride}"] = {"n_runs": int(len(terminal))} | _correlate(
                terminal, bank
            )
        if protocol:
            fits = pd.DataFrame(protocol)
            summary["protocol"] = {
                "n_iterates": N_ITERATES,
                "n_runs": int(len(fits)),
                "ph_dim_median": float(fits.ph_dim.median()),
                "steps_spanned": int(fits.steps_spanned.median()),
            } | _correlate(fits.set_index("run").ph_dim.dropna(), bank)
            fits.to_csv(args.out / "phdim_protocol.csv", index=False)

    print("\nmedian terminal dimension by stride, and its correlation with the gap:")
    for stride, sub in table.groupby("stride"):
        block = summary["strides"].get(f"s{stride}", {})
        rho = block.get("spearman_rho")
        print(
            f"  stride {stride:4d}  n {len(sub):3d}  dim {sub.ph_dim_terminal.median():.3f}  "
            f"span {int(sub.steps_spanned.median()):6d} steps  "
            + (f"rho {rho:+.3f} (p={block['spearman_p']:.3g}, n={block['n']})" if rho else "")
        )
    if summary["protocol"]:
        block = summary["protocol"]
        print(
            f"\nBirdal et al.'s protocol, the last {N_ITERATES} consecutive iterates on "
            f"{block['n_runs']} runs: dim {block['ph_dim_median']:.3f}, "
            f"rho {block.get('spearman_rho', float('nan')):+.3f} "
            f"(p={block.get('spearman_p', float('nan')):.3g}, n={block.get('n')})"
        )
    (args.out / "phdim_stride.json").write_text(json.dumps(summary, indent=2))
    written = "phdim_stride.csv, .json" + (" and phdim_protocol.csv" if protocol else "")
    print(f"\nwritten to {args.out}/{written}")


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--windows", type=int, nargs="*", default=list(WINDOWS))
    ap.add_argument("--projections", type=int, nargs="*", default=list(PROJECTIONS))
    ap.add_argument("--projection-seeds", type=int, default=2)
    ap.add_argument("--draw-sweep", action="store_true",
                    help="only sweep the number of subsamples per size, and stop")
    ap.add_argument("--stride-sweep", action="store_true",
                    help="only sweep the sampling rate, with Birdal et al.'s protocol, and stop")
    ap.add_argument("--stride-window", type=int, default=200,
                    help="points per window in the stride sweep; the window the chapters quote")
    ap.add_argument("--bank", type=Path, default=None,
                    help="the bank to correlate against; defaults to one beside --out")
    ap.add_argument("--alpha-calibration", action="store_true",
                    help="only calibrate on alpha-stable walks of known image dimension, and stop")
    args = ap.parse_args()

    if args.alpha_calibration:
        table = pd.DataFrame(
            [row for window in args.windows for row in calibrate_alpha(window)]
        )
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / "phdim_alpha_calibration.csv", index=False)
        print("estimator on alpha-stable walks, whose image dimension is alpha:\n")
        summary = table.groupby(["window", "alpha"]).ph_dim.agg(["median", "std"])
        print(summary.round(3).to_string())
        print(f"\nwritten to {args.out}/phdim_alpha_calibration.csv")
        return

    runs = sorted(p.parent for p in args.root.glob("*/trajectory.npz"))
    if not runs:
        raise SystemExit(f"no trajectory.npz under {args.root}")

    if args.stride_sweep:
        stride_main(runs, args)
        return

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
