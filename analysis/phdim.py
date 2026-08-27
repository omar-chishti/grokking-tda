"""Is the trajectory-dimension negative about the quantity or about the construction? (6.4)

Section 6.4 reports that the persistent-homology dimension of the projected optimisation
path is flat in training time, identical in a run that memorises noise and one that learns a
rule, and unable to separate a generalisation gap of one from a gap of zero. It then names
three things that could each account for that and separates none of them. Two are settled
here from stored trajectories; the third can only be settled halfway, and this says which
half.

**Window length.** The reported series reads two hundred iterates at a stride of twenty
steps, so each window spans four thousand optimisation steps --- a substantial fraction of
the interval the transition occupies, and quite possibly an average over it. Swept.

**The projection.** Each iterate was written through a fixed random projection into $128$
dimensions *at training time*, so a second projection seed or a **larger** dimension needs
the runs again and is not a re-analysis, contrary to what the chapter implies. What the
stored trajectory does support is projecting further **down**, under fresh seeds. That is
still informative in the direction that matters: Johnson--Lindenstrauss says $128$
dimensions preserve pairwise distances, so if the estimate is stable from $128$ to $32$ the
projection is not what is destroying the signal, and if it moves then $128$ was already too
few and the negative belongs to the construction.

**Across runs, in Birdal's own form.** Their claim is stated between models --- terminal dimension
against the generalisation gap --- and the thesis could not state it that way, because every run
carrying a trajectory either groks completely or never generalises at all. `R13` gives a trajectory
to the conditions in between, so the correlation is computed here over a gap that is continuous.
The gap is ``train - test`` and not ``1 - test``: a run that never fits its training set has a
gap of about zero rather than of one, and is excluded, because the claim is about models that fit.

**The estimator's floor.** ``dim = alpha / (1 - slope)`` is stiff near the bottom of its
range: a dimension of $1.15$ is a slope of $0.13$, and the fitted slope is now reported
beside the dimension. More decisive is a calibration --- run the same estimator on synthetic
clouds of *known* dimension, at the same window size, and see what it returns for a true
line. If the measured trajectory dimensions sit at or below that value, the series is
resting on the estimator's floor and no experiment on this construction can lift it.

Usage (from ``Code/``)::

    uv run python -m analysis.phdim
    uv run python -m analysis.phdim --windows 100 200 400
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import bootstrap_median_ci
from grokking_tda.artifacts.reader import Run
from grokking_tda.tda.phdim import ph_dimension_fit

WINDOWS = (100, 200, 400)
PROJECTIONS = (0, 64, 32)  # 0 keeps the stored 128 dimensions
STRIDE = 50
CALIBRATION_DIMS = (1, 2, 3, 4)


def calibrate(window: int, *, seed: int = 0, repeats: int = 5) -> list[dict]:
    """What the estimator returns on clouds whose dimension is known, at this window size."""
    rng = np.random.default_rng(seed)
    rows = []
    for dim in CALIBRATION_DIMS:
        for repeat in range(repeats):
            # A line, a plane, ... sampled as a random walk: the trajectory's own shape,
            # not an i.i.d. blob, so the comparison is against a like object.
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


def project(points: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """A further Johnson-Lindenstrauss projection of the already-projected path."""
    if dim <= 0 or dim >= points.shape[1]:
        return points
    rng = np.random.default_rng(seed)
    matrix = rng.normal(scale=1.0 / np.sqrt(dim), size=(points.shape[1], dim))
    return points @ matrix


def condition_of(name: str) -> str:
    return name.rsplit("_s", 1)[0]


def birdal_correlation(table: pd.DataFrame, bank: pd.DataFrame, window: int, dim: int) -> dict:
    """Terminal PH-dimension against the generalisation gap, across runs.

    Terminal dimension is the median over each run's last five windows; runs that never fit
    their training set are dropped. Birdal et al. report a *positive* association --- a
    higher-dimensional trajectory going with a larger gap.
    """
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
    return {
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
    args = ap.parse_args()

    runs = sorted(p.parent for p in args.root.glob("*/trajectory.npz"))
    if not runs:
        raise SystemExit(f"no trajectory.npz under {args.root}")

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
