"""How much of the circularity association is mechanical, and does it need Fourier? (§4.5)"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

from analysis import cli
from analysis.bank import iter_runs, load_bank
from grokking_tda.artifacts.reader import Run
from grokking_tda.config.schema import HomologyCfg, PointCloudCfg
from grokking_tda.tda.homology import compute_persistence
from grokking_tda.tda.pointcloud import build_point_cloud
from grokking_tda.tda.summaries import finite_lifetimes, max_persistence, total_persistence

N_PERMUTATIONS = 20  # column shuffles per run; the effect is large, so few are needed


def geometric_circularity(embedding: np.ndarray) -> dict[str, float]:
    x = np.asarray(embedding, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    plane = u[:, :2] * s[:2]

    radius = np.linalg.norm(plane, axis=1)
    mean_radius = float(radius.mean())
    if mean_radius <= 0:
        return {"circle_fit": 0.0, "cyclic_order": 0.0}

    # 1 - CV of the radius: 1 on a circle, 0 on a blob
    circle_fit = float(max(0.0, 1.0 - radius.std() / mean_radius))

    # Do consecutive residues sit at a constant angular step? In residue order the increments
    # are 2*pi/p and their resultant is 1; scrambled it falls to ~1/sqrt(p)
    angle = np.arctan2(plane[:, 1], plane[:, 0])
    increment = np.diff(np.concatenate([angle, angle[:1]]))
    cyclic_order = float(np.abs(np.exp(1j * increment).mean()))
    return {"circle_fit": circle_fit, "cyclic_order": cyclic_order}


def residual_analysis(bank: pd.DataFrame) -> dict:
    """Does what circularity fails to explain in the ratio say anything about ``t_g``?"""
    frame = bank.dropna(subset=["circularity", "h1_max_persistence_normalised__ratio", "t_g"])
    frame = frame[
        frame.grokked
        & (frame.operation != "compose")
        & (frame.h1_max_persistence_normalised__ratio > 0)
    ]
    if len(frame) < 12:
        return {}

    y = np.log10(frame["h1_max_persistence_normalised__ratio"].to_numpy(float))
    x = frame["circularity"].to_numpy(float)
    target = np.log10(frame["t_g"].to_numpy(float))

    fit = stats.linregress(x, y)
    residual = y - (fit.intercept + fit.slope * x)
    full = stats.spearmanr(y, target)
    left = stats.spearmanr(residual, target)
    return {
        "n": int(len(frame)),
        "ratio_on_circularity_r2": float(fit.rvalue**2),
        "ratio_vs_t_g": {"rho": float(full.statistic), "p": float(full.pvalue)},
        "residual_vs_t_g": {"rho": float(left.statistic), "p": float(left.pvalue)},
        "circularity_vs_t_g": {
            "rho": float(stats.spearmanr(x, target).statistic),
            "p": float(stats.spearmanr(x, target).pvalue),
        },
    }


def column_shuffle_null(run_dir: Path, *, seed: int = 0) -> dict | None:
    """H1 against the same matrix column-shuffled: marginals kept, correspondence gone.

    Columns and not rows. A row permutation conjugates the distance matrix, which persistent
    homology reads alone, so every diagram would come back bit-identical.
    """
    run = Run(run_dir)
    snapshots = run.snapshots()
    if not snapshots:
        return None
    embedding = snapshots[-1].representation("embedding")
    if embedding is None or not np.isfinite(embedding).all():
        return None

    cfg = PointCloudCfg(**run.config["analysis"]["pointcloud"])
    homology = HomologyCfg(maxdim=1)
    cloud = build_point_cloud(embedding, cfg, seed=int(run.config["seed"]))
    observed = compute_persistence(cloud, homology).get(1)

    rng = np.random.default_rng(seed)

    def shuffle_columns() -> np.ndarray:
        return np.take_along_axis(
            embedding, rng.permuted(np.tile(np.arange(embedding.shape[0])[:, None],
                                           embedding.shape[1]), axis=0), axis=0
        )

    rows, shuffled_geometry = [], []
    for _ in range(N_PERMUTATIONS):
        surrogate = shuffle_columns()
        diagram = compute_persistence(
            build_point_cloud(surrogate, cfg, seed=int(run.config["seed"])), homology
        ).get(1)
        rows.append((max_persistence(diagram), total_persistence(diagram)))
        shuffled_geometry.append(geometric_circularity(surrogate))

    permuted = np.asarray(rows, dtype=float)
    geometry = geometric_circularity(embedding)
    return {
        "run": run.run_name,
        "h1_max": max_persistence(observed),
        "h1_max_null_median": float(np.median(permuted[:, 0])),
        "h1_total": total_persistence(observed),
        "h1_total_null_median": float(np.median(permuted[:, 1])),
        "n_bars": int(finite_lifetimes(observed).size),
        "diameter": float(pdist(cloud).max()),
        **geometry,
        "cyclic_order_null_median": float(
            np.median([g["cyclic_order"] for g in shuffled_geometry])
        ),
        "circle_fit_null_median": float(
            np.median([g["circle_fit"] for g in shuffled_geometry])
        ),
    }


def recipe_variance_share(bank: pd.DataFrame, min_seeds: int = 3) -> dict:
    """How much of terminal circularity the recipe fixes and how much is left to the seed."""
    frame = bank.dropna(subset=["circularity"]).copy()
    frame["condition"] = frame.run.str.replace(r"_s\d+$", "", regex=True)
    frame = frame[frame.groupby("condition").circularity.transform("size") >= min_seeds]
    if frame.condition.nunique() < 2:
        return {}
    grand = frame.circularity.mean()
    groups = list(frame.groupby("condition").circularity)
    between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in groups)
    within = sum(((g - g.mean()) ** 2).sum() for _, g in groups)
    medians = frame.groupby("condition").circularity.median()
    spreads = frame.groupby("condition").circularity.agg(lambda s: s.max() - s.min())
    return {
        "n_runs": int(len(frame)),
        "n_conditions": int(frame.condition.nunique()),
        "between_share": float(between / (between + within)),
        "median_min": float(medians.min()),
        "median_max": float(medians.max()),
        "within_spread_median": float(spreads.median()),
        "within_spread_max": float(spreads.max()),
    }


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--permutation-runs", type=int, default=20)
    args = ap.parse_args()

    bank, _ = load_bank(args.root)
    bank = bank[~bank.replicate] if "replicate" in bank else bank

    geometry = []
    for run in iter_runs(args.root):
        embedding_path = args.root / run.name
        try:
            snapshots = Run(embedding_path).snapshots()
            embedding = snapshots[-1].representation("embedding") if snapshots else None
        except (FileNotFoundError, KeyError, IndexError):
            embedding = None
        if embedding is None or not np.isfinite(embedding).all():
            continue
        geometry.append({"run": run.name, **geometric_circularity(embedding)})
    geometry = pd.DataFrame(geometry)

    merged = bank.merge(geometry, on="run", how="inner")
    args.out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out / "circularity_measures.csv", index=False)

    # §4.5's population: S_n is non-abelian, so no basis measures circularity there
    grokked = merged[
        merged.grokked & merged.circularity.notna() & (merged.operation != "compose")
    ]
    ratio = "h1_max_persistence_normalised__ratio"
    associations = {}
    print(f"association with the normalised H1 max ratio, {len(grokked)} grokking runs:")
    for measure in ("circularity", "circle_fit", "cyclic_order"):
        sub = grokked.dropna(subset=[measure, ratio])
        rho = stats.spearmanr(sub[measure], sub[ratio])
        associations[measure] = {
            "n": int(len(sub)),
            "rho": float(rho.statistic),
            "p": float(rho.pvalue),
        }
        label = "Fourier (spectral)" if measure == "circularity" else f"{measure} (geometric)"
        print(f"  {label:26s} n={len(sub):3d}  rho={rho.statistic:+.3f}  p={rho.pvalue:.2g}")

    agreement = stats.spearmanr(grokked["circularity"], grokked["cyclic_order"])
    associations["fourier_vs_cyclic_order"] = {
        "rho": float(agreement.statistic),
        "p": float(agreement.pvalue),
    }
    print(f"\n  the two circularity measures agree at rho={agreement.statistic:+.3f}")

    residual = residual_analysis(merged)
    if residual:
        print(
            f"\nresidual (§2.2): circularity explains "
            f"{residual['ratio_on_circularity_r2']:.2f} of the log ratio's variance; "
            f"the ratio tracks log t_g at rho={residual['ratio_vs_t_g']['rho']:+.3f}, "
            f"what is left of it at rho={residual['residual_vs_t_g']['rho']:+.3f} "
            f"(p={residual['residual_vs_t_g']['p']:.2g})"
        )

    picked = (
        grokked.sort_values("circularity", ascending=False)
        .drop_duplicates("operation")
        .head(args.permutation_runs)
    )
    nulls = [r for r in (column_shuffle_null(args.root / n) for n in picked.run) if r]
    if nulls:
        nulls = pd.DataFrame(nulls)
        nulls.to_csv(args.out / "column_shuffle_null.csv", index=False)
        print(f"\ncolumn-shuffle null (§2.8), {len(nulls)} trained embeddings:")
        for _, r in nulls.iterrows():
            print(
                f"  {r.run:46s} H1 max {r.h1_max:.4f} -> {r.h1_max_null_median:.4f}"
                f"   cyclic order {r.cyclic_order:.3f} -> {r.cyclic_order_null_median:.3f}"
            )

    variance = recipe_variance_share(merged)
    print(
        f"\nrecipe against seed, over {variance['n_conditions']} conditions with three or more "
        f"seeds: the recipe accounts for {variance['between_share']:.2f} of the variance in "
        f"terminal circularity, condition medians running from {variance['median_min']:.3f} to "
        f"{variance['median_max']:.3f} against a within-condition spread with median "
        f"{variance['within_spread_median']:.3f}"
    )

    (args.out / "circularity_measures.json").write_text(
        json.dumps(
            {
                "associations": associations,
                "residual": residual,
                "recipe_variance": variance,
            },
            indent=2,
        )
    )
    print(f"\nwritten to {args.out}/circularity_measures.csv and .json")


if __name__ == "__main__":
    main()
