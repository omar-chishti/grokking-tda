"""How much of the circularity association is mechanical, and does it need Fourier at all?

Section 4.5 reports $\\rho = 0.705$ between the normalised $H_1$ ratio and terminal Fourier
concentration, then concedes that the correlation is "partly mechanical" because both come
off the same embedding matrix. In an assessed document *partly* is doing a lot of work, and
the concession can be converted into numbers. Three of them.

**What the residual carries.** Regressing $\\log$ ratio on circularity and asking whether
what is left over says anything about $\\log \\tg$ is the sharpest available form of RQ3:
if the residual carries nothing, the topological observable is circularity plus noise; if
it carries something, that something is what persistent homology adds.

**A circularity measure that owes nothing to Fourier.** Everything in section 4.5 rests on
one statistic, and that statistic is the competitor the thesis is trying to beat. Two
geometric alternatives are added here, computed from positions in the top-two principal
plane rather than from any spectrum: the uniformity of the radius, which says the points
lie on a circle at all, and the resultant length of the angular gaps between consecutive
residues, which says they go round it *in order*. If an independently constructed measure
gives the same association, section 4.5 stops being a comparison of one statistic with its
own cousin.

**A null at the level of the observable.** The two existing nulls are *task* nulls: they
change the data and retrain. A third is nearly free and stronger for internal validity, but
not the one that suggests itself. Permuting the **rows** of a trained embedding cannot work:
persistent homology reads the distance matrix alone, a row permutation conjugates that
matrix by a permutation, and every diagram comes back bit-identical --- which is exactly the
invariance section 5.2 is built on. What does work is shuffling each **column**
independently, which preserves every coordinate's marginal distribution exactly and destroys
the joint arrangement, so a statistic that responds to the circle must collapse and one that
responds to the scale of a trained matrix will not.

Usage (from ``Code/``)::

    uv run python -m analysis.circularity
"""

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

N_PERMUTATIONS = 20  # row shuffles per run; the effect is large, so few are needed


def geometric_circularity(embedding: np.ndarray) -> dict[str, float]:
    """Two spectral-basis-free measures of how circular a residue embedding is."""
    x = np.asarray(embedding, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    plane = u[:, :2] * s[:2]

    radius = np.linalg.norm(plane, axis=1)
    mean_radius = float(radius.mean())
    if mean_radius <= 0:
        return {"circle_fit": 0.0, "cyclic_order": 0.0}

    # 1 - (coefficient of variation of the radius): 1 on a circle, 0 on a blob.
    circle_fit = float(max(0.0, 1.0 - radius.std() / mean_radius))

    # Do consecutive residues sit at a constant angular step? On a circle traversed in
    # residue order the increments are all 2*pi/p, so their circular resultant is 1;
    # scramble the arrangement and the increments are uniform and it falls to ~1/sqrt(p).
    # Rotation and reflection leave the increments alone, so nothing has to be searched.
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
    """H1 of a trained embedding, against the same matrix with each column shuffled.

    Shuffling within columns preserves each coordinate's marginal distribution exactly --- so
    the cloud keeps its extent, its per-dimension variances and its overall scale --- while
    destroying every correspondence between coordinates, and with it the circle. What
    survives is a null with the same univariate statistics and no geometry, which is the
    comparison that isolates arrangement from magnitude.
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


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--permutation-runs", type=int, default=20)
    args = ap.parse_args()

    bank, _ = load_bank(args.root)
    bank = bank[~bank.dense] if "dense" in bank else bank

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

    # The same population thesis section 4.5 quotes: composition in S_n is non-abelian, so
    # no basis exists in which spectral concentration could measure circularity there.
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
            f"\nresidual (section 2.2): circularity explains "
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
        print(f"\ncolumn-shuffle null (section 2.8), {len(nulls)} trained embeddings:")
        for _, r in nulls.iterrows():
            print(
                f"  {r.run:46s} H1 max {r.h1_max:.4f} -> {r.h1_max_null_median:.4f}"
                f"   cyclic order {r.cyclic_order:.3f} -> {r.cyclic_order_null_median:.3f}"
            )

    (args.out / "circularity_measures.json").write_text(
        json.dumps({"associations": associations, "residual": residual}, indent=2)
    )
    print(f"\nwritten to {args.out}/circularity_measures.csv and .json")


if __name__ == "__main__":
    main()
