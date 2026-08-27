"""Sensitivity analysis for the scale normalisation of thesis section 3.3.1.

The normalisation is the thesis's claimed methodological contribution and it rests on
one scalar: every H1 summary is divided by the largest finite H0 death, justified in
``tda/observables.py`` as "exactly linear in the cloud's size". That is true of a
*similarity* transform, which rescales every pairwise distance by one factor. It is not
true of a contraction that flattens the cloud, and if the cloud flattens then loops in
the collapsing directions shrink faster than the connectivity scale does and the
normalised ratio is biased. Nothing in the thesis currently tests this.

Two questions, both answered from cached embeddings without recomputing a diagram:

**Is the contraction a similarity?** Measured by the singular-value spectrum of the
centred embedding over training. A similarity leaves the spectrum's *shape* alone, so a
constant effective rank means isotropic contraction and a falling one means the cloud is
collapsing onto a subspace --- in which case the normaliser's justification does not hold
and what it removes has to be re-argued.

**Does the verdict pattern survive a different normaliser?** The raw H1 summaries are
stored, so dividing them by any other length scale of the same cloud is arithmetic.
Three alternatives are swept: mean pairwise distance, diameter, and sqrt(tr Sigma), the
root-mean-square radius. If the conditions that clear their null band are the same set
under all four, the contribution does not depend on the choice.

Usage (from ``Code/``)::

    uv run python -m analysis.normalisation
    uv run python -m analysis.normalisation --runs transformer_add113_f0.3_wd0.1_softmax_ce_s0
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from analysis import cli
from grokking_tda.artifacts.reader import Run
from grokking_tda.config.schema import PointCloudCfg
from grokking_tda.tda.pointcloud import build_point_cloud

# Singular values below this fraction of the largest carry no structure, only round-off.
RANK_FLOOR = 1e-10

# The alternative length scales, each a different answer to "how big is this cloud".
# ``connectivity`` is the thesis normaliser and comes from the stored diagram, not here.
ALT_SCALES = ("mean_pairwise", "diameter", "rms_radius")

SUMMARIES = ("h1_max_persistence", "h1_total_persistence")


def cloud_geometry(cloud: np.ndarray) -> dict:
    """Length scales and anisotropy summaries of one point cloud."""
    # A diverged run's weights are NaN. ``compute_persistence`` already treats that as a
    # result to record rather than a crash; LAPACK instead fails to converge, which would
    # take down the whole sweep.
    if not np.isfinite(cloud).all():
        return {}
    d = pdist(cloud)
    s = np.linalg.svd(cloud - cloud.mean(axis=0, keepdims=True), compute_uv=False)
    s = s[s > RANK_FLOOR * s[0]] if s.size and s[0] > 0 else s
    if s.size == 0 or d.size == 0:
        return {}

    var = s**2
    p = var / var.sum()
    return {
        "mean_pairwise": float(d.mean()),
        "diameter": float(d.max()),
        "rms_radius": float(np.sqrt(var.sum() / cloud.shape[0])),
        # sigma_1 / sigma_10 rather than sigma_1 / sigma_min: the centred embedding is
        # rank-deficient, so the smallest retained singular value sits against the
        # numerical floor and a true condition number is noise.
        "spectral_decay": float(s[0] / s[min(9, s.size - 1)]),
        # Participation ratio and exponentiated spectral entropy: two standard, and
        # differently weighted, answers to "how many directions carry this cloud".
        "participation_ratio": float(var.sum() ** 2 / (var**2).sum()),
        "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
        "top2_variance_fraction": float(p[:2].sum()),
    }


def run_series(run_dir: Path) -> pd.DataFrame:
    """Per-snapshot geometry, joined to the stored persistence summaries."""
    run = Run(run_dir)
    cfg = PointCloudCfg(**run.config["analysis"]["pointcloud"])
    obs = pd.read_csv(run_dir / "analysis" / "observables.csv").set_index("step")

    rows = []
    for snap in run.snapshots():
        try:
            emb = snap.representation("embedding")
        except (FileNotFoundError, KeyError):
            continue
        if emb is None or snap.step not in obs.index:
            continue
        geom = cloud_geometry(build_point_cloud(emb, cfg, seed=int(run.config["seed"])))
        if not geom:
            continue
        row = {"run": run.run_name, "step": snap.step, **geom}
        row["connectivity"] = float(obs.at[snap.step, "pointcloud_scale"])
        for summary in SUMMARIES:
            raw = float(obs.at[snap.step, summary])
            row[f"{summary}__raw"] = raw
            for scale in ("connectivity", *ALT_SCALES):
                row[f"{summary}__{scale}"] = raw / row[scale] if row[scale] > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def window_ratios(df: pd.DataFrame, t_g: float | None, columns) -> dict:
    """Plateau-over-baseline for each column, under the thesis window rule."""
    from analysis.bank import (
        BASELINE_WINDOW,
        NULL_BASELINE_WINDOW,
        NULL_PLATEAU_FROM,
        PLATEAU_FROM,
    )

    steps, end = df["step"].to_numpy(float), float(df["step"].max())
    if t_g:
        lo, hi, plat = BASELINE_WINDOW[0] * t_g, BASELINE_WINDOW[1] * t_g, PLATEAU_FROM * t_g
    else:
        lo, hi = NULL_BASELINE_WINDOW[0] * end, NULL_BASELINE_WINDOW[1] * end
        plat = NULL_PLATEAU_FROM * end

    base_rows, plat_rows = (steps >= lo) & (steps <= hi), steps >= plat
    out = {}
    for col in columns:
        b, a = df.loc[base_rows, col], df.loc[plat_rows, col]
        if b.empty or a.empty:
            continue
        out[f"{col}__baseline"] = float(b.median())
        out[f"{col}__plateau"] = float(a.median())
        out[f"{col}__ratio"] = float(a.median() / b.median()) if b.median() else np.nan
    return out


def verdict_invariance(root: Path, summaries: pd.DataFrame) -> pd.DataFrame:
    """Condition-level above/inside/below verdicts under each candidate normaliser.

    The question thesis section 4.4 turns on is not the value of any one ratio but which
    conditions clear their null band. Rebuilding that comparison with a different
    denominator, against a null band recomputed under the same denominator, says whether
    the answer belongs to the data or to the choice of scale.
    """
    from analysis.bank import CONDITION_KEYS, condition_table, load_bank, verdicts

    bank, _ = load_bank(root)
    ratios = summaries.set_index("run")
    columns = [f"{s}__{n}" for s in SUMMARIES for n in ("connectivity", *ALT_SCALES)]
    for column in columns:
        source = f"{column}__ratio"
        bank[source] = bank["run"].map(ratios[source]) if source in ratios else np.nan

    merged = bank.dropna(subset=[f"{c}__ratio" for c in columns], how="all")
    table = verdicts(
        merged,
        condition_table(merged, observables=columns),
        observables=columns,
    )
    return table[[*CONDITION_KEYS, "n_runs", "n_grokked", *[f"{c}__verdict" for c in columns]]]


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--runs", nargs="*", help="run names; default is every run with embeddings")
    args = ap.parse_args()

    names = args.runs or [
        d.name
        for d in sorted(args.root.iterdir())
        if (d / "snapshots" / "index.json").exists()
        and (d / "analysis" / "observables.csv").exists()
    ]

    tracked = ["spectral_decay", "participation_ratio", "effective_rank", "rms_radius"]
    tracked += [f"{s}__{n}" for s in SUMMARIES for n in ("connectivity", *ALT_SCALES)]

    series, summaries = [], []
    for name in names:
        run_dir = args.root / name
        df = run_series(run_dir)
        if df.empty:
            continue
        series.append(df)

        summary_path = run_dir / "analysis" / "summary.json"
        t_g = (
            json.loads(summary_path.read_text()).get("grokking_step")
            if summary_path.exists()
            else None
        )
        summaries.append({"run": name, "t_g": t_g, **window_ratios(df, t_g, tracked)})
        s = summaries[-1]
        print(
            f"  {name:52s} t_g={t_g!s:>8}"
            f"  eff.rank {s.get('effective_rank__baseline', np.nan):5.1f} ->"
            f" {s.get('effective_rank__plateau', np.nan):5.1f}"
            f"  decay x{s.get('spectral_decay__ratio', np.nan):6.2f}"
            f"  H1max/conn {s.get('h1_max_persistence__connectivity__ratio', np.nan):6.2f}"
            f"  /meanpair {s.get('h1_max_persistence__mean_pairwise__ratio', np.nan):6.2f}"
        )

    if not series:
        raise SystemExit(f"no runs with cached embedding snapshots under {args.root}")

    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(summaries)
    pd.concat(series, ignore_index=True).to_csv(args.out / "normalisation_series.csv", index=False)
    frame.to_csv(args.out / "normalisation.csv", index=False)

    if not args.runs:
        table = verdict_invariance(args.root, frame)
        table.to_csv(args.out / "normaliser_verdicts.csv", index=False)
        print("\nconditions clearing their null band, by normaliser:")
        for column in (c for c in table if c.endswith("__verdict")):
            counts = table[column].value_counts()
            name = column[: -len("__verdict")]
            print(
                f"  {name:44s} above {counts.get('above', 0):2d}"
                f"  inside {counts.get('inside', 0):2d}  below {counts.get('below', 0):2d}"
            )

    print(f"\nwritten to {args.out}/normalisation.csv and normalisation_series.csv")


if __name__ == "__main__":
    main()
