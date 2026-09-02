"""Sensitivity of the scale normalisation of §3.3.1: is the contraction a similarity?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from analysis import cli
from analysis.bank import CONDITION_KEYS
from grokking_tda.artifacts.reader import Run
from grokking_tda.config.schema import PointCloudCfg
from grokking_tda.tda.pointcloud import build_point_cloud

# singular values below this fraction of the largest are round-off, not structure
RANK_FLOOR = 1e-10

# Alternative answers to "how big is this cloud"; ``connectivity`` is the thesis normaliser
ALT_SCALES = ("mean_pairwise", "diameter", "rms_radius")
SUMMARIES = ("h1_max_persistence", "h1_total_persistence")


def cloud_geometry(cloud: np.ndarray) -> dict:
    # A diverged run's weights are NaN, and LAPACK fails to converge on them.
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
        # sigma_1 / sigma_10: the centred embedding is rank-deficient, so sigma_min is noise
        "spectral_decay": float(s[0] / s[min(9, s.size - 1)]),
        # Two weightings of "how many directions carry this cloud". The normaliser is justified
        # by being linear in the cloud's size, which holds for a similarity; if these fall, the
        # cloud is collapsing onto a subspace and loops in the lost directions shrink faster.
        "participation_ratio": float(var.sum() ** 2 / (var**2).sum()),
        "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
        "top2_variance_fraction": float(p[:2].sum()),
    }


def run_series(run_dir: Path) -> pd.DataFrame:
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
    # through bank.window_medians, so a normaliser is compared under the rule §4.2 states,
    # relaxed plateau included, and not under a second one written here
    from analysis.bank import window_medians

    out = {}
    for col in columns:
        base, plateau = window_medians(df, col, t_g)
        if not (np.isfinite(base) and np.isfinite(plateau)):
            continue
        out[f"{col}__baseline"] = base
        out[f"{col}__plateau"] = plateau
        out[f"{col}__ratio"] = plateau / base if base else np.nan
    return out


def verdict_invariance(root: Path, summaries: pd.DataFrame) -> pd.DataFrame:
    """Verdicts under each candidate normaliser, against a band recomputed under the same one."""
    from analysis.bank import condition_table, load_bank, verdicts

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


def in_window_drift(bank: pd.DataFrame) -> pd.DataFrame:
    """How much the cloud's scale moves *between the two windows an effect is read across*.

    §3.3.1 motivates the correction with the contraction over the whole run, which reaches
    $239\\times$; but the baseline window opens at $0.5\\,t_g$, so neither compared window touches
    initialisation and that is not the drift the ratio suffers. The quotient of the raw and
    normalised ratios *is* that drift, because the two differ only by the scale divided out.
    """
    raw, norm = "h1_max_persistence__ratio", "h1_max_persistence_normalised__ratio"
    # the condition table excludes re-runs, so this must too, or the two disagree on the same run
    bank = bank[~bank.replicate] if "replicate" in bank else bank
    drift = (bank[raw] / bank[norm]).replace([np.inf, -np.inf], np.nan)
    out = bank[["run", *CONDITION_KEYS, "scale_collapse"]].copy()
    out["in_window_drift"] = drift
    return out.dropna(subset=["in_window_drift"])


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
        from analysis.bank import load_bank

        bank, _ = load_bank(args.root)
        drift = in_window_drift(bank)
        drift.to_csv(args.out / "in_window_drift.csv", index=False)
        d = drift["in_window_drift"]
        ref = drift[
            (drift.modulus == 113) & (drift.weight_decay == 0.1) & (drift.operation == "add")
        ]
        print(
            f"\nscale drift between the compared windows, n={len(d)}: median {d.median():.3f}"
            f"  IQR {d.quantile(.25):.3f}-{d.quantile(.75):.3f}"
            f"  (whole-run collapse reaches {drift.scale_collapse.max():.0f}x)"
        )
        print(
            f"  reference regime: median {ref['in_window_drift'].median():.3f}"
            f" over {len(ref)} runs"
        )

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
