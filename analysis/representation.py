"""Which representation space carries the most informative topology? (RQ4, §7.1)"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from analysis import cli
from analysis.bank import (
    HEADLINE_OBSERVABLE,
    bootstrap_median_ci,
    condition_label,
    window_medians,
)
from grokking_tda.analysis import run_observables
from grokking_tda.analysis.identity import config_fields, task_modulus
from grokking_tda.artifacts import Run
from grokking_tda.config.schema import AnalysisCfg
from grokking_tda.evaluation import transition_step

KINDS = ("embedding", "hidden", "logits")

SPLIT = "test"  # as Tang et al.: the fitted split must not leak into the topology

LANDMARKS = 300  # the density analysis/torus.py works at

OBSERVABLES = (
    "h1_max_persistence",
    "h1_max_persistence_normalised",
    "pointcloud_scale",
    "test_acc",
)

# The two regimes §4.5 separates, the MLP with the bank's largest ratio, and the null, which
# reaches the embedding cell and no further
CONDITIONS = (
    "transformer_add113_f0.3_wd0.1_softmax_ce_s",
    "transformer_add97_f0.3_wd1.0_softmax_ce_s",
    "mlp_add97_f0.3_wd1.0_softmax_ce_s",
    "transformer_add97_permuted_s",
)

# The generalising condition with no signature; every cell is scored against it.
CONTROL = "transformer_add97_f0.3_wd1.0_softmax_ce_s"


def analysis_cfg(kind: str, max_points: int) -> AnalysisCfg:
    cfg = OmegaConf.structured(AnalysisCfg)
    cfg.representation = kind
    cfg.representation_split = SPLIT
    cfg.observables = list(OBSERVABLES)
    cfg.pointcloud.max_points = 0 if kind == "embedding" else max_points
    cfg.pointcloud.subsample = "maxmin"
    return cfg


def measure(run: Run, kind: str, max_points: int, *, stride: int) -> dict:
    cfg = analysis_cfg(kind, max_points)
    obs = run_observables(run, cfg)
    if stride > 1:
        obs = obs.iloc[::stride].reset_index(drop=True)
    summary = json.loads((run.dir / "analysis" / "summary.json").read_text())
    t_g = summary.get("grokking_step")

    row: dict[str, object] = {
        "run": run.run_name,
        "kind": kind,
        "landmarks": max_points,
        "n_snapshots": len(obs),
        "t_g": t_g,
    }
    for column in (HEADLINE_OBSERVABLE, "h1_max_persistence"):
        base, plateau = window_medians(obs, column, t_g)
        row[f"{column}__baseline"] = base
        row[f"{column}__plateau"] = plateau
        row[f"{column}__ratio"] = plateau / base if base else float("nan")
    t_top = transition_step(
        obs["step"].to_numpy(float), obs[HEADLINE_OBSERVABLE].to_numpy(float)
    )
    row["t_top"] = t_top
    row["lead_lag_steps"] = (t_g - t_top) if (t_g is not None and t_top is not None) else None
    row["scale_collapse"] = float(obs.pointcloud_scale.iloc[0] / obs.pointcloud_scale.iloc[-1])
    return row


def compare(table: pd.DataFrame) -> pd.DataFrame:
    """Each condition's ratio over seeds and its lead over the control, by representation."""
    ratio = f"{HEADLINE_OBSERVABLE}__ratio"
    table = table.copy()
    table["cardinality"] = np.where(
        table.kind == "embedding", "—", np.where(table.landmarks == LANDMARKS, "300", "matched")
    )
    rows = []
    for (kind, cardinality), cell in table.groupby(["kind", "cardinality"]):
        null = cell.loc[cell.is_null, ratio].dropna()
        band = (float(null.min()), float(null.max())) if len(null) else (np.nan, np.nan)
        control = cell.loc[cell.run.str.startswith(CONTROL), ratio].median()
        for condition, sub in cell[~cell.is_null].groupby("condition"):
            lo, med, hi = bootstrap_median_ci(sub[ratio])
            lag = sub.lead_lag_steps.dropna()
            rows.append(
                {
                    "condition": condition,
                    "kind": kind,
                    "cardinality": cardinality,
                    "n_seeds": len(sub),
                    "n_located": int(sub.t_top.notna().sum()),
                    "ratio_lo": lo,
                    "ratio_med": med,
                    "ratio_hi": hi,
                    "separation": med / control if control else float("nan"),
                    "null_lo": band[0],
                    "null_hi": band[1],
                    "verdict": _verdict(lo, hi, band),
                    "median_lag": float(lag.median()) if len(lag) else float("nan"),
                    "lag_spread": float(lag.max() - lag.min()) if len(lag) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _verdict(lo: float, hi: float, band: tuple[float, float]) -> str:
    if not np.isfinite(band[0]):
        return "no null"
    return "above" if lo > band[1] else "below" if hi < band[0] else "inside"


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--landmarks", type=int, default=LANDMARKS)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th snapshot")
    ap.add_argument("--runs", nargs="*")
    args = ap.parse_args()

    names = args.runs or [
        d.name
        for d in sorted(args.root.iterdir())
        if d.name.startswith(CONDITIONS) and (d / "analysis" / "summary.json").exists()
    ]
    if not names:
        raise SystemExit("no analysed runs in the conditions this comparison is defined on")

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "representation.csv"
    written = False
    for name in names:
        run = Run(args.root / name)
        modulus = task_modulus(run.config["data"])
        rows = []
        for kind in args.kinds:
            # the embedding is the residue table: no landmark choice to make
            sizes = (0,) if kind == "embedding" else (modulus, args.landmarks)
            for size in sizes:
                row = measure(run, kind, size, stride=args.stride)
                fields = config_fields(run.config)
                row["condition"] = condition_label(pd.Series(fields))
                row["is_null"] = bool(fields["label_permutation"])
                rows.append(row)
                print(
                    f"  {name:46s} {kind:9s} k={size:<4d}"
                    f"  ratio {row[f'{HEADLINE_OBSERVABLE}__ratio']:6.2f}"
                    f"  lag {row['lead_lag_steps'] if row['lead_lag_steps'] is not None else '—'}",
                    flush=True,
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(destination, mode="a" if written else "w", header=not written, index=False)
        written = True

    table = pd.read_csv(destination)
    summary = compare(table)
    summary.to_csv(args.out / "representation_conditions.csv", index=False)

    print(f"\n{HEADLINE_OBSERVABLE} across the transition, by representation:")
    header = ("condition", "rep", "k", "ratio", "vs control", "lag", "null")
    print(f"{header[0]:<40}{header[1]:>10}{header[2]:>9}{header[3]:>22}"
          f"{header[4]:>11}{header[5]:>10}{header[6]:>10}")
    for _, r in summary.sort_values(["kind", "cardinality", "condition"]).iterrows():
        interval = f"{r.ratio_med:.2f} [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}]"
        band = "—" if not np.isfinite(r.null_lo) else f"{r.null_lo:.2f}-{r.null_hi:.2f}"
        lag = "—" if not np.isfinite(r.median_lag) else f"{r.median_lag:,.0f}"
        print(
            f"{r.condition[:40]:<40}{r.kind:>10}{r.cardinality:>9}{interval:>22}"
            f"{r.separation:>11.2f}{lag:>10}{band:>10}"
        )
    print(f"\nwritten to {destination} and representation_conditions.csv")


if __name__ == "__main__":
    main()
