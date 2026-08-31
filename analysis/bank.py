"""The run bank as one frame, and the window rule of §4.2, stated once here."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

# Baseline is late memorisation, past the initialisation transient in raw H1; the plateau
# starts late because persistence keeps moving after t_g
BASELINE_WINDOW = (0.5, 0.9)
PLATEAU_FROM = 1.2
NULL_BASELINE_WINDOW = (0.3, 0.6)  # no t_g to anchor to; split the run instead
NULL_PLATEAU_FROM = 0.8

HEADLINE_OBSERVABLE = "h1_max_persistence_normalised"

RATIO_OBSERVABLES = (
    "h1_max_persistence",
    "h1_total_persistence",
    "h1_max_persistence_normalised",
    "h1_total_persistence_normalised",
    "h0_total_persistence",
    "lid",
)

_FOURIER = re.compile(r"^fourier_concentration(?P<group>_group)?(?:_k(?P<k>\d+))?$")


@dataclass(frozen=True)
class Run:
    name: str
    directory: Path
    config: dict
    summary: dict
    observables: pd.DataFrame


def iter_runs(root: Path, *, require_summary: bool = True):
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest, obs = d / "manifest.json", d / "analysis" / "observables.csv"
        summary = d / "analysis" / "summary.json"
        if not (manifest.exists() and obs.exists()):
            continue
        if require_summary and not summary.exists():
            continue
        config = json.loads(manifest.read_text())
        config = config.get("config", config)
        yield Run(
            name=d.name,
            directory=d,
            config=config,
            summary=json.loads(summary.read_text()) if summary.exists() else {},
            observables=pd.read_csv(obs).sort_values("step"),
        )


def fourier_variants(columns) -> dict[tuple[bool, int | None], str]:
    out = {}
    for c in columns:
        m = _FOURIER.match(c)
        if m:
            out[(bool(m.group("group")), int(m.group("k")) if m.group("k") else None)] = c
    return out


def select_fourier_k(runs: list[Run]) -> int | None:
    """The k whose series tracks test accuracy best, picked on the series and not the scalar."""
    scores: dict[int | None, list[float]] = {}
    for run in runs:
        if run.summary.get("grokking_step") is None:
            continue
        obs = run.observables
        if "test_acc" not in obs:
            continue
        for (group, k), col in fourier_variants(obs.columns).items():
            if group or col not in obs:
                continue
            a, b = obs[col].to_numpy(float), obs["test_acc"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 5 or np.ptp(a[ok]) == 0:
                continue
            rho = pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman")
            if np.isfinite(rho):
                scores.setdefault(k, []).append(abs(float(rho)))
    if not scores:
        return None
    return max(scores, key=lambda k: float(np.median(scores[k])))


# A circle is power in one frequency pair, so circularity wants a small k — unlike the
# predictive comparison, where the strongest competitor is a large one.
CIRCULARITY_K = 5


def circularity_column(operation: str, columns, k: int | None = CIRCULARITY_K) -> str | None:
    """Terminal concentration in the fairest basis: mul and div order theirs by discrete log."""
    if operation == "compose":
        return None
    variants = fourier_variants(columns)
    group = operation in {"mul", "div"}
    for key in ((group, k), (group, None), (False, k), (False, None)):
        if key in variants:
            return variants[key]
    return None


def is_replicate(run: Run) -> bool:
    """A re-run of a condition the main programme already covers; pooling would double-count."""
    return (
        "_dense_" in run.name
        or "_traj_" in run.name
        or run.config["train"].get("dense_to", 0) > 0
    )


def plateau_relaxed(
    obs: pd.DataFrame, tg: float | None, *, plateau_from: float = PLATEAU_FROM
) -> bool:
    """Has this run too few snapshots past the plateau bound to take a median?

    Thirteen grok inside their last fifth. Theirs starts at ``t_g`` instead, which understates
    a rising effect, and is reported because it departs from the rule §4.2 states.
    """
    steps = obs["step"].to_numpy(float)
    bound = plateau_from * tg if tg else NULL_PLATEAU_FROM * float(steps.max())
    return int((steps >= bound).sum()) < 2


def window_medians(
    obs: pd.DataFrame,
    column: str,
    tg: float | None,
    *,
    baseline: tuple[float, float] = BASELINE_WINDOW,
    plateau_from: float = PLATEAU_FROM,
) -> tuple[float, float]:
    steps, end = obs["step"].to_numpy(float), float(obs["step"].max())
    if tg:
        lo, hi, plat_lo = baseline[0] * tg, baseline[1] * tg, plateau_from * tg
    else:
        lo, hi = NULL_BASELINE_WINDOW[0] * end, NULL_BASELINE_WINDOW[1] * end
        plat_lo = NULL_PLATEAU_FROM * end
    if plateau_relaxed(obs, tg, plateau_from=plateau_from):
        plat_lo = tg or end
    base = obs.loc[(steps >= lo) & (steps <= hi), column].to_numpy(float)
    plat = obs.loc[steps >= plat_lo, column].to_numpy(float)
    with warnings.catch_warnings():  # an all-NaN window is expected for a diverged run
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            float(np.nanmedian(base)) if base.size else float("nan"),
            float(np.nanmedian(plat)) if plat.size else float("nan"),
        )


def task_modulus(data: dict) -> int:
    """The modulus a condition is defined by, recomputed rather than trusted: an early S_5
    batch predates the guard and carries the default 97."""
    if data.get("task") == "permutation_group":
        return factorial(int(data["n_symbols"]))
    return int(data["modulus"])


def config_fields(config: dict) -> dict:
    return {
        "model": config["model"]["name"],
        "operation": config["data"]["operation"],
        "modulus": task_modulus(config["data"]),
        "train_fraction": config["data"]["train_fraction"],
        "label_permutation": config["data"]["label_permutation"],
        "loss": config["train"]["loss"],
        "optimizer": config["train"]["optimizer"]["name"],
        "lr": config["train"]["optimizer"]["lr"],
        "weight_decay": config["train"]["optimizer"]["weight_decay"],
    }


def summarise(run: Run, fourier_k: int | None) -> dict:
    cfg, obs, summ = run.config, run.observables, run.summary
    tg = summ.get("grokking_step")
    row: dict[str, object] = {
        "run": run.name,
        **config_fields(cfg),
        "steps": cfg["train"]["steps"],
        "seed": cfg["seed"],
        "n_snapshots": len(obs),
        "replicate": is_replicate(run),
        "t_c": summ.get("train_convergence_step"),
        "t_g": tg,
        # the summary's top-level timing belongs to the raw series; the chapters quote the
        # normalised one, so both are named for the observable they describe
        "t_top__raw": summ.get("topological_transition_step"),
        "lead_lag_steps__raw": summ.get("lead_lag_steps"),
        "t_top": (summ.get("transitions") or {}).get(HEADLINE_OBSERVABLE, {}).get("t_top"),
        "lead_lag_steps": (summ.get("transitions") or {})
        .get(HEADLINE_OBSERVABLE, {})
        .get("delta"),
        # the same two under the second detector, which §3.6 promises to report
        "t_changepoint": (summ.get("transitions") or {})
        .get(HEADLINE_OBSERVABLE, {})
        .get("t_changepoint"),
        "lead_lag_steps__changepoint": (summ.get("transitions") or {})
        .get(HEADLINE_OBSERVABLE, {})
        .get("delta_changepoint"),
        "diverged": summ.get("diverged"),
        "grokked": tg is not None,
    }
    row["plateau_relaxed"] = plateau_relaxed(obs, tg)
    for column in RATIO_OBSERVABLES:
        if column not in obs:
            continue
        base, plat = window_medians(obs, column, tg)
        row[f"{column}__baseline"] = base
        row[f"{column}__plateau"] = plat
        row[f"{column}__ratio"] = plat / base if base not in (0.0, None) else float("nan")
    if "pointcloud_scale" in obs:
        scale = obs["pointcloud_scale"]
        first, last = float(scale.iloc[0]), float(scale.iloc[-1])
        row["scale_initial"], row["scale_final"] = first, last
        row["scale_collapse"] = first / last if last else float("nan")
    col = circularity_column(row["operation"], obs.columns)
    row["circularity_column"] = col
    row["circularity"] = float(obs[col].iloc[-1]) if col else float("nan")
    # weight_norm rides along here because it is Tan et al.'s comparator in §6.4
    for column in ("test_acc", "test_acc_novel", "weight_norm"):
        if column in obs:
            row[f"{column}__final"] = float(obs[column].iloc[-1])
    row.update(final_accuracies(run.directory))
    return row


def final_accuracies(run_dir: Path) -> dict:
    """Terminal accuracies and the gap, ``train - test``: §6.4's claim is about models that fit."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    last = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last = line
    if last is None:
        return {}
    record = json.loads(last)
    train, test = record.get("train_acc"), record.get("test_acc")
    if train is None or test is None:
        return {}
    return {
        "train_acc__final": float(train),
        "generalisation_gap": float(train) - float(test),
        "fits_train_set": bool(train >= 0.99),
    }


def load_bank(root: Path) -> tuple[pd.DataFrame, int | None]:
    runs = list(iter_runs(root))
    k = select_fourier_k(runs)
    return pd.DataFrame([summarise(r, k) for r in runs]), k


CONDITION_KEYS = [
    "model",
    "operation",
    "modulus",
    "train_fraction",
    "label_permutation",
    "loss",
    "optimizer",
    "lr",
    "weight_decay",
]


def condition_label(row) -> str:
    return (
        f"{row.model} {row.operation}{int(row.modulus)} f{row.train_fraction:g} "
        f"wd{row.weight_decay:g} lr{row.lr:g}"
        + (" PERM" if row.label_permutation else "")
        + ("" if row.loss == "softmax_ce" else " stablemax")
        + ("" if row.optimizer == "adamw" else " ortho")
    )


def bootstrap_median_ci(
    values, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    return (
        float(np.quantile(draws, alpha / 2)),
        float(np.median(v)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def condition_table(
    bank: pd.DataFrame,
    *,
    seed: int = 0,
    include_replicates: bool = False,
    observables: Sequence[str] = RATIO_OBSERVABLES,
) -> pd.DataFrame:
    if not include_replicates and "replicate" in bank:
        bank = bank[~bank.replicate]
    rows = []
    for key, sub in bank.groupby(CONDITION_KEYS, dropna=False):
        grokked = sub[sub.grokked]
        row = dict(zip(CONDITION_KEYS, key, strict=True))
        row["n_runs"] = len(sub)
        row["n_grokked"] = len(grokked)
        row["t_g_values"] = " ".join(str(int(x)) for x in sorted(grokked.t_g.dropna()))
        row["t_g_median"] = float(grokked.t_g.median()) if len(grokked) else float("nan")
        for column in observables:
            name = f"{column}__ratio"
            if name not in sub:
                continue
            source = grokked if len(grokked) else sub
            lo, med, hi = bootstrap_median_ci(source[name], seed=seed)
            row[f"{column}__lo"], row[f"{column}__med"], row[f"{column}__hi"] = lo, med, hi
        row["circularity"] = float(sub.circularity.median())
        row["scale_collapse"] = float(sub.scale_collapse.median())
        row["test_acc_final"] = float(sub["test_acc__final"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def null_runs(bank: pd.DataFrame) -> pd.DataFrame:
    """Runs that fit and never generalise: permuted labels, and the polynomial task."""
    bank = bank[~bank.replicate] if "replicate" in bank else bank
    return bank[(bank.label_permutation) | (bank.operation == "poly")]


def null_band(bank: pd.DataFrame, column: str = "h1_max_persistence_normalised__ratio") -> dict:
    v = np.asarray(null_runs(bank)[column], dtype=float)
    v = v[np.isfinite(v)]
    lo, med, hi = bootstrap_median_ci(v)
    return {
        "observable": column,
        "n_runs": int(v.size),
        "observed_min": float(v.min()) if v.size else float("nan"),
        "observed_max": float(v.max()) if v.size else float("nan"),
        "median": med,
        "bootstrap_ci_on_median": [lo, hi],
    }


def verdicts(
    bank: pd.DataFrame,
    conditions: pd.DataFrame,
    *,
    observables: Sequence[str] = RATIO_OBSERVABLES,
) -> pd.DataFrame:
    """Above / below / inside the null band: ``above`` needs the whole interval to clear it."""
    out = conditions.copy()
    for column in observables:
        name = f"{column}__ratio"
        if name not in bank:
            continue
        band = null_band(bank, name)
        lo_col, hi_col = f"{column}__lo", f"{column}__hi"
        if lo_col not in out:
            continue
        out[f"{column}__verdict"] = np.where(
            out[lo_col] > band["observed_max"],
            "above",
            np.where(out[hi_col] < band["observed_min"], "below", "inside"),
        )
    return out
