"""The run bank as one tidy frame, plus the window rule the thesis quotes.

Every number in Chapters 4-7 is derived here, so that each has a path from the artefact
store to the page. Two rules keep this layer honest.

Timing quantities (``t_top``, lead/lag) are **passed through** from each run's summary,
never recomputed: the detector lives in ``grokking_tda.evaluation.transitions`` and a
second implementation here would silently diverge from it. Anything this module derives
itself is a function of the observable series and ``t_g`` alone, and ``t_g`` is a
threshold crossing on test accuracy, so none of it depends on the transition detector.

The window rule is stated once, here, and matches thesis section 4.2.
"""

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

# Thesis section 4.2. Baseline is late memorisation: after the initialisation transient
# in raw H1 has decayed, before the transition begins. The plateau starts at 1.2 t_g
# because persistence keeps moving for a while after the accuracy threshold is crossed.
BASELINE_WINDOW = (0.5, 0.9)
PLATEAU_FROM = 1.2
# A run that never transitions has no t_g to anchor to; split the run instead.
NULL_BASELINE_WINDOW = (0.3, 0.6)
NULL_PLATEAU_FROM = 0.8

# The series every timing claim in chapters 4-7 is stated on.
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
    """Map ``(is_group, k)`` to column name for every Fourier column present."""
    out = {}
    for c in columns:
        m = _FOURIER.match(c)
        if m:
            out[(bool(m.group("group")), int(m.group("k")) if m.group("k") else None)] = c
    return out


def select_fourier_k(runs: list[Run]) -> int | None:
    """The k whose series tracks test accuracy best, chosen once for the whole bank.

    Concentration rises monotonically in k, so the strongest competitor cannot be picked
    by maximising the scalar; it has to be picked on the time series. One global choice,
    made on grokking runs only, and reported.
    """
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


# Circularity is concentration in a *few* modes: a circle is one dominant frequency pair,
# so a large k measures something other than circularity even though it is the strongest
# competitor in a predictive comparison. The two roles use different k, and the
# sensitivity of the association to this choice is reported.
CIRCULARITY_K = 5


def circularity_column(operation: str, columns, k: int | None = CIRCULARITY_K) -> str | None:
    """Terminal concentration in the fairest basis available for this operation.

    Multiplication and division arrange the grokked circle by discrete logarithm, so the
    residue-axis transform is blind to it and the group variant is the fair competitor.
    """
    variants = fourier_variants(columns)
    group = operation in {"mul", "div"}
    for key in ((group, k), (group, None), (False, k), (False, None)):
        if key in variants:
            return variants[key]
    return None


def _window(obs: pd.DataFrame, column: str, tg: float | None) -> tuple[float, float]:
    steps, end = obs["step"].to_numpy(float), float(obs["step"].max())
    if tg:
        lo, hi, plat_lo = BASELINE_WINDOW[0] * tg, BASELINE_WINDOW[1] * tg, PLATEAU_FROM * tg
    else:
        lo, hi = NULL_BASELINE_WINDOW[0] * end, NULL_BASELINE_WINDOW[1] * end
        plat_lo = NULL_PLATEAU_FROM * end
    base = obs.loc[(steps >= lo) & (steps <= hi), column].to_numpy(float)
    plat = obs.loc[steps >= plat_lo, column].to_numpy(float)
    if plat.size < 2:  # a run that groks near its budget edge
        plat = obs.loc[steps >= (tg or end), column].to_numpy(float)
    # A diverged run has NaN observables throughout; an all-NaN window is expected there
    # and the divergence is recorded in the summary, so the warning is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            float(np.nanmedian(base)) if base.size else float("nan"),
            float(np.nanmedian(plat)) if plat.size else float("nan"),
        )


def task_modulus(data: dict) -> int:
    """The modulus a condition is actually defined by.

    For the permutation-group task this field records the *group order*, which is what
    ``TaskMeta`` is built from and what every downstream quantity depends on. An early
    batch of S_5 runs was launched before ``data/permutation.py`` gained the guard that
    rejects a mismatch, so their config carries the default 97 while their data, model and
    vocabulary are all S_5 exactly like the later batch's. Left uncorrected, one condition
    appears in the table as two --- which is why the S_5 result rested on five seeds when
    seven exist. The order is recomputed here rather than trusted, so the fix does not
    depend on which batch a run came from.
    """
    if data.get("task") == "permutation_group":
        return factorial(int(data["n_symbols"]))
    return int(data["modulus"])


def summarise(run: Run, fourier_k: int | None) -> dict:
    cfg, obs, summ = run.config, run.observables, run.summary
    tg = summ.get("grokking_step")
    row: dict[str, object] = {
        "run": run.name,
        "model": cfg["model"]["name"],
        "operation": cfg["data"]["operation"],
        "modulus": task_modulus(cfg["data"]),
        "train_fraction": cfg["data"]["train_fraction"],
        "label_permutation": cfg["data"]["label_permutation"],
        "loss": cfg["train"]["loss"],
        "optimizer": cfg["train"]["optimizer"]["name"],
        "lr": cfg["train"]["optimizer"]["lr"],
        "weight_decay": cfg["train"]["optimizer"]["weight_decay"],
        "steps": cfg["train"]["steps"],
        "seed": cfg["seed"],
        "n_snapshots": len(obs),
        # The dense re-runs repeat existing conditions on a different snapshot schedule,
        # which changes every window median; they belong to the trajectory analysis and
        # must not be pooled with the main programme.
        "dense": "_dense_" in run.name or cfg["train"].get("dense_to", 0) > 0,
        # passed through from the pipeline, never recomputed here
        "t_c": summ.get("train_convergence_step"),
        "t_g": tg,
        # The summary's top-level timing belongs to its *default* observable, which is the
        # raw H1 maximum; the chapters quote the scale-normalised series. Both are named for
        # the observable they describe, because a bare "lead_lag_steps" invites reading the
        # raw lag as the headline one.
        "t_top__raw": summ.get("topological_transition_step"),
        "lead_lag_steps__raw": summ.get("lead_lag_steps"),
        "t_top": (summ.get("transitions") or {}).get(HEADLINE_OBSERVABLE, {}).get("t_top"),
        "lead_lag_steps": (summ.get("transitions") or {})
        .get(HEADLINE_OBSERVABLE, {})
        .get("delta"),
        "diverged": summ.get("diverged"),
        "grokked": tg is not None,
    }
    for column in RATIO_OBSERVABLES:
        if column not in obs:
            continue
        base, plat = _window(obs, column, tg)
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
    for column in ("test_acc", "test_acc_novel"):
        if column in obs:
            row[f"{column}__final"] = float(obs[column].iloc[-1])
    row.update(final_accuracies(run.directory))
    return row


def final_accuracies(run_dir: Path) -> dict:
    """Terminal train and test accuracy, and the generalisation gap between them.

    Read from the last line of ``metrics.jsonl`` rather than from the observables, which
    carry test accuracy only. The gap is ``train - test`` and not ``1 - test``: the two
    agree wherever the network fits its training set, and where it does not they disagree
    completely. A run that reaches training accuracy 0.01 has a gap of roughly zero and
    not of one, because it has learned nothing to fail to transfer -- which is a different
    object from a network that memorises perfectly and generalises not at all, and only
    the second is what the trajectory-dimension claim of thesis section 6.4 is about.
    """
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
    """The bank, and the k whose Fourier series best tracks the transition.

    The returned k is the strongest *competitor* for the redundancy comparison; the
    circularity column uses ``CIRCULARITY_K`` instead, for the reason given above it.
    """
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
    include_dense: bool = False,
    observables: Sequence[str] = RATIO_OBSERVABLES,
) -> pd.DataFrame:
    """One row per experimental condition, with bootstrap intervals over seeds."""
    if not include_dense and "dense" in bank:
        bank = bank[~bank.dense]
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
    """Runs that fit their training data and never generalise.

    Two nulls, pooled: permuted labels (no rule to find) and the polynomial task (a real
    rule the network memorises completely and never generalises on). Dense re-runs are
    excluded for the same reason they are excluded from the condition table.
    """
    bank = bank[~bank.dense] if "dense" in bank else bank
    return bank[(bank.label_permutation) | (bank.operation == "poly")]


def null_band(bank: pd.DataFrame, column: str = "h1_max_persistence_normalised__ratio") -> dict:
    """The interval a ratio takes under the null models, per observable."""
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
    """Mark each condition above / below / inside the null band, per observable.

    ``above`` requires the whole bootstrap interval over seeds to exceed the largest
    ratio any null run produced; ``below`` requires it to fall short of the smallest.
    """
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
