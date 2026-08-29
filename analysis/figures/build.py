"""Tidy data for every thesis figure, one CSV per figure.

Separating figure *data* from figure *rendering* means the plotting layer never reaches
into the artefact store, and every panel can be checked as a table before it is drawn.
Each builder declares its figure number and the one claim that figure has to make
undeniable; both are written into the manifest beside the data.

Usage (from ``Code/``)::

    uv run python -m analysis.figures.build
    uv run python -m analysis.figures.build --only 4.2 4.6

Each builder returns a frame and is registered by figure number; a builder whose inputs
do not exist yet skips with a note instead of failing, so the set fills in as analyses
land.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bank import (
    BASELINE_WINDOW,
    PLATEAU_FROM,
    RATIO_OBSERVABLES,
    bootstrap_median_ci,
    condition_table,
    load_bank,
    null_band,
    verdicts,
)

BUILDERS: dict[str, Callable] = {}


def builder(number: str, name: str, claim: str):
    def register(fn):
        fn.number, fn.figure_name, fn.claim = number, name, claim
        BUILDERS[number] = fn
        return fn

    return register


class Missing(Exception):
    """An input this figure needs has not been produced yet."""


# --------------------------------------------------------------------------- helpers


def _metrics(root: Path, run: str) -> pd.DataFrame:
    path = root / run / "metrics.jsonl"
    if not path.exists():
        raise Missing(f"no metrics for {run}")
    return pd.DataFrame([json.loads(line) for line in path.open() if line.strip()]).sort_values(
        "step"
    )


def _observables(root: Path, run: str) -> pd.DataFrame:
    path = root / run / "analysis" / "observables.csv"
    if not path.exists():
        raise Missing(f"no observables for {run}")
    return pd.read_csv(path).sort_values("step")


def _summary(root: Path, run: str) -> dict:
    path = root / run / "analysis" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _pick(bank: pd.DataFrame, **conditions) -> list[str]:
    mask = pd.Series(True, index=bank.index)
    for key, value in conditions.items():
        mask &= bank[key] == value
    return sorted(bank.loc[mask, "run"])


def _diagrams(root: Path, run: str) -> tuple[list[int], list[np.ndarray | None]]:
    """Cached per-snapshot degree-one diagrams, in step order."""
    cache = root / run / "analysis" / "diagrams"
    if not cache.exists():
        raise Missing(f"no cached diagrams for {run}")
    steps, bars = [], []
    for path in sorted(cache.glob("step_*.npz")):
        with np.load(path, allow_pickle=True) as f:
            bars.append(f["dim1"] if "dim1" in f.files else None)
        steps.append(int(path.stem.split("_")[1]))
    return steps, bars


def _diagram_file(root: Path, run: str, step: int) -> str:
    return next((root / run / "analysis" / "diagrams").glob(f"step_{step:08d}_*.npz")).name


def _finite(bars: np.ndarray | None) -> np.ndarray:
    if bars is None or bars.size == 0:
        return np.empty((0, 2))
    return bars[np.isfinite(bars[:, 1])]


# The fifteen conditions of thesis table 4.2: everything that groks under the reference
# loss and optimiser on a task with a group structure. The interventions are chapter 5's
# subject and the polynomial is a null, so both are excluded here; the two-seed S_5
# fraction sweep is excluded with them, leaving the five-seed condition the chapter reports.
def _main_programme(table: pd.DataFrame) -> pd.DataFrame:
    return table[
        (table.loss == "softmax_ce")
        & (table.optimizer == "adamw")
        & (table.operation != "poly")
        & (~table.label_permutation)
        & (table.n_grokked > 0)
    ]


# -------------------------------------------------------------------------- builders


@builder("1.1", "hero", "The network fits in 200 steps and generalises 27,600 steps later.")
def fig_hero(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    run = "transformer_add97_f0.3_wd1.0_softmax_ce_s1"
    metrics, obs, summ = _metrics(root, run), _observables(root, run), _summary(root, run)
    frame = metrics[["step", "train_acc", "test_acc"]].copy()
    frame["run"] = run
    frame["t_c"] = summ.get("train_convergence_step")
    frame["t_g"] = summ.get("grokking_step")
    novel = obs[["step", "test_acc_novel"]].rename(columns={"test_acc_novel": "test_acc_novel"})
    return frame.merge(novel, on="step", how="outer").sort_values("step")


@builder("4.1", "reproduction", "Both architectures and both recipes grok; the null does not.")
def fig_reproduction(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    panels = {
        "canonical transformer": dict(
            model="transformer", operation="add", modulus=97, train_fraction=0.3, weight_decay=1.0
        ),
        "reference transformer": dict(
            model="transformer", operation="add", modulus=113, weight_decay=0.1
        ),
        "canonical MLP": dict(model="mlp", operation="add", modulus=97, weight_decay=1.0),
    }
    rows = []
    # the permuted-label null is the fourth panel: the same measurement on the condition
    # that must not grok, so the claim and its control share one field
    main = bank[(bank.loss == "softmax_ce") & (~bank.replicate) & (~bank.label_permutation)]
    null = bank[(bank.loss == "softmax_ce") & (~bank.replicate) & bank.label_permutation]
    sources = {p: main for p in panels}
    panels["permuted labels"] = dict(model="transformer", operation="add", modulus=97,
                                     train_fraction=0.3, weight_decay=1.0)
    sources["permuted labels"] = null
    for panel, conditions in panels.items():
        for run in _pick(sources[panel], **conditions):
            summ = _summary(root, run)
            metrics = _metrics(root, run)[["step", "train_acc", "test_acc"]]
            metrics = metrics.assign(
                panel=panel, run=run, seed=int(bank.loc[bank.run == run, "seed"].iloc[0])
            )
            metrics["t_g"] = summ.get("grokking_step")
            rows.append(metrics)
    if not rows:
        raise Missing("no reproduction runs")
    return pd.concat(rows, ignore_index=True)


@builder(
    "4.2",
    "signature",
    "Normalised, the reference regime rises cleanly; the canonical regime has nothing to clean.",
)
def fig_signature(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    panels = {
        "reference": dict(model="transformer", operation="add", modulus=113, weight_decay=0.1),
        "canonical": dict(
            model="transformer", operation="add", modulus=97, train_fraction=0.3, weight_decay=1.0
        ),
    }
    columns = [
        "h1_max_persistence",
        "h1_max_persistence_normalised",
        "pointcloud_scale",
        "test_acc",
    ]
    rows = []
    main = bank[(bank.loss == "softmax_ce") & (~bank.replicate)]
    for panel, conditions in panels.items():
        for run in _pick(main, **conditions):
            obs = _observables(root, run)
            keep = ["step"] + [c for c in columns if c in obs]
            frame = obs[keep].assign(panel=panel, run=run)
            frame["t_g"] = _summary(root, run).get("grokking_step")
            rows.append(frame)
    if not rows:
        raise Missing("no signature runs")
    return pd.concat(rows, ignore_index=True)


@builder("4.3", "diagrams", "One long-lived cycle separates from the diagonal and stays separated.")
def fig_diagrams(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    """Three degree-one diagrams from one reference run, with the scale that normalises them.

    Steps are taken from the run's own snapshot grid — the nearest available to
    mid-memorisation, the grokking step and the end — never interpolated.
    """
    run = "transformer_add113_f0.3_wd0.1_softmax_ce_s0"
    steps, bars = _diagrams(root, run)
    obs, summ = _observables(root, run), _summary(root, run)
    tg = summ.get("grokking_step") or steps[len(steps) // 2]
    wanted = [0.25 * tg, tg, steps[-1]]
    chosen = [min(steps, key=lambda s: abs(s - w)) for w in wanted]

    rows = []
    for stage, step in zip(("memorising", "grokking", "final"), chosen, strict=True):
        scale = float(obs.loc[obs.step == step, "pointcloud_scale"].iloc[0])
        for dim, key in ((0, "dim0"), (1, "dim1")):
            with np.load(root / run / "analysis" / "diagrams" / _diagram_file(root, run, step),
                         allow_pickle=True) as f:
                d = _finite(f[key] if key in f.files else None)
            for birth, death in d:
                rows.append({"stage": stage, "step": step, "dim": dim, "birth": birth,
                             "death": death, "scale": scale, "run": run, "t_g": tg})
    return pd.DataFrame(rows)


@builder("6.1", "crocker", "The band of live features slides as the cloud contracts.")
def fig_crocker(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    """CROCKER surfaces on both scale axes, for the two regimes, seed 0.

    The scale grid is geometric rather than uniform. Filtration values span three orders
    of magnitude over training, so a uniform grid spends nearly all of its rows on the
    early cloud and resolves the late one into two or three cells.

    Long format — one row per (regime, axis, step, scale) cell — because a tidy CSV a
    reader can check beats a pickled matrix, and 2 x 2 x 120 x 64 is small.
    """
    from grokking_tda.tda.trajectory import betti_at_scales

    regimes = {
        "reference": "transformer_add113_f0.3_wd0.1_softmax_ce_s0",
        "canonical": "transformer_add97_f0.3_wd1.0_softmax_ce_s0",
    }
    rows = []
    for regime, run in regimes.items():
        steps, bars = _diagrams(root, run)
        obs, tg = _observables(root, run), _summary(root, run).get("grokking_step")
        scale = obs.set_index("step")["pointcloud_scale"]
        raw = [_finite(b) for b in bars]
        normalised = [d / scale.get(s, np.nan) if d.size else d
                      for s, d in zip(steps, raw, strict=True)]
        for axis, diagrams in (("raw", raw), ("normalised", normalised)):
            births = np.concatenate([d[:, 0] for d in diagrams if d.size])
            deaths = np.concatenate([d[:, 1] for d in diagrams if d.size])
            grid = np.geomspace(max(births[births > 0].min(), 1e-6), deaths.max(), 64)
            for step, d in zip(steps, diagrams, strict=True):
                for s, b in zip(grid, betti_at_scales(d, grid), strict=True):
                    rows.append({"regime": regime, "axis": axis, "step": step,
                                 "scale": float(s), "betti1": int(b), "t_g": tg, "run": run})
    return pd.DataFrame(rows)


@builder("4.4", "robustness", "Five of sixteen conditions clear the null; one runs below it.")
def fig_robustness(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    """The rows of thesis table 4.2, plus the two pooled null rows beneath them.

    The nulls are pooled across their conditions here, as the table pools them: eleven
    polynomial runs spanning two weight decays are one null model, not four.
    """
    table = verdicts(bank, condition_table(bank))
    keep = [
        "model",
        "operation",
        "modulus",
        "train_fraction",
        "weight_decay",
        "label_permutation",
        "loss",
        "optimizer",
        "n_runs",
        "n_grokked",
        "t_g_median",
        "circularity",
        "scale_collapse",
    ]
    stats = [c for c in table.columns if c.endswith(("__lo", "__med", "__hi", "__verdict"))]
    out = _main_programme(table)[keep + stats].copy()
    out["block"] = "condition"
    out = out.sort_values("h1_max_persistence_normalised__med", ascending=False)

    # The intervention arms are chapter 5's subject and do not belong in the forest, but
    # figure 5.6 needs their intervals against the same band; tagged, not duplicated.
    arms = table[
        ((table.loss == "stablemax_ce") | (table.optimizer.str.startswith("orthograd")))
        & (table.n_grokked > 0)
    ][keep + stats].copy()
    arms["block"] = "intervention"
    out = pd.concat([out, arms], ignore_index=True)

    main = bank[~bank.replicate]
    nulls = {"permuted": main[main.label_permutation], "poly": main[main.operation == "poly"]}
    out["lr"] = np.nan
    for name, runs in nulls.items():
        # "null" alone round-trips through read_csv as a missing value
        row = {"block": "null model", "operation": name, "n_runs": len(runs), "n_grokked": 0,
               "circularity": float(runs.circularity.median()),
               "scale_collapse": float(runs.scale_collapse.median())}
        for column in RATIO_OBSERVABLES:
            lo, med, hi = bootstrap_median_ci(runs[f"{column}__ratio"])
            row[f"{column}__lo"], row[f"{column}__med"], row[f"{column}__hi"] = lo, med, hi
            row[f"{column}__verdict"] = "inside"
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    for column in RATIO_OBSERVABLES:
        band = null_band(bank, f"{column}__ratio")
        out[f"{column}__null_lo"] = band["observed_min"]
        out[f"{column}__null_hi"] = band["observed_max"]
    return out


@builder("4.6", "circularity", "Circularity predicts the signature; generalisation does not.")
def fig_circularity(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    # the dense re-runs repeat conditions the main programme already holds, so they are
    # excluded here for the same reason analysis/circularity.py excludes them: the panel and
    # the association section 4.5 quotes have to be computed over one set of runs
    m = bank[bank.grokked & (bank.operation != "compose") & (~bank.replicate)]
    keep = [
        "run",
        "model",
        "operation",
        "modulus",
        "train_fraction",
        "weight_decay",
        "seed",
        "circularity",
        "circularity_column",
        "h1_max_persistence__ratio",
        "h1_max_persistence_normalised__ratio",
        "h1_total_persistence_normalised__ratio",
        "test_acc__final",
    ]
    return m[[c for c in keep if c in m]].copy()


@builder(
    "5.1", "basis", "The circle is there under multiplication; the residue basis cannot see it."
)
def fig_basis(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for operation in ("mul", "div"):
        for run in _pick(bank, operation=operation):
            obs = _observables(root, run)
            columns = [
                c
                for c in obs.columns
                if c.startswith("fourier_concentration") or c == "h1_max_persistence_normalised"
            ]
            frame = obs[["step"] + columns].assign(run=run, operation=operation)
            frame["t_g"] = _summary(root, run).get("grokking_step")
            rows.append(frame)
    if not rows:
        raise Missing("no mul/div runs")
    return pd.concat(rows, ignore_index=True)


@builder(
    "5.2", "noncyclic", "S5 groks with a huge seed spread, and the signature is present anyway."
)
def fig_noncyclic(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    runs = _pick(bank, operation="compose", modulus=120, train_fraction=0.6)
    if not runs:
        raise Missing("no five-seed S_5 runs")
    rows = []
    for run in runs:
        obs, summ = _observables(root, run), _summary(root, run)
        tg = summ.get("grokking_step")
        frame = obs[
            ["step", "test_acc", "test_acc_novel", "h1_total_persistence_normalised"]
        ].assign(run=run, t_g=tg)
        frame["step_over_tg"] = frame["step"] / tg if tg else np.nan
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    band = null_band(bank, "h1_total_persistence_normalised__ratio")
    out["null_lo"], out["null_hi"] = band["observed_min"], band["observed_max"]
    return out


@builder(
    "5.6",
    "interventions",
    "The interventions work on one architecture and destabilise the other.",
)
def fig_interventions(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    m = bank[
        (bank.loss == "stablemax_ce") | (bank.optimizer.str.startswith("orthograd"))
    ]
    # both architectures' weight-decay baselines, so each panel has its own reference
    baseline = bank[
        (bank.operation == "add") & (bank.modulus == 97) & (bank.train_fraction == 0.3)
        & (bank.weight_decay == 1.0) & (bank.loss == "softmax_ce") & (bank.optimizer == "adamw")
        & (~bank.label_permutation) & (~bank.replicate)
    ]
    rows = []
    for run in sorted(set(m.run) | set(baseline.run)):
        info = bank[bank.run == run].iloc[0]
        metrics = _metrics(root, run)
        # a diverged run keeps logging accuracy after its loss goes non-finite; the series
        # is cut there, because what follows is not a trajectory
        finite = np.isfinite(metrics[["train_loss", "test_loss"]].to_numpy(float)).all(axis=1)
        metrics = metrics.iloc[: len(finite) if finite.all() else int(finite.argmin())]
        metrics = metrics[["step", "train_acc", "test_acc"]].assign(
            run=run,
            model=info.model,
            loss=info.loss,
            optimizer=info.optimizer,
            lr=info.lr,
            weight_decay=info.weight_decay,
            t_g=info.t_g,
            diverged=info.get("diverged"),
        )
        rows.append(metrics)
    if not rows:
        raise Missing("no intervention runs")
    return pd.concat(rows, ignore_index=True)


@builder("5.3", "headtohead",
         "Topology adds little over the cheap baselines, and nothing to timing.")
def fig_headtohead(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    """The grouped-fold comparison, as ``gtda-compare`` wrote it."""
    path = Path("results/processed/head_to_head.csv")
    if not path.exists():
        raise Missing("no head_to_head.csv — run gtda-compare")
    frame = pd.read_csv(path)
    frame["window_steps"] = frame.window.str.lstrip("w").replace({"tc": np.nan}).astype(float)
    return frame.sort_values(["task", "window_steps", "feature_set"])


@builder("5.4", "pid", "What topology says about the transition is mostly said by Fourier too.")
def fig_pid(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    """One row per (regime, estimator, atom), with its interval and its permutation null.

    The atoms are meaningless without the null: the binned estimator has an upward bias at
    this sample size, so a synergy of 0.33 against a shuffled-target null of 0.23 is a very
    different claim from a synergy of 0.33 against nothing.
    """
    path = Path("results/processed/thesis/pid.json")
    if not path.exists():
        raise Missing("no pid.json — run analysis.pid")
    d = json.loads(path.read_text())
    rows = []
    for regime, estimators in d.get("regimes", {}).items():
        for estimator, block in estimators.items():
            total = block.get("atoms", {}).get("total", {}).get("estimate")
            for atom, value in block.get("atoms", {}).items():
                if atom == "total" or not isinstance(value, dict):
                    continue
                rows.append({
                    "regime": regime, "estimator": estimator, "atom": atom,
                    "bits": value["estimate"],
                    "share": value["estimate"] / total if total else float("nan"),
                    "ci_lo": value["ci"][0], "ci_hi": value["ci"][1],
                    "null_median": value["null_median"], "null_p": value["null_p"],
                    "total": total, "n": block["n"],
                    "n_configurations": block["n_configurations"],
                    "n_effective": block["n_effective"],
                    "target": d["target"], "source_a": d["source_a"], "source_b": d["source_b"],
                })
    if not rows:
        raise Missing("pid.json has no regimes")
    return pd.DataFrame(rows)


@builder(
    "4.7", "torus", "The joint representation is a torus upstream and a circle at the readout."
)
def fig_torus(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    """Degree-two homology by stage, depth and ambient dimension.

    The ambient rows are the honest half of the panel and belong on it: they are where the
    structure is invisible, and the contrast with the projected rows is the finding.
    """
    path = Path("results/processed/thesis/torus.csv")
    if not path.exists():
        raise Missing("no torus.csv — run analysis.torus")
    frame = pd.read_csv(path)
    condition = frame.run.str.replace(r"_s\d+$", "", regex=True)
    return frame.assign(
        condition=condition,
        regime=np.where(
            condition.str.startswith("mlp"),
            "mlp",
            np.where(condition.str.contains("113"), "reference", "canonical"),
        ),
    )


@builder("6.4", "depth", "The torus appears at the operand layer, and only after the transition.")
def fig_depth(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    """The same measurement as 4.7, crossed over depth and training stage.

    4.7 asks where in *depth* the structure lives at one stage; this asks the same of every
    stage at once, which is the pair of axes section 6.5 exists to cross. Only the plane the
    structure is legible in is kept --- section 4.6 establishes that it is nowhere else ---
    and the cell is the mean over seeds and landmark draws, so a cell of exactly 2 means
    every draw agreed.
    """
    frame = fig_torus(root, bank)
    plane = frame[frame.pca_dim == 2]
    if plane.empty:
        raise Missing("torus.csv has no top-two-plane rows")
    return (
        plane.groupby(["regime", "condition", "stage", "depth", "layer"], as_index=False)
        .agg(
            cells=("betti_1", "size"),
            betti_1=("betti_1", "mean"),
            betti_2=("betti_2", "mean"),
            torus_frac=("betti_2", lambda s: float((s == 1).mean())),
            life_1=("life_1", "median"),
            life_2=("life_2", "median"),
        )
    )


@builder("5.5", "lag", "Topology follows generalisation, and leads nowhere.")
def fig_lag(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    """Signed lag per observable per run, plus the null's located transitions.

    Timing comes from each run's summary and is never recomputed here: the detector lives
    in ``grokking_tda.evaluation.transitions``.
    """
    observables = (
        "h1_max_persistence_normalised",
        "h1_total_persistence_normalised",
        "h1_max_persistence",
    )
    rows = []
    for run in bank[~bank.replicate].run:
        summ = _summary(root, run)
        transitions = summ.get("transitions") or {}
        info = bank[bank.run == run].iloc[0]
        for obs in observables:
            entry = transitions.get(obs) or {}
            rows.append(
                {
                    "run": run,
                    "observable": obs,
                    "model": info.model,
                    "operation": info.operation,
                    "modulus": info.modulus,
                    "train_fraction": info.train_fraction,
                    "weight_decay": info.weight_decay,
                    "label_permutation": info.label_permutation,
                    "loss": info.loss,
                    "optimizer": info.optimizer,
                    "t_g": summ.get("grokking_step"),
                    "t_top": entry.get("t_top"),
                    "delta": entry.get("delta"),
                }
            )
    if not rows:
        raise Missing("no transitions in any summary")
    return pd.DataFrame(rows)


@builder("6.2", "velocity", "The representation reorganises fastest around the transition.")
def fig_velocity(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    """Topological speed for three conditions, recomputed on scale-normalised diagrams.

    The pipeline's ``trajectory_distance.csv`` takes the distance between raw diagrams,
    so it inherits the contraction of section 3.3.1 and measures shrinkage as motion.
    Dividing each diagram by its own snapshot's connectivity scale first is the same
    correction the level series carries, and the diagrams are cached, so it is cheap.

    The rate rather than the distance is reported: the snapshot grid is logarithmic
    outside the dense window, and a raw consecutive distance confounds reorganisation
    with sampling interval.
    """
    from grokking_tda.tda.distances import diagram_distance

    panels = {
        "reference": dict(model="transformer", operation="add", modulus=113, weight_decay=0.1),
        "canonical": dict(model="transformer", operation="add", modulus=97,
                          train_fraction=0.3, weight_decay=1.0),
        "permuted": dict(model="transformer", operation="add", modulus=97,
                         label_permutation=True),
    }
    main = bank[(bank.loss == "softmax_ce") & (~bank.replicate)]
    rows = []
    for panel, conditions in panels.items():
        pick = main if panel == "permuted" else main[~main.label_permutation]
        for run in _pick(pick, **conditions):
            steps, bars = _diagrams(root, run)
            scale = _observables(root, run).set_index("step")["pointcloud_scale"]
            diagrams = [
                _finite(b) / scale.get(s, np.nan) if _finite(b).size else _finite(b)
                for s, b in zip(steps, bars, strict=True)
            ]
            tg = _summary(root, run).get("grokking_step")
            for i in range(1, len(steps)):
                gap = steps[i] - steps[i - 1]
                distance = diagram_distance(diagrams[i - 1], diagrams[i])
                rows.append({"panel": panel, "run": run, "step": steps[i], "gap": gap,
                             "distance": distance, "rate": distance / gap if gap else np.nan,
                             "t_g": tg})
    if not rows:
        raise Missing("no cached diagrams for the velocity panels")
    return pd.DataFrame(rows)


@builder("6.3", "phdim", "The geometry of the search turns after the network begins to generalise.")
def fig_phdim(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in bank.run:
        path = root / run / "analysis" / "ph_dimension.csv"
        if not path.exists():
            continue
        info = bank[bank.run == run].iloc[0]
        rows.append(
            pd.read_csv(path).assign(
                run=run, t_g=info.t_g, operation=info.operation,
                weight_decay=info.weight_decay, label_permutation=info.label_permutation,
                generalisation_gap=info.generalisation_gap, fits_train_set=info.fits_train_set,
            )
        )
    if not rows:
        raise Missing("no ph_dimension.csv anywhere — run gtda-analyse on the dense set")
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------ cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/raw"))
    parser.add_argument("--out", type=Path, default=Path("results/processed/thesis/figures"))
    parser.add_argument("--only", nargs="*", help="figure numbers, e.g. 4.2 4.6")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank, _ = load_bank(args.root)
    wanted = args.only or sorted(BUILDERS)
    manifest = []
    for number in wanted:
        fn = BUILDERS.get(number)
        if fn is None:
            print(f"  {number:5s} no builder")
            continue
        try:
            frame = fn(args.root, bank)
        except Missing as exc:
            print(f"  {number:5s} {fn.figure_name:14s} skipped — {exc}")
            continue
        path = args.out / f"fig-{number.replace('.', '-')}-{fn.figure_name}.csv"
        frame.to_csv(path, index=False)
        print(f"  {number:5s} {fn.figure_name:14s} {len(frame):6d} rows  -> {path.name}")
        manifest.append(
            {"figure": number, "name": fn.figure_name, "claim": fn.claim,
             "file": path.name, "rows": int(len(frame))}
        )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {"window_rule": {"baseline": BASELINE_WINDOW, "plateau_from": PLATEAU_FROM},
             "figures": manifest},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
