r"""When does a network acquire the operand symmetry the benchmark leaks through? (§3.5, §8.2)

§3.5 argues that the memorisation-phase plateau on modular addition is the transpose of a
memorised training pair, not partial generalisation, and that the leak needs two ingredients:
a commutative task, and an architecture that learns to ignore operand order. It asserts the
second on a timescale --- "within a few hundred steps, before memorisation is even complete"
--- without measuring it, and it notes that not every architecture does, without saying which.

The measure is how often the network gives $(a, b)$ and $(b, a)$ the same answer, whether or
not that answer is right --- corrected for chance, because raw agreement is degenerate exactly
where it matters. An untrained transformer predicts five of ninety-seven classes and so agrees
with itself on $99.6\%$ of pairs while having learnt nothing; the correction divides out the
agreement its own predicted-class distribution would produce by coincidence.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from analysis import cli
from analysis.bank import iter_runs
from grokking_tda.analysis.identity import is_replicate
from grokking_tda.analysis.representations import dataset_for
from grokking_tda.artifacts.reader import Run as ArtifactRun

AGREEMENT = 0.9  # the level of chance-corrected agreement at which the symmetry counts as acquired


def has_weights(run: ArtifactRun) -> bool:
    """Most of the bank was fetched as derived data and carries cached embeddings only.

    The measure needs a forward pass on transposed inputs, so it needs the weights; the
    fifteen runs that kept them are the two architectures under the canonical recipe and
    the reference regime, which is the comparison §3.5 is about.
    """
    snapshots = run.snapshots()
    return bool(snapshots) and (snapshots[0].directory / "weights.pt").exists()


def agreement_series(run: ArtifactRun) -> pd.DataFrame:
    """Per snapshot: how often the two operand orders receive the same prediction."""
    data = dataset_for(run)
    inputs = data.inputs
    transposed = inputs.clone()
    transposed[:, [0, 1]] = transposed[:, [1, 0]]
    novel = ~data.train_mask.cpu().numpy()

    rows = []
    for snapshot in run.snapshots():
        model = run.rebuild_model(snapshot)
        with torch.no_grad():
            straight = model(inputs).argmax(dim=-1).cpu().numpy()
            swapped = model(transposed).argmax(dim=-1).cpu().numpy()
        agree = straight == swapped
        rows.append(
            {
                "step": snapshot.step,
                "agreement": float(agree.mean()),
                "agreement_novel": float(agree[novel].mean()),
                "kappa": chance_corrected(agree.mean(), straight),
                "n_classes_used": int(np.unique(straight).size),
            }
        )
    return pd.DataFrame(rows)


def chance_corrected(observed: float, predictions: np.ndarray) -> float:
    """Agreement above what this network's own predicted-class distribution gives for free.

    Cohen's form: two draws from the marginal coincide with probability ``sum p^2``, which is
    near one for a network that has collapsed onto a handful of classes.
    """
    _, counts = np.unique(predictions, return_counts=True)
    expected = float(((counts / counts.sum()) ** 2).sum())
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def acquisition_step(series: pd.DataFrame, level: float = AGREEMENT) -> float | None:
    """The first snapshot at which agreement crosses ``level`` and stays there."""
    steps = series["step"].to_numpy(float)
    values = series["kappa"].to_numpy(float)
    above = values >= level
    if not above.any():
        return None
    # the last time it was below, plus one: a single early spike is not acquisition
    below = np.flatnonzero(~above)
    first = 0 if below.size == 0 else int(below[-1]) + 1
    return float(steps[first]) if first < steps.size else None


def memorisation_plateau(run_dir, t_c: float | None, t_g: float | None) -> float:
    """Median test accuracy while the training set is fit and the rule is not learnt.

    This is the quantity §3.5 attributes to the transpose leak, so it is the check on the
    symmetry measure: the two should appear together or not at all.
    """
    metrics = pd.read_json(run_dir / "metrics.jsonl", lines=True)
    if metrics.empty or "test_acc" not in metrics:
        return float("nan")
    start = t_c if t_c is not None else metrics.step.min()
    end = t_g if t_g is not None else metrics.step.max()
    window = metrics[(metrics.step > start) & (metrics.step < end)]
    return float(window.test_acc.median()) if len(window) else float("nan")


def first_crossing(series: pd.DataFrame, level: float = AGREEMENT) -> float | None:
    """The first snapshot at which the symmetry is there at all, whether or not it survives."""
    above = series[series["kappa"] >= level]
    return float(above["step"].iloc[0]) if len(above) else None


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--operation", default="add", help="the commutative task the leak needs")
    args = ap.parse_args()

    frames, summary, skipped = [], [], []
    for run in iter_runs(args.root):
        cfg = run.config
        if cfg["data"]["operation"] != args.operation or cfg["data"]["label_permutation"]:
            continue
        if is_replicate(run.name, cfg):
            continue
        raw = ArtifactRun(run.directory)
        if not has_weights(raw):
            skipped.append(run.name)
            continue
        series = agreement_series(raw)
        if series.empty:
            continue
        series.insert(0, "run", run.name)
        series.insert(1, "model", cfg["model"]["name"])
        frames.append(series)
        t_g = run.summary.get("grokking_step")
        t_c = run.summary.get("train_convergence_step")
        plateau = memorisation_plateau(run.directory, t_c, t_g)
        acquired = acquisition_step(series)
        first = first_crossing(series)
        summary.append(
            {
                "run": run.name,
                "model": cfg["model"]["name"],
                "weight_decay": cfg["train"]["optimizer"]["weight_decay"],
                "t_symmetry": acquired,
                "t_c": t_c,
                "t_g": t_g,
                "t_symmetry_first": first,
                "final_kappa": float(series["kappa"].iloc[-1]),
                "memorisation_plateau": plateau,
            }
        )

    if not frames:
        raise SystemExit(f"no {args.operation} runs under {args.root}")
    table = pd.concat(frames, ignore_index=True)
    runs = pd.DataFrame(summary)

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "symmetry.csv", index=False)
    runs.to_csv(args.out / "symmetry_runs.csv", index=False)

    by_model: dict[str, dict] = {}
    for model, sub in runs.groupby("model"):
        acquired = sub["t_symmetry"].dropna()
        by_model[model] = {
            "n_runs": int(len(sub)),
            "n_acquiring": int(acquired.size),
            "median_t_symmetry": float(acquired.median()) if acquired.size else None,
            "range_t_symmetry": [float(acquired.min()), float(acquired.max())]
            if acquired.size
            else None,
            "median_t_c": float(sub["t_c"].dropna().median()) if sub["t_c"].notna().any() else None,
            "median_t_g": float(sub["t_g"].dropna().median()) if sub["t_g"].notna().any() else None,
            "median_t_symmetry_first": float(sub["t_symmetry_first"].dropna().median())
            if sub["t_symmetry_first"].notna().any()
            else None,
            "median_final_kappa": float(sub["final_kappa"].median()),
            "median_memorisation_plateau": float(sub["memorisation_plateau"].median()),
        }
    (args.out / "symmetry.json").write_text(
        json.dumps(
            {"by_model": by_model, "runs_without_weights": len(skipped), "level": AGREEMENT},
            indent=2,
        )
    )

    print(f"{'model':12s} {'runs':>5} {'acquired':>9} {'t_sym':>10} {'t_c':>10} {'t_g':>10}")
    for model, s in by_model.items():
        print(
            f"{model:12s} {s['n_runs']:5d} {s['n_acquiring']:9d} "
            f"{_fmt(s['median_t_symmetry']):>10} {_fmt(s['median_t_c']):>10} "
            f"{_fmt(s['median_t_g']):>10}"
        )
    print(f"\n{len(skipped)} runs carry no per-snapshot weights and cannot be measured")
    print(f"written to {args.out}/symmetry.csv, symmetry_runs.csv and symmetry.json")


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}"


if __name__ == "__main__":
    main()
