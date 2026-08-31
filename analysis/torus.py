"""Degree-two homology of the joint-input representation (§4.6, §6.5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis import cli
from analysis.bank import BASELINE_WINDOW
from grokking_tda.analysis.representations import dataset_for
from grokking_tda.artifacts.reader import Run, Snapshot
from grokking_tda.tda.betti import betti_profile
from grokking_tda.tda.pointcloud import _maxmin_landmarks

LANDMARKS = 300  # 500 is intractable on the ambient cloud
DRAWS = 3

PCA_DIMS = (0, 8, 4, 2)  # 0 is ambient; the rest are leading principal subspaces

# The two conditions §4.5 finds circular enough for the prediction to mean anything, plus
# the canonical regime, which generalises completely and is not circular
CONDITIONS = (
    "transformer_add113_f0.3_wd0.1_softmax_ce_s",
    "mlp_add97_f0.3_wd1.0_softmax_ce_s",
    "transformer_add97_f0.3_wd1.0_softmax_ce_s",
)

# "before" is the top of the baseline window the ratios are measured from
STAGES = {"before": BASELINE_WINDOW[1], "at": 1.0}


def principal_subspace(matrix: np.ndarray, dim: int) -> np.ndarray:
    """The cloud in its own leading ``dim`` principal directions (``dim = 0``: ambient)."""
    x = np.asarray(matrix, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if dim <= 0 or dim >= min(x.shape):
        return x
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :dim] * s[:dim]


def joint_representations(
    run: Run, snapshot: Snapshot, pca_dim: int
) -> dict[str, np.ndarray]:
    """Per-layer representations of every input pair; the projection is applied before pairing."""
    data = dataset_for(run)
    inputs = data.inputs
    model = run.rebuild_model(snapshot)

    names = [n for n, _ in model._named_hook_points()]
    wanted = [n for n in names if n.endswith(("hook_attn_out", "hook_mlp_out"))]
    wanted.append(model.hidden_hook)

    with torch.no_grad():
        logits, cache = model.run_with_cache(inputs, names=wanted)
        table = model.embedding_matrix().cpu().numpy()

    factor = principal_subspace(table, pca_dim)
    operands = inputs[:, :2].cpu().numpy()
    out = {"operands": np.concatenate([factor[operands[:, 0]], factor[operands[:, 1]]], axis=1)}

    for name in wanted:
        if name not in cache:
            continue
        tensor = cache[name]
        # Transformer hooks carry a position axis; the answer is at the last one.
        matrix = (tensor[:, -1, :] if tensor.dim() == 3 else tensor).cpu().numpy()
        out[name] = principal_subspace(matrix, pca_dim)
    out["logits"] = principal_subspace(logits.cpu().numpy(), pca_dim)
    return out


def profile_cloud(cloud: np.ndarray, *, landmarks: int, draws: int) -> list[dict]:
    x = np.ascontiguousarray(cloud, dtype=np.float64)
    if len(x) <= landmarks:
        return [{"draw": 0, **betti_profile(x, maxdim=2)}]
    return [
        {"draw": draw, **betti_profile(x[_maxmin_landmarks(x, landmarks, draw)], maxdim=2)}
        for draw in range(draws)
    ]


def stage_steps(run: Run, t_g: float | None) -> dict[str, int]:
    """The snapshot nearest each stage, over snapshots whose weights are on disk."""
    steps = np.array(
        [s.step for s in run.snapshots() if (s.directory / "weights.pt").exists()]
    )
    if steps.size == 0:
        return {}
    chosen = {"init": int(steps.min())}
    if t_g:
        for name, fraction in STAGES.items():
            chosen[name] = int(steps[np.abs(steps - fraction * t_g).argmin()])
    chosen["after"] = int(steps.max())
    return chosen


def analyse_run(run_dir: Path, *, landmarks: int, draws: int, dims) -> pd.DataFrame:
    run = Run(run_dir)
    summary = run_dir / "analysis" / "summary.json"
    t_g = json.loads(summary.read_text()).get("grokking_step") if summary.exists() else None

    rows = []
    for stage, step in stage_steps(run, t_g).items():
        snapshot = run.snapshot(step)
        for pca_dim in dims:
            reps = joint_representations(run, snapshot, pca_dim)
            for depth, (layer, cloud) in enumerate(reps.items()):
                for row in profile_cloud(cloud, landmarks=landmarks, draws=draws):
                    rows.append(
                        {
                            "run": run.run_name,
                            "model": run.config["model"]["name"],
                            "t_g": t_g,
                            "stage": stage,
                            "step": step,
                            "pca_dim": pca_dim,
                            "depth": depth,
                            "layer": layer,
                            **row,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    ap = cli.parser(__doc__)
    ap.add_argument("--landmarks", type=int, default=LANDMARKS)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--dims", type=int, nargs="*", default=list(PCA_DIMS))
    ap.add_argument("--runs", nargs="*")
    args = ap.parse_args()

    names = args.runs or [
        d.name
        for d in sorted(args.root.iterdir())
        if d.name.startswith(CONDITIONS) and (d / "snapshots" / "index.json").exists()
    ]
    if not names:
        raise SystemExit("no runs with stored weights for the reference regime or the MLP")

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "torus.csv"
    # Written per run: a full sweep is hours of ripser to lose to a late failure.
    written = False
    for name in names:
        df = analyse_run(
            args.root / name, landmarks=args.landmarks, draws=args.draws, dims=args.dims
        )
        df.to_csv(destination, mode="a" if written else "w", header=not written, index=False)
        written = True
        for (stage, dim, _), sub in df.groupby(["stage", "pca_dim", "depth"], sort=False):
            b1, b2 = sub.betti_1.to_numpy(), sub.betti_2.to_numpy()
            print(
                f"  {name:44s} {sub.layer.iloc[0]:16s} {stage:6s} d={dim:<3}"
                f"  beta1 {b1.min():.0f}-{b1.max():.0f}"
                f"  beta2 {b2.min():.0f}-{b2.max():.0f}"
                f"  life1 {sub.life_1.median():.3f}  life2 {sub.life_2.median():.3f}",
                flush=True,
            )

    print(f"\nwritten to {destination}  ({args.landmarks} landmarks, {args.draws} draws)")


if __name__ == "__main__":
    main()
