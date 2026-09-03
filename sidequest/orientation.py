"""Does one model reuse one circle for `a + b` and `a - b`, or build two? (R24)

Persistence cannot answer it: the diagrams are computed over Z/2, where -1 = +1, and a rotation
and a reflection of a circle have identical barcodes anyway. The distinction lives in the induced
map on H_1, whose degree is +1 for a rotation and -1 for a reflection.

So this measures the degree, twice over, and the two measurements answer different halves:

  * the winding of each operator's loop, in **one plane fitted to both**. The sign of a principal
    axis is arbitrary, so a single winding means nothing; the two windings **against each other**
    are basis-free, and w_sub = -w_add says the operators traverse one curve opposite ways.
  * the principal angles between each operator's **own** plane, which says whether there is one
    circle to traverse or two.
  * the best cyclic alignment of one loop onto the other, with and without reversal. Winding is a
    summary; this asks the claim directly, since a - b = a + (-b) makes the subtraction loop the
    addition loop re-indexed by b -> -b, point for point.

Design and outcomes: Documentation/SideQuest_Orientation_2026-09-03.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import subspace_angles

from grokking_tda.analysis.representations import dataset_for
from grokking_tda.artifacts import Run
from grokking_tda.data.multiop import operators_of

N_ANCHORS = 16  # how many a_0 the loops are traced from; reported as a distribution


def _plane(x: np.ndarray) -> np.ndarray:
    """An orthonormal basis for the top-2 principal plane of ``x`` (rows are points)."""
    centred = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:2]


def _turning(loop: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """The signed angular step between consecutive points, closing the loop."""
    xy = (loop - loop.mean(axis=0, keepdims=True)) @ basis.T
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    steps = np.diff(np.concatenate([angles, angles[:1]]))
    return (steps + np.pi) % (2 * np.pi) - np.pi  # the short way round, every step


def winding(loop: np.ndarray, basis: np.ndarray) -> float:
    """Turns accumulated by a closed loop projected into ``basis``, signed.

    Always an integer, and that is the trap: a closed circuit of *noise* also winds, typically by
    one. The winding says which way something was traversed and never whether the something is a
    circle — ``monotonicity`` is what says that, and it gates every reading of this.
    """
    return float(_turning(loop, basis).sum() / (2 * np.pi))


def monotonicity(loop: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of steps turning the same way as the loop overall: 1 a traversal, ~0.5 noise.

    A circle read in order advances monotonically in angle. Random points close into a circuit
    that winds just as surely, but their steps alternate in sign, so this separates them where the
    winding cannot.
    """
    steps = _turning(loop, basis)
    total = steps.sum()
    if total == 0.0:
        return 0.0
    return float((np.sign(steps) == np.sign(total)).mean())


def planarity(loop: np.ndarray) -> float:
    """Fraction of the loop's variance lying in its own top-2 plane."""
    centred = loop - loop.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centred, compute_uv=False) ** 2
    return float(spectrum[:2].sum() / (spectrum.sum() + 1e-12))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine of the principal angles between two planes: 1 shared, 0 orthogonal."""
    return float(np.cos(subspace_angles(a.T, b.T)).mean())


def antisymmetry(forward: np.ndarray, backward: np.ndarray) -> float:
    """Correlation of two operators' windings across anchors: -1 one curve read both ways.

    The product of two windings was the first statistic here and it is a poor one. A loop at
    Fourier frequency k winds k times, so the product is -k^2 and reads as -256 where the degree
    is -1; worse, it varies with k across anchors and hides under its own scale. The correlation
    is the same claim at unit scale.
    """
    return float(np.corrcoef(forward, backward)[0, 1])


def residual_profile(target: np.ndarray, source: np.ndarray, *, reverse: bool) -> np.ndarray:
    """Relative residual of ``target`` against every cyclic re-indexing of ``source``.

    ``reverse`` reads the source backwards -- the map b -> -b. Both families are searched over all
    p shifts, because the two loops start at whatever offset the operator token imposes and only
    their shape is at issue. Two uncorrelated centred loops of equal norm sit at sqrt(2).
    """
    p = len(target)
    t = target - target.mean(axis=0, keepdims=True)
    s = source - source.mean(axis=0, keepdims=True)
    index = np.arange(p)
    scale = np.linalg.norm(t) + 1e-12
    return np.array([
        np.linalg.norm(t - s[(k - index) % p if reverse else (k + index) % p]) / scale
        for k in range(p)
    ])


def alignment(target: np.ndarray, source: np.ndarray, *, reverse: bool) -> tuple[float, int]:
    """The best cyclic re-indexing of ``source`` onto ``target``, and its residual."""
    residuals = residual_profile(target, source, reverse=reverse)
    best = int(np.argmin(residuals))
    return float(residuals[best]), best


def _hidden(model, tokens: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names=[model.hidden_hook])
    return cache[model.hidden_hook].cpu().numpy()


def loops(run: Run, snapshot, anchors: np.ndarray) -> dict[str, np.ndarray]:
    """``{operator: (n_anchors, p, d)}`` — the hidden state as ``b`` runs the residues."""
    model = run.rebuild_model(snapshot)
    meta = run.task_meta
    p, ops = int(meta["modulus"]), operators_of(meta["operation"])
    equals = int(meta["equals_token"])
    b = torch.arange(p, dtype=torch.long)
    out = {}
    for i, name in enumerate(ops):
        traced = []
        for a0 in anchors:
            tokens = torch.stack(
                [torch.full((p,), int(a0)), torch.full((p,), p + i), b, torch.full((p,), equals)],
                dim=1,
            )
            traced.append(_hidden(model, tokens))
        out[name] = np.stack(traced)
    return out


def leak_by_operator(run: Run) -> pd.DataFrame:
    """Held-out accuracy per operator, per snapshot.

    `(a,+,b)` shares a label with `(b,+,a)` and `(a,-,b)` does not, so thesis 3.5 predicts a
    memorisation plateau at the train fraction on one and at chance on the other, inside one run.
    """
    data = dataset_for(run)
    meta = run.task_meta
    p, ops = int(meta["modulus"]), operators_of(meta["operation"])
    x, y = data.test_inputs, data.test_targets
    rows = []
    for snapshot in run.snapshots():
        model = run.rebuild_model(snapshot)
        with torch.no_grad():
            predicted = model(x).argmax(dim=-1)
        for i, name in enumerate(ops):
            sel = x[:, 1] == p + i
            rows.append({
                "run": run.run_name,
                "step": snapshot.step,
                "operator": name,
                "n": int(sel.sum()),
                "test_acc": float((predicted[sel] == y[sel]).float().mean()),
            })
    return pd.DataFrame(rows)


def measure(run: Run, snapshot=None) -> tuple[pd.DataFrame, dict]:
    snapshot = snapshot or run.snapshots()[-1]
    meta = run.task_meta
    p, ops = int(meta["modulus"]), operators_of(meta["operation"])
    if len(ops) != 2:
        raise ValueError(f"{run.run_name}: expected two operators, got {ops}")

    rng = np.random.default_rng(0)
    anchors = rng.choice(p, size=min(N_ANCHORS, p), replace=False)
    traced = loops(run, snapshot, anchors)
    first, second = ops

    rows = []
    for k, a0 in enumerate(anchors):
        lo_a, lo_b = traced[first][k], traced[second][k]
        shared = _plane(np.vstack([lo_a - lo_a.mean(0), lo_b - lo_b.mean(0)]))
        w_a, w_b = winding(lo_a, shared), winding(lo_b, shared)
        reflected, shift = alignment(lo_b, lo_a, reverse=True)
        rotated, _ = alignment(lo_b, lo_a, reverse=False)
        rows.append({
            "run": run.run_name,
            "a0": int(a0),
            f"winding_{first}": w_a,
            f"winding_{second}": w_b,
            "product": w_a * w_b,
            "overlap": overlap(_plane(lo_a), _plane(lo_b)),
            "residual_reflected": reflected,
            "residual_rotated": rotated,
            "reflection_shift": shift,
            # whether the wound thing is a circle. It is not: the hidden state carries several
            # Fourier frequencies at once, so its projection is not convex and its angle about
            # the centroid is not monotone. The reversal survives that; the null is the guard.
            "monotonicity": 0.5 * (monotonicity(lo_a, shared) + monotonicity(lo_b, shared)),
            "planarity": 0.5 * (planarity(lo_a) + planarity(lo_b)),
        })
    table = pd.DataFrame(rows)

    embedding = snapshot.representation("embedding")
    operator_rows = run.rebuild_model(snapshot).embed.weight.detach().cpu().numpy()[p : p + 2]
    plane = _plane(embedding)
    difference = operator_rows[1] - operator_rows[0]
    summary = {
        "run": run.run_name,
        "model": run.config["model"]["name"],
        "modulus": p,
        "operators": list(ops),
        "step": int(snapshot.step),
        "n_anchors": int(len(anchors)),
        # the gate on everything below: noise winds too, so a low monotonicity voids the reading
        "monotonicity_median": float(table["monotonicity"].median()),
        "planarity_median": float(table["planarity"].median()),
        f"winding_{first}_median": float(table[f"winding_{first}"].median()),
        f"winding_{second}_median": float(table[f"winding_{second}"].median()),
        "product_median": float(table["product"].median()),
        "fraction_opposite": float((table["product"] < 0).mean()),
        # the headline: one curve read both ways, at unit scale and against the null
        "winding_antisymmetry": antisymmetry(
            table[f"winding_{first}"].values, table[f"winding_{second}"].values
        ),
        "fraction_reversed": float(
            (table[f"winding_{first}"] + table[f"winding_{second}"]).abs().le(1).mean()
        ),
        "residual_reflected_median": float(table["residual_reflected"].median()),
        "residual_rotated_median": float(table["residual_rotated"].median()),
        "overlap_median": float(table["overlap"].median()),
        # does the operator act in the plane the circle lives in?
        "operator_difference_in_plane": float(
            np.linalg.norm(plane @ difference) / (np.linalg.norm(difference) + 1e-12)
        ),
    }
    return table, summary


def trace(run: Run, every: int) -> pd.DataFrame:
    """The two headline statistics at every ``every``-th snapshot.

    The terminal reading says the operators share a circle; this says *when* they come to share it,
    which is the only question here that touches grokking rather than the trained solution.
    """
    snapshots = run.snapshots()[::every]
    rows = []
    for snapshot in snapshots:
        _, summary = measure(run, snapshot)
        rows.append({k: summary[k] for k in
                     ("run", "step", "winding_antisymmetry", "overlap_median",
                      "residual_reflected_median", "residual_rotated_median")})
    return pd.DataFrame(rows)


def export(run: Run) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two loops as drawn, and the search landscape behind §9.4 — the geometry the figures need.

    ``measure`` keeps only the minimum of each search; a figure of a search has to show what was
    refused as well as what was chosen.
    """
    snapshot = run.snapshots()[-1]
    meta = run.task_meta
    modulus, ops = int(meta["modulus"]), operators_of(meta["operation"])
    rng = np.random.default_rng(0)
    anchors = rng.choice(modulus, size=min(N_ANCHORS, modulus), replace=False)
    traced = loops(run, snapshot, anchors)
    first, second = ops

    drawn, searched = [], []
    for k, a0 in enumerate(anchors):
        lo_a, lo_b = traced[first][k], traced[second][k]
        basis = _plane(np.vstack([lo_a - lo_a.mean(0), lo_b - lo_b.mean(0)]))
        for name, loop in ((first, lo_a), (second, lo_b)):
            xy = (loop - loop.mean(axis=0, keepdims=True)) @ basis.T
            drawn.append(pd.DataFrame({
                "run": run.run_name, "a0": int(a0), "operator": name,
                "b": np.arange(modulus), "x": xy[:, 0], "y": xy[:, 1],
            }))
        for family, reverse in (("reflected", True), ("rotated", False)):
            searched.append(pd.DataFrame({
                "run": run.run_name, "a0": int(a0), "family": family,
                "shift": np.arange(modulus),
                "residual": residual_profile(lo_b, lo_a, reverse=reverse),
            }))
    return pd.concat(drawn, ignore_index=True), pd.concat(searched, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("results-sidequest/raw"))
    ap.add_argument("--out", type=Path, default=Path("results-sidequest/processed"))
    ap.add_argument("--skip-leak", action="store_true", help="windings only; the leak sweep is "
                                                            "one forward pass per snapshot")
    ap.add_argument("--trace", type=int, metavar="EVERY",
                    help="also measure at every EVERY-th snapshot, to date the reflection")
    ap.add_argument("--only", help="restrict to runs whose name contains this")
    ap.add_argument("--export", action="store_true",
                    help="also write the projected loops and the shift landscapes, for the figures")
    args = ap.parse_args()

    runs = [Run(d) for d in sorted(args.root.iterdir()) if (d / "manifest.json").exists()]
    if not runs:
        raise SystemExit(f"no runs under {args.root}")

    tables, summaries, leaks, traces = [], [], [], []
    drawn, searched = [], []
    for run in runs:
        if run.config["data"].get("task") != "modular_multiop":
            continue
        if args.only and args.only not in run.run_name:
            continue
        table, summary = measure(run)
        tables.append(table)
        summaries.append(summary)
        if not args.skip_leak:
            leaks.append(leak_by_operator(run))
        if args.trace:
            traces.append(trace(run, args.trace))
        if args.export:
            a, b = export(run)
            drawn.append(a)
            searched.append(b)
        print(f"  {run.run_name:52s} antisym {summary['winding_antisymmetry']:+.3f}  "
              f"overlap {summary['overlap_median']:.2f}  "
              f"reflected {summary['residual_reflected_median']:.2f}  "
              f"rotated {summary['residual_rotated_median']:.2f}")

    if not summaries:
        raise SystemExit(f"no modular_multiop runs under {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(args.out / "orientation.csv", index=False)
    (args.out / "orientation.json").write_text(json.dumps(summaries, indent=2))
    if leaks:
        pd.concat(leaks, ignore_index=True).to_csv(args.out / "leak_by_operator.csv", index=False)
    if traces:
        pd.concat(traces, ignore_index=True).to_csv(args.out / "orientation_trace.csv", index=False)
    if drawn:
        pd.concat(drawn, ignore_index=True).to_csv(args.out / "loops.csv", index=False)
        pd.concat(searched, ignore_index=True).to_csv(args.out / "shift_landscape.csv", index=False)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
