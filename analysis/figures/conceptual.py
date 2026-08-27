"""Computed geometry for the conceptual TikZ figures.

The teaching figures in Chapter 2 draw a point cloud, a Vietoris--Rips complex and the
barcode of that complex. Drawing those by hand invites them to disagree: an octagon of
edges beside four bars of invented length says nothing true about either. So the geometry
is computed here from one small point set and emitted as TikZ coordinates, and the picture
file reads them. The composition stays authored; the mathematics does not.

Writes ``tikz-methodology-data.tex`` into the figure output root, which the picture
file inputs. One command regenerates it.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from .style import OUTPUT_ROOT

NAME = "tikz-methodology-data.tex"
SIGNATURE = Path("results/processed/thesis/figures/fig-4-2-signature.csv")
RUN = "transformer_add113_f0.3_wd0.1_softmax_ce_s0"

N_POINTS = 18
SEED = 11
# As fractions of the H1 birth--death interval. The first is early enough that only first-
# and second-neighbour chords exist, which leaves an annulus of triangles around a visible
# hole -- a denser complex is correct and unreadable. The second is past 1.0 deliberately:
# the commonest misreading of a barcode is that persistence measures only birth, so the
# panel has to show the cycle die.
RADIUS_SMALL, RADIUS_LARGE = 0.27, 1.09


def ring(n: int = N_POINTS, seed: int = SEED) -> np.ndarray:
    """An irregular ring. Irregular because a perfect polygon reads as a diagram of one."""
    rng = np.random.default_rng(seed)
    angle = np.sort(rng.uniform(0, 2 * np.pi, n))
    # nudge the angles apart so no two points collide, then jitter the radius
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.35 * (angle - angle.mean()) / n
    radius = 1.0 + rng.uniform(-0.085, 0.085, n)
    return np.c_[radius * np.cos(angle), radius * np.sin(angle)]


def complex_at(points: np.ndarray, eps: float) -> tuple[list, list]:
    """The Rips edges and triangles at ``eps``: every close pair, every mutually close triple."""
    d = squareform(pdist(points))
    edges = [(i, j) for i, j in combinations(range(len(points)), 2) if d[i, j] <= eps]
    close = {frozenset(e) for e in edges}
    triangles = [
        (i, j, k)
        for i, j, k in combinations(range(len(points)), 3)
        if all(frozenset(p) in close for p in ((i, j), (i, k), (j, k)))
    ]
    return edges, triangles


def h0_deaths(points: np.ndarray) -> np.ndarray:
    """When each component merges. For a Rips filtration these are the MST's edge lengths."""
    tree = minimum_spanning_tree(squareform(pdist(points))).toarray()
    return np.sort(tree[tree > 0])


def h1_interval(points: np.ndarray) -> tuple[float, float]:
    """The birth and death of the longest-lived one-cycle."""
    bars = ripser(points, maxdim=1)["dgms"][1]
    return tuple(bars[np.argmax(bars[:, 1] - bars[:, 0])])


def signature(n: int = 46) -> dict:
    """The measured series panel 4 draws, decimated onto a log-spaced grid."""
    d = pd.read_csv(SIGNATURE)
    run = d[d.run == RUN].sort_values("step").dropna(subset=["test_acc"])
    t_g = float(run.t_g.dropna().iloc[0])
    steps = run.step.to_numpy()
    grid = np.unique(np.geomspace(max(steps.min(), 1.0), steps.max(), n).astype(int))
    take = np.searchsorted(steps, grid).clip(0, len(steps) - 1)

    sub = run.iloc[take]
    # a rolling median on the schematic's copy only: at this size the raw series reads as
    # scribble, and the panel's job is the shape and the two marks. D4.2 draws it raw.
    h1 = (sub.h1_max_persistence_normalised
          .rolling(5, center=True, min_periods=1).median().to_numpy())
    lo, hi = np.nanmin(h1), np.nanmax(h1)
    # t_top: the midpoint crossing of the normalised series, as the detector defines it
    full = run.h1_max_persistence_normalised.to_numpy()
    mid = np.nanmin(full) + 0.5 * (np.nanmax(full) - np.nanmin(full))
    crossed = np.flatnonzero(full >= mid)
    t_top = float(steps[crossed[0]]) if crossed.size else float("nan")

    x = np.log10(np.maximum(sub.step.to_numpy(), 1.0))
    span = x.max() - x.min()
    return {
        "acc": list(zip((x - x.min()) / span, sub.test_acc.to_numpy(), strict=True)),
        "h1": list(zip((x - x.min()) / span, (h1 - lo) / (hi - lo), strict=True)),
        "t_g": (np.log10(t_g) - x.min()) / span,
        "t_top": (np.log10(t_top) - x.min()) / span,
        "t_g_steps": t_g,
        "t_top_steps": t_top,
    }


def _path(pairs, sx: float, sy: float, x0: float, y0: float) -> str:
    return " -- ".join(f"({x0 + sx * u:.4f},{y0 + sy * v:.4f})" for u, v in pairs)


def emit() -> str:
    points = ring()
    birth, death = h1_interval(points)
    eps = {"a": birth + RADIUS_SMALL * (death - birth),
           "b": birth + RADIUS_LARGE * (death - birth)}
    edges_a, tris_a = complex_at(points, eps["a"])
    edges_b, tris_b = complex_at(points, eps["b"])
    only_b = sorted(set(edges_b) - set(edges_a))
    deaths = h0_deaths(points)
    sig = signature()

    # the figure asserts three things about these points; check them rather than hope
    ring_edges = {frozenset((i, (i + 1) % N_POINTS)) for i in range(N_POINTS)}
    assert ring_edges <= {frozenset(e) for e in edges_a}, "the ring is not closed at eps_a"
    assert birth <= eps["a"] < death, "eps_a does not sit inside the cycle's lifetime"
    assert eps["b"] > death, "eps_b does not outlive the cycle"

    # the barcode axis runs 0 -> axis_max in cloud units, mapped to 0 -> 1 in the picture
    axis_max = death * 1.16
    scale = 1.0 / axis_max

    # the pair whose discs demonstrate the rule: close enough that the lens is unmissable,
    # far enough that it is not a coincidence of two points sitting on top of each other
    d = squareform(pdist(points))
    pairs = [(i, j) for i, j in combinations(range(len(points)), 2)]
    lens_i, lens_j = min(pairs, key=lambda e: abs(d[e] - 0.50 * eps["a"]))

    lines = [
        "% Generated by analysis/figures/conceptual.py -- do not edit by hand.",
        "% The point set, its Rips complex and its barcode are one computation, so the",
        "% mesh in panel 2 and the bars in panel 3 describe the same eighteen points.",
        "",
        f"\\def\\MethN{{{N_POINTS}}}",
        f"\\def\\MethEpsA{{{eps['a'] * scale:.4f}}}",
        f"\\def\\MethEpsB{{{eps['b'] * scale:.4f}}}",
        f"\\def\\MethBirth{{{birth * scale:.4f}}}",
        f"\\def\\MethDeath{{{death * scale:.4f}}}",
        f"\\def\\MethEpsARaw{{{eps['a']:.4f}}}",
        f"\\def\\MethEpsBRaw{{{eps['b']:.4f}}}",
        f"\\def\\MethLensI{{{lens_i}}}",
        f"\\def\\MethLensJ{{{lens_j}}}",
        f"\\def\\MethTg{{{sig['t_g']:.4f}}}",
        f"\\def\\MethTtop{{{sig['t_top']:.4f}}}",
        f"\\def\\MethTgSteps{{{sig['t_g_steps']:,.0f}}}",
        f"\\def\\MethTtopSteps{{{sig['t_top_steps']:,.0f}}}",
        "",
    ]

    coords = ", ".join(f"{i}/{x:.4f}/{y:.4f}" for i, (x, y) in enumerate(points))
    lines += [f"\\def\\MethPoints{{{coords}}}", ""]
    for name, items in (("EdgesA", edges_a), ("EdgesOnlyB", only_b)):
        lines.append(f"\\def\\Meth{name}{{{', '.join(f'{i}/{j}' for i, j in items)}}}")
    for name, items in (("TrisA", tris_a), ("TrisB", tris_b)):
        lines.append(f"\\def\\Meth{name}{{{', '.join(f'{i}/{j}/{k}' for i, j, k in items)}}}")
    lines.append("")

    bars = ", ".join(f"{i}/{d * scale:.4f}" for i, d in enumerate(deaths))
    # TeX macro names cannot contain digits, so the H0 list is not \MethH0
    lines += [f"\\def\\MethHZero{{{bars}}}", ""]

    # panel 4's two curves, already normalised to the unit square
    lines += [
        f"\\def\\MethAcc{{{_path(sig['acc'], 1.0, 1.0, 0.0, 0.0)}}}",
        f"\\def\\MethHOne{{{_path(sig['h1'], 1.0, 1.0, 0.0, 0.0)}}}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / NAME
    path.write_text(emit())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
