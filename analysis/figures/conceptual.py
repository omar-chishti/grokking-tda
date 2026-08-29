"""Computed geometry for the conceptual TikZ figures.

The teaching figures in Chapter 2 draw a point cloud, a Vietoris--Rips complex and the
barcode of that complex. Drawing those by hand invites them to disagree: an octagon of
edges beside four bars of invented length says nothing true about either. So the geometry
is computed here from one small point set and emitted as TikZ coordinates, and the picture
file reads them. The composition stays authored; the mathematics does not.

The scale-problem figure in Chapter 3 has the same requirement for a different reason: it
claims that normalising by the connectivity scale removes a change of units and leaves a
change of shape, and a hand-drawn diagram could be made to say that whether or not it is
true.

Writes ``tikz-methodology-data.tex`` and ``tikz-scale-problem-data.tex`` into the figure
output root, which the picture files input. One command regenerates both.
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
SCALE_NAME = "tikz-scale-problem-data.tex"
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

# The scale-problem figure. Twelve points at 30 degrees, radially jittered so the ring does
# not read as a diagram of a polygon; SCALE_SHRINK is the factor between panel (a)'s two
# clouds and SCALE_BLUR the radial noise that makes panel (d)'s third cloud a worse circle
# at the same nominal radius. The seed is fixed by inspection: it is the one whose blurred
# cloud is still legibly a ring while its normalised lifetime falls by half.
SCALE_JITTER = (1.00, 0.94, 1.05, 0.97, 1.02, 0.92, 1.04, 0.99, 1.06, 0.95, 1.01, 0.96)
SCALE_SHRINK = 3.0
SCALE_BLUR = 0.40
SCALE_RADIUS_POINT = 2   # the ring point panel (a)'s radius line is drawn to
SCALE_SEED = 14


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
    # the panel is about the interval between fitting and generalising, so it starts where
    # that interval does. Included, the initialisation transient is the tallest thing on
    # the plate and the timing mark competes with it.
    run = run[run.step >= 0.05 * t_g]
    steps = run.step.to_numpy()
    grid = np.unique(np.geomspace(max(steps.min(), 1.0), steps.max(), n).astype(int))
    take = np.searchsorted(steps, grid).clip(0, len(steps) - 1)

    sub = run.iloc[take]
    # a rolling median on the schematic's copy only: at this size the raw series reads as
    # scribble, and the panel's job is the shape and the two marks. D4.2 draws it raw.
    h1 = (sub.h1_max_persistence_normalised
          .rolling(5, center=True, min_periods=1).median().to_numpy())
    # scaled against the plateau rather than the highest point in the window: normalising
    # by the maximum lets one late spike become the ceiling, after which the real plateau
    # reads as a decline — the opposite of what section 4.3 measures on the same run
    lo, hi = np.nanmin(h1), float(np.nanpercentile(h1, 80))
    # t_top: the midpoint crossing of the normalised series, as the detector defines it
    full = run.h1_max_persistence_normalised.to_numpy()
    mid = np.nanmin(full) + 0.5 * (np.nanmax(full) - np.nanmin(full))
    crossed = np.flatnonzero(full >= mid)
    t_top = float(steps[crossed[0]]) if crossed.size else float("nan")
    scaled = np.clip((h1 - lo) / (hi - lo), 0.0, 1.0)
    pre = sub.step.to_numpy() < t_g

    x = np.log10(np.maximum(sub.step.to_numpy(), 1.0))
    span = x.max() - x.min()
    return {
        "acc": list(zip((x - x.min()) / span, sub.test_acc.to_numpy(), strict=True)),
        "h1": list(zip((x - x.min()) / span, scaled, strict=True)),
        # where each series sits before the transition, which is where they are far enough
        # apart to be named without a leader
        "acc_pre": float(np.median(sub.test_acc.to_numpy()[pre])),
        "h1_pre": float(np.median(scaled[pre])),
        "t_g": (np.log10(t_g) - x.min()) / span,
        "t_top": (np.log10(t_top) - x.min()) / span,
        "t_g_steps": t_g,
        "t_top_steps": t_top,
    }


# --- the scale problem ------------------------------------------------------------------


def scale_ring(radius: float = 1.0, blur: float = 0.0) -> np.ndarray:
    """Twelve points on a ring of the given radius, optionally blurred outward and in."""
    angle = np.arange(12) * np.pi / 6
    jitter = np.array(SCALE_JITTER)
    if blur:
        jitter = jitter * (1 + np.random.default_rng(SCALE_SEED).uniform(-blur, blur, 12))
    return np.c_[radius * jitter * np.cos(angle), radius * jitter * np.sin(angle)]


def scale_summary(points: np.ndarray) -> dict:
    """The connectivity scale, the dominant cycle, and that cycle in units of the scale."""
    scale = float(h0_deaths(points).max())
    birth, death = h1_interval(points)
    return {"s": scale, "b": birth, "d": death,
            "bn": birth / scale, "dn": death / scale, "ln": (death - birth) / scale}


def emit_scale() -> str:
    clouds = {
        "Large": scale_ring(),
        "Small": scale_ring(1 / SCALE_SHRINK),
        "Blur": scale_ring(blur=SCALE_BLUR),
    }
    m = {k: scale_summary(v) for k, v in clouds.items()}

    # the figure's three claims, checked rather than hoped for: scaling is exactly a change
    # of units, normalisation removes it, and it does not remove a change of shape
    assert np.isclose(m["Large"]["s"] / m["Small"]["s"], SCALE_SHRINK), "not a pure rescaling"
    assert np.isclose(m["Large"]["ln"], m["Small"]["ln"]), "normalisation is not invariant"
    assert m["Blur"]["ln"] < 0.6 * m["Large"]["ln"], "the blurred ring is not visibly worse"

    lines = [
        "% Generated by analysis/figures/conceptual.py -- do not edit by hand.",
        "% One point set at two radii, and a third that is a worse circle at the first",
        "% radius. Every number panels (b)-(d) print is computed from these three clouds.",
        "",
    ]
    for name, cloud in clouds.items():
        coords = ", ".join(f"{x:.4f}/{y:.4f}" for x, y in cloud)
        lines.append(f"\\def\\Scale{name}Points{{{coords}}}")
    lines.append("")
    # the point panel (a)'s radius line is drawn to, emitted rather than guessed at: a ray
    # at a hand-chosen angle lands between two points of a ring whose radii are perturbed,
    # and reads as a line that misses
    mark = SCALE_RADIUS_POINT
    for name in ("Large", "Small"):
        x, y = clouds[name][mark]
        lines.append(f"\\def\\Scale{name}RadiusX{{{x:.4f}}}")
        lines.append(f"\\def\\Scale{name}RadiusY{{{y:.4f}}}")
    lines.append("")
    for name, v in m.items():
        for key, fmt in (("s", ".4f"), ("b", ".4f"), ("d", ".4f"),
                         ("bn", ".4f"), ("dn", ".4f"), ("ln", ".2f")):
            lines.append(f"\\def\\Scale{name}{key.upper()}{{{v[key]:{fmt}}}}")
        lines.append(f"\\def\\Scale{name}L{{{v['d'] - v['b']:.2f}}}")
        lines.append("")
    lines.append(f"\\def\\ScaleShrink{{{SCALE_SHRINK:.0f}}}")
    return "\n".join(lines) + "\n"



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
        f"\\def\\MethAccPre{{{sig['acc_pre']:.4f}}}",
        f"\\def\\MethHOnePre{{{sig['h1_pre']:.4f}}}",
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
    for name, body in ((NAME, emit()), (SCALE_NAME, emit_scale())):
        (args.out / name).write_text(body)
        print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
