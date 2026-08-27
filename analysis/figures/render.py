"""Render the generated thesis figures from the tidy CSVs.

One function per figure, each drawing one stated claim. Reads only
``results/processed/thesis/figures/`` — never the artefact store directly — so a panel can
be checked as a table before it is drawn.

Usage (from ``Code/``)::

    uv run python -m analysis.figures.render                 # everything buildable
    uv run python -m analysis.figures.render --only 4.2
    uv run python -m analysis.figures.render --variant slide
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm

from analysis.figures import style as S

DATA = Path("results/processed/thesis/figures")
RENDERERS: dict[str, Callable] = {}


def renders(number: str, name: str):
    def register(fn):
        fn.number, fn.figure_name = number, name
        RENDERERS[number] = fn
        return fn

    return register


def _load(stem: str) -> pd.DataFrame:
    path = DATA / stem
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _condition_label(key) -> str:
    model, op, mod, frac, wd, loss, opt = key
    name = {"add": "$a+b$", "sub": "$a-b$", "mul": r"$a \times b$", "div": r"$a \div b$",
            "compose": r"$S_5$"}.get(op, op)
    arch = "MLP" if model == "mlp" else "tf"
    extra = "" if loss == "softmax_ce" else " smax"
    extra += "" if opt == "adamw" else r" $\perp$G"
    return f"{arch}  {name}  $p${int(mod)}  $f${frac:g}  wd{wd:g}{extra}"


def _claims() -> dict:
    """The ledger of quoted numbers, so a figure and the prose cannot disagree."""
    return json.loads((DATA.parent / "claims.json").read_text())


def _condition_row(**conditions):
    """One row of the robustness table, so a figure and the table cannot drift apart."""
    frame = _load("fig-4-4-robustness.csv")
    for key, value in conditions.items():
        frame = frame[frame[key] == value]
    row = frame.iloc[0]
    return SimpleNamespace(
        lo=float(row["h1_max_persistence_normalised__lo"]),
        med=float(row["h1_max_persistence_normalised__med"]),
        hi=float(row["h1_max_persistence_normalised__hi"]),
        verdict=row["h1_max_persistence_normalised__verdict"],
        circularity=float(row["circularity"]),
        t_g=float(row["t_g_median"]),
    )


def _logx(ax, lo: float = 1.0) -> None:
    ax.set_xscale("symlog", linthresh=lo)


# ---------------------------------------------------------------------------- D1.1


@renders("1.1", "hero")
def hero(variant: S.Variant) -> str:
    """Grokking, in one run. Spec: D1-1-hero.md.

    The emptiest figure in the thesis, deliberately: the reader does not yet know what
    grokking is, and every mark that is not the phenomenon competes with it.
    """
    S.use(variant)
    d = _load("fig-1-1-hero.csv")
    t_c, t_g = float(d.t_c.dropna().iloc[0]), float(d.t_g.dropna().iloc[0])
    acc = d.dropna(subset=["test_acc"]).sort_values("step")
    novel = d.dropna(subset=["test_acc_novel"]).sort_values("step")

    fig = S.figure(S.FULL, 0.545, variant)
    ax = fig.add_axes([0.062, 0.132, 0.925, 0.836])

    ax.axvspan(t_c, t_g, facecolor=S.INK, alpha=0.04, lw=0, zorder=0)
    ax.axvline(t_c, color=S.RULE, lw=S.HAIRLINE, zorder=1)
    ax.axvline(t_g, color=S.SIENNA, lw=S.HAIRLINE, zorder=1)
    ax.plot(acc.step, acc.train_acc, color=S.RULE, lw=S.SECONDARY, zorder=2)
    ax.plot(novel.step, novel.test_acc_novel, color=S.INK, lw=0.9, ls=(0, (4, 2)), zorder=3)
    ax.plot(acc.step, acc.test_acc, color=S.INK, lw=1.15, zorder=4)

    _logx(ax, 100)
    ax.set_xlim(0, acc.step.max())
    ax.set_ylim(-0.03, 1.06)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    S.range_frame(ax, y=(0, 1))

    # named on the plateau, where the three series are furthest apart, rather than at the
    # right-hand edge, where all three have converged on one and a label column would be
    # needed to tell them apart
    S.direct_label(ax, 1150, 1.00, "train accuracy", S.BRONZE, dy=-4, va="top")
    S.direct_label(ax, 1150, 0.305, "test accuracy", S.INK, dy=4, va="bottom")
    S.direct_label(ax, 1150, 0.010, "novel pairs only", S.INK, dy=4, va="bottom")

    trans = ax.get_xaxis_transform()
    ax.text(t_c * 1.30, 0.048, r"$t_c$ = 200", transform=trans, ha="left", va="bottom",
            color=S.INK, fontsize=7.0 * variant.scale)
    ax.text(t_g * 0.62, 0.048, r"$t_g$ = 28,600", transform=trans, ha="right", va="bottom",
            color=S.SIENNA, fontsize=7.0 * variant.scale, zorder=6)

    # the span the thesis is about, measured on the drawing rather than asserted beside it.
    # The multiple sits in a break in its own rule, so neither has to dodge the other.
    y = 0.685
    ax.annotate("", xy=(t_c, y), xytext=(t_g, y), xycoords=trans, textcoords=trans,
                arrowprops=dict(arrowstyle="<->", color=S.BRONZE, lw=S.HAIRLINE,
                                shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(t_c * 3.0, y + 0.024, r"$143\times$", transform=trans, ha="center",
            va="bottom", color=S.BRONZE, style="italic", fontsize=7.6 * variant.scale)

    return S.save(fig, "fig-1-1-hero", variant)


# ---------------------------------------------------------------------------- D4.1


@renders("4.1", "reproduction")
def reproduction(variant: S.Variant) -> str:
    """It reproduces, three ways, five seeds each. Spec: D4-1-reproduction.md.

    The seed comb is the design idea: it turns "five of five grokked" into a picture and
    shows the spread in transition time without spending a second panel on it.
    """
    S.use(variant)
    d = _load("fig-4-1-reproduction.csv")
    panels = ["canonical transformer", "reference transformer", "canonical MLP"]
    titles = {
        "canonical transformer": r"transformer   $p=97$, wd $1.0$",
        "reference transformer": r"transformer   $p=113$, wd $0.1$",
        "canonical MLP": r"MLP   $p=97$, wd $1.0$",
    }

    fig = S.figure(S.FULL, 0.425, variant)
    gs = fig.add_gridspec(1, 3, left=0.068, right=0.992, bottom=0.168, top=0.868, wspace=0.095)

    for ci, panel in enumerate(panels):
        ax = fig.add_subplot(gs[0, ci])
        sub = d[d.panel == panel]
        # one entry per run, not per distinct value: two seeds may share a grokking step
        tg = sorted(sub.groupby("run").t_g.first().dropna())
        end = sub.step.max()

        for _, g in sub.groupby("run"):
            g = g.sort_values("step")
            ax.plot(g.step, g.train_acc, color=S.RULE, lw=0.45, zorder=2)
            ax.plot(g.step, g.test_acc, color=S.INK, lw=0.55, alpha=0.55, zorder=3)
        median = sub.groupby("step").test_acc.median()
        ax.plot(median.index, median.values, color=S.INK, lw=1.15, zorder=5)

        _logx(ax, 100)
        ax.set_xlim(0, end)
        ax.set_ylim(-0.03, 1.06)
        ax.set_yticks([0, 0.5, 1.0])
        S.seed_comb(ax, tg, height=0.075)
        S.range_frame(ax, y=(0, 1))
        S.panel_title(ax, titles[panel], pad=7)
        ax.set_xlabel("training step")
        if ci == 0:
            ax.set_yticklabels(["0", "0.5", "1"])
            ax.set_ylabel("accuracy")
            S.direct_label(ax, 320, 0.900, "train", S.BRONZE, dx=0, size=6.8 * variant.scale)
            S.direct_label(ax, 320, 0.400, "test", S.INK, dx=0, size=6.8 * variant.scale)
        else:
            ax.set_yticklabels([])

        # the comb, read: how many seeds grokked, and — set on the comb itself, since that
        # is the thing it describes — over what span
        S.value(ax, 0.030, 0.930, f"{len(tg)}/{sub.run.nunique()}", "seeds grokked",
                colour=S.INK)
        S.direct_label(ax, max(tg) * 1.20, 0.030,
                       f"{min(tg) / 1000:.1f}\u2013{max(tg) / 1000:.1f}k", S.SIENNA,
                       dx=0, va="bottom", size=6.6 * variant.scale)
        if variant.name == "thesis":
            S.panel_letter(ax, "ABC"[ci], dx_mm=6.5 if ci == 0 else 3.0, dy_mm=1.0)

    return S.save(fig, "fig-4-1-reproduction", variant)


# ---------------------------------------------------------------------------- D4.2


@renders("4.2", "signature")
def signature(variant: S.Variant) -> str:
    """Raw versus normalised, both regimes. Spec: D4-2-signature.md.

    The connectivity scale is drawn in the raw row only: its presence above and absence
    below is the argument.
    """
    S.use(variant)
    d = _load("fig-4-2-signature.csv")
    fig = S.figure(S.FULL, S.RATIOS["standard"], variant)
    gs = fig.add_gridspec(
        2, 2, left=0.082, right=0.982, bottom=0.105, top=0.885, hspace=0.235, wspace=0.075
    )

    panels = ["reference", "canonical"]
    rows = [
        ("h1_max_persistence", "raw"),
        ("h1_max_persistence_normalised", "normalised"),
    ]
    titles = {
        "reference": r"reference   $p=113$, wd $0.1$",
        "canonical": r"canonical   $p=97$, wd $1.0$",
    }
    ratios = {
        ("reference", "h1_max_persistence"): "1.40  [0.70, 2.04]",
        ("reference", "h1_max_persistence_normalised"): "2.90  [2.20, 5.77]",
        ("canonical", "h1_max_persistence"): "0.91  [0.85, 1.21]",
        ("canonical", "h1_max_persistence_normalised"): "1.09  [0.88, 1.21]",
    }
    collapse = {"reference": r"scale falls $26\times$", "canonical": r"scale falls $155\times$"}
    letters = {(0, 0): "A", (0, 1): "B", (1, 0): "C", (1, 1): "D"}

    axes = {}
    for ci, panel in enumerate(panels):
        sub = d[d.panel == panel]
        tg = sorted(sub.groupby("run").t_g.first().dropna().unique())
        tg_med = float(np.median(tg))
        for ri, (col, rowname) in enumerate(rows):
            ax = fig.add_subplot(gs[ri, ci])
            axes[(ri, ci)] = ax

            # the window rule of §4.2, drawn so the ratio is checkable rather than asserted
            ax.axvspan(0.5 * tg_med, 0.9 * tg_med, facecolor=S.INK, alpha=0.045, lw=0, zorder=0)
            ax.axvspan(1.2 * tg_med, sub.step.max(), facecolor=S.INK, alpha=0.045, lw=0, zorder=0)

            for _, g in sub.groupby("run"):
                g = g.sort_values("step")
                ax.plot(g.step, g[col], color=S.RULE, lw=0.5, zorder=2)
            med = sub.groupby("step")[col].median()
            ax.plot(med.index, med.values, color=S.INK,
                    lw=S.EMPHASIS if rowname == "normalised" else S.DATA, zorder=4)

            # the connectivity scale — raw row only; its absence below is the argument
            if rowname == "raw":
                sc = sub.groupby("step").pointcloud_scale.median()
                axs = ax.twinx()
                axs.plot(sc.index, sc.values, color=S.RULE, lw=0.7, ls=(0, (1, 2.2)), zorder=1)
                axs.set_yscale("log")
                axs.yaxis.set_visible(False)
                for sp in axs.spines.values():
                    sp.set_visible(False)
                S.annotate(ax, 0.035, 0.045, collapse[panel], size=6.8 * variant.scale)

            _logx(ax, 100)
            S.seed_comb(ax, tg)
            ax.set_xlim(0, sub.step.max() * 1.02)
            ax.margins(y=0.10)
            # ratio placed in each row's empty quadrant: upper-right for the decaying raw
            # series, upper-left for the flat normalised one
            rx, ry, rha = (0.985, 0.90, "right") if rowname == "raw" else (0.035, 0.92, "left")
            # named once per row, in the left column: the grid is a small multiple, so the
            # name carries across to the panel beside it
            S.value(ax, rx, ry, ratios[(panel, col)],
                    r"plateau $\div$ baseline" if ci == 0 else "", ha=rha)
            if ri == 0:
                S.panel_title(ax, titles[panel], pad=8)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("training step")
            if ci == 0:
                ax.set_ylabel(
                    r"raw   $H_1^{\max}$" if rowname == "raw"
                    else r"normalised   $H_1^{\max}/s$"
                )

    # share y within a row: the comparison across regimes is the argument
    for ri in (0, 1):
        lo = min(axes[(ri, c)].get_ylim()[0] for c in (0, 1))
        hi = max(axes[(ri, c)].get_ylim()[1] for c in (0, 1))
        for c in (0, 1):
            axes[(ri, c)].set_ylim(lo, hi)
            if c == 1:
                axes[(ri, c)].set_yticklabels([])
            S.range_frame(axes[(ri, c)])

    if variant.name == "thesis":
        for (ri, ci), ax in axes.items():
            S.panel_letter(ax, letters[(ri, ci)])
    for ci in (0, 1):
        ax = axes[(1, ci)]
        tg = sorted(d[d.panel == panels[ci]].groupby("run").t_g.first().dropna().unique())
        ax.text(float(np.median(tg)) * 2.4, ax.get_ylim()[0], r"$t_g$", color=S.SIENNA,
                fontsize=6.4 * variant.scale, ha="left", va="bottom")

    # name the one subordinate trace, and the two windows the ratio is measured between.
    # The window names run up inside their own bands: set below the panel they read as a
    # second x-axis, and there is no room beside them for anything horizontal.
    S.annotate(axes[(0, 0)], 0.300, 0.775, "cloud scale, $s$", colour=S.RULE,
               size=6.6 * variant.scale)
    ax = axes[(0, 0)]
    tg_med = float(np.median(sorted(
        d[d.panel == "reference"].groupby("run").t_g.first().dropna().unique())))
    for lab, at in (("baseline", 0.67 * tg_med), ("plateau", 2.6 * tg_med)):
        ax.text(at, 0.30, lab, transform=ax.get_xaxis_transform(), ha="center", va="center",
                rotation=90, color=S.INK, alpha=0.50, fontsize=6.4 * variant.scale)

    return S.save(fig, "fig-4-2-signature", variant)


# ---------------------------------------------------------------------------- D4.3


@renders("4.3", "diagrams")
def diagrams(variant: S.Variant) -> str:
    """The dominant cycle leaving the diagonal. Spec: D4-3-diagrams.md.

    Each of the three stages carries its own axis limits, because the cloud contracts by
    a factor of ten across them and a shared axis would compress the first into a dot.
    The fourth panel puts all three on one axis after dividing each by its own scale,
    which is the argument of section 3.3.1 in a single picture.

    Laid out two by two rather than four across. A persistence diagram is a scatter whose
    reading is the distance of a point from the diagonal, so the panel has to be square and
    it has to be large enough to resolve a few hundred markers; four of them in a row on a
    158 mm column gives each one thirty millimetres, which resolves nothing.

    The panels are framed to where the loops live rather than to the origin. In a
    high-dimensional embedding a one-cycle is born late and lives briefly relative to its
    birth scale, so a [0, max death] frame renders the whole degree-one diagram as a
    smudge in one corner. The degree-zero features are born at zero and so cannot be
    drawn in that frame at all; their deaths are given as a rug along the foot of each
    panel, which is where the connectivity scale comes from.
    """
    S.use(variant)
    d = _load("fig-4-3-diagrams.csv")
    stages = ["memorising", "grokking", "final"]

    fig = S.figure(S.FULL, 0.885, variant)
    gs = fig.add_gridspec(2, 2, left=0.105, right=0.965, bottom=0.078, top=0.930,
                          wspace=0.215, hspace=0.235)

    for i, stage in enumerate(stages):
        ax = fig.add_subplot(gs[divmod(i, 2)])
        sub = d[d.stage == stage]
        h0, h1 = sub[sub.dim == 0], sub[sub.dim == 1]
        lo = min(h1.birth.min(), h0.death.min())
        hi = h1.death.max()
        pad = 0.07 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        ax.plot([lo, hi], [lo, hi], color=S.RULE, lw=S.HAIRLINE, zorder=1)
        life = (h1.death - h1.birth).to_numpy()
        ax.scatter(h1.birth, h1.death, s=4.0 + 36.0 * np.sqrt(life / life.max()), marker="o",
                   facecolors="none", edgecolors=S.BRONZE, linewidths=0.45, zorder=3)

        # where the degree-zero features die: the cloud's connectivity, as a rug
        ax.scatter(h0.death, np.full(len(h0), lo + 0.011 * (hi - lo)), marker="|", s=14,
                   color=S.RULE, linewidths=0.4, zorder=2, clip_on=False)

        # the dominant cycle, and how far it stands off the diagonal
        top = h1.loc[(h1.death - h1.birth).idxmax()]
        ax.scatter([top.birth], [top.death], s=26, marker="o", color=S.BRONZE, zorder=5,
                   linewidths=0)
        mid = 0.5 * (top.birth + top.death)
        ax.plot([top.birth, mid], [top.death, mid], color=S.BRONZE, lw=0.5, zorder=4)
        ax.annotate(f"{top.death - top.birth:.3f}", xy=(top.birth, top.death),
                    xytext=(-4.0, 3.0), textcoords="offset points", ha="right", va="bottom",
                    color=S.BRONZE, fontsize=7.0 * variant.scale)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        span = np.linspace(float(h0.death.min()), float(h1.death.max()), 3)
        ticks = [float(f"{t:.2g}") for t in span]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        S.range_frame(ax, x=(lo, hi), y=(lo, hi))
        S.panel_title(ax, f"step {int(sub.step.iloc[0]):,}", pad=5)
        S.annotate(ax, 0.045, 0.925, f"$s$ = {sub.scale.iloc[0]:.2f}")
        if i % 2 == 0:
            ax.set_ylabel("death")
        if i == 2:
            ax.set_xlabel("birth")
        if i == 0:
            S.annotate(ax, 0.985, 0.02, r"$H_0$ deaths", colour=S.RULE, ha="right",
                       size=6.6 * variant.scale, style="normal")
        if variant.name == "thesis":
            S.panel_letter(ax, "ABC"[i], dx_mm=8.5 if i % 2 == 0 else 4.0, dy_mm=0.5)

    # ... and the same three, each divided by its own scale
    ax = fig.add_subplot(gs[1, 1])
    shades = {"memorising": 0.55, "grokking": 0.78, "final": 1.0}
    ax.plot([0.70, 1.16], [0.70, 1.16], color=S.RULE, lw=S.HAIRLINE, zorder=1)
    for row, (stage, shade) in enumerate(shades.items()):
        h1 = d[(d.stage == stage) & (d.dim == 1)]
        colour = S.SEQUENTIAL(shade)
        b, dd = h1.birth / h1.scale, h1.death / h1.scale
        ax.scatter(b, dd, s=5.0, marker="o", color=colour, zorder=3, linewidths=0)
        top = (dd - b).idxmax()
        ax.scatter([b[top]], [dd[top]], s=30, marker="o", color=colour, zorder=5,
                   edgecolors=S.PAGE, linewidths=0.5)
        # the three dominant points sit within a few hundredths of each other on the
        # diagonal, so each is named on a leader into the half-plane below it, which in a
        # persistence diagram is empty by construction
        y = 0.265 - 0.088 * row
        ax.annotate(f"{stage}   {dd[top] - b[top]:.3f}", xy=(b[top], dd[top]),
                    xytext=(0.545, y), textcoords="axes fraction", color=colour,
                    va="center", ha="left", fontsize=7.0 * variant.scale,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.35, alpha=0.65,
                                    shrinkA=1.0, shrinkB=4.0,
                                    connectionstyle="arc3,rad=0.10"))
    ax.set_xlim(0.70, 1.16)
    ax.set_ylim(0.70, 1.16)
    ax.set_aspect("equal")
    ax.set_xticks([0.8, 0.9, 1.0, 1.1])
    ax.set_yticks([0.8, 0.9, 1.0, 1.1])
    S.range_frame(ax, x=(0.70, 1.16), y=(0.70, 1.16))
    S.panel_title(ax, r"all three,  $\div\, s$", pad=5)
    ax.set_xlabel(r"birth $\div s$")
    ax.set_ylabel(r"death $\div s$")
    if variant.name == "thesis":
        S.panel_letter(ax, "D", dx_mm=8.5, dy_mm=0.5)

    return S.save(fig, "fig-4-3-diagrams", variant)


# ---------------------------------------------------------------------------- D4.4


_OPERATION = {"add": "$a+b$", "sub": "$a-b$", "mul": r"$a \times b$", "div": r"$a \div b$",
              "compose": r"$S_5$", "permuted": "permuted labels", "poly": "$a^3+ab$"}


def _robustness_label(row) -> str:
    if row.block == "null model":
        return f"{_OPERATION[row.operation]}   $^{{{int(row.n_runs)}}}$"
    arch = "MLP" if row.model == "mlp" else "tf"
    modulus = "" if row.operation == "compose" else f"  $p${int(row.modulus)}"
    # the seed count rides on the label only where it is not the usual five of five
    n = "" if (row.n_runs == 5 and row.n_grokked == 5) else \
        f"  $^{{{int(row.n_grokked)}/{int(row.n_runs)}}}$"
    return (f"{arch}  {_OPERATION[row.operation]}{modulus}  $f${row.train_fraction:g}"
            f"  wd{row.weight_decay:g}{n}")


@renders("4.4", "robustness")
def robustness(variant: S.Variant) -> str:
    """The robustness map. Spec: D4-4-robustness.md.

    The table beside this carries the numbers an examiner checks; the figure carries the
    pattern the table cannot, which is that the ordering by signature strength is very
    nearly the ordering by circularity. That is why the sparkline column is here.
    """
    S.use(variant)
    d = _load("fig-4-4-robustness.csv")
    obs = "h1_max_persistence_normalised"
    lo_band, hi_band = float(d[f"{obs}__null_lo"].iloc[0]), float(d[f"{obs}__null_hi"].iloc[0])
    conditions = d[d.block == "condition"].reset_index(drop=True)
    nulls = d[d.block == "null model"].reset_index(drop=True)

    # Rows run top to bottom in the table's order, so y is inverted: index 0 is the top.
    fig = S.figure(S.FULL, 0.680, variant)
    gs = fig.add_gridspec(
        2, 3, height_ratios=[len(conditions), len(nulls) + 0.6], width_ratios=[1.0, 0.17, 0.17],
        left=0.228, right=0.975, bottom=0.108, top=0.912, hspace=0.075, wspace=0.055
    )
    ax = fig.add_subplot(gs[0, 0])
    axn = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[0, 2])

    def forest(a, frame, offset=0.0):
        for i, r in frame.iterrows():
            verdict = r[f"{obs}__verdict"]
            colour = {"above": S.BRONZE, "below": S.SLATE}.get(verdict, S.INK)
            y = i + offset
            a.plot([r[f"{obs}__lo"], r[f"{obs}__hi"]], [y, y], color=colour, lw=S.DATA,
                   solid_capstyle="butt", zorder=3)
            a.scatter([r[f"{obs}__med"]], [y], s=20, marker="o", zorder=4, linewidths=0.8,
                      facecolors=colour if verdict == "above" else "none", edgecolors=colour)

    for a in (ax, axn):
        S.null_band(a, lo_band, hi_band, horizontal=False)
        a.set_xscale("log")
        a.set_xlim(0.15, 7.0)
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
        a.minorticks_off()

    forest(ax, conditions)
    forest(axn, nulls)

    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([_robustness_label(r) for _, r in conditions.iterrows()],
                       fontsize=6.4 * variant.scale)
    ax.set_ylim(len(conditions) - 0.4, -0.6)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)

    axn.set_yticks(range(len(nulls)))
    axn.set_yticklabels([_robustness_label(r) for _, r in nulls.iterrows()],
                        fontsize=6.4 * variant.scale)
    axn.set_ylim(len(nulls) - 0.4, -0.8)
    axn.set_xticks([0.2, 0.5, 1, 2, 5])
    axn.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    axn.set_xlabel(r"plateau $\div$ baseline,   normalised $H_1^{\max}$")
    axn.spines["bottom"].set_color(S.RULE)
    axn.spines["bottom"].set_bounds(float(d[f"{obs}__lo"].min()), float(d[f"{obs}__hi"].max()))
    S.annotate(axn, -0.010, 0.98, "null models", colour=S.INK, ha="right", style="normal",
               size=6.4 * variant.scale)

    # bilateral labelling: condition names down the left, the variable that orders them
    # down the right (device 4), so the association of the next section is visible here
    S.sparkline(axc, conditions.circularity.fillna(0.0),
                colours=[{"above": S.BRONZE, "below": S.SLATE}.get(v, S.INK)
                         for v in conditions[f"{obs}__verdict"]],
                ticks=(0, 1), label="circularity")
    S.sparkline(axs, conditions.scale_collapse,
                colours=[{"above": S.BRONZE, "below": S.SLATE}.get(v, S.INK)
                         for v in conditions[f"{obs}__verdict"]],
                log=True, kind="dot", ticks=(10, 100), label=r"scale $\div$")
    # the null rows carry the same two variables; theirs are the lowest circularity in the set
    axcn = fig.add_subplot(gs[1, 1])
    axsn = fig.add_subplot(gs[1, 2])
    S.sparkline(axcn, nulls.circularity, ticks=(0, 1))
    S.sparkline(axsn, nulls.scale_collapse, log=True, kind="dot", ticks=(10, 100))
    for a, source in ((axc, ax), (axs, ax), (axcn, axn), (axsn, axn)):
        a.set_ylim(source.get_ylim())
    for a in (axc, axs):
        a.set_xticklabels([])
        a.spines["bottom"].set_visible(False)
        a.tick_params(axis="x", length=0)
    for a in (axs, axsn):
        a.set_xlim(4, 400)
    S.annotate(axcn, 0.5, -0.62, r"$\rho = 0.705$", ha="center", va="top",
               size=6.6 * variant.scale)

    if variant.name == "thesis":
        S.panel_letter(ax, "A", dx_mm=34.0, dy_mm=3.6)
        S.panel_letter(axn, "B", dx_mm=34.0, dy_mm=1.2)

    return S.save(fig, "fig-4-4-robustness", variant)


# ---------------------------------------------------------------------------- D4.6


@renders("4.6", "circularity")
def circularity(variant: S.Variant) -> str:
    """Circularity predicts the signature; generalisation does not. Spec: D4-6-circularity.md."""
    S.use(variant)
    d = _load("fig-4-6-circularity.csv").dropna(
        subset=["circularity", "h1_max_persistence_normalised__ratio"]
    )
    y = d["h1_max_persistence_normalised__ratio"]

    fig = S.figure(S.FULL, 0.615, variant)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[135, 20], left=0.080, right=0.978, bottom=0.150, top=0.918,
        wspace=0.185
    )
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)

    S.null_band(ax, 0.75, 1.32)

    # fit on log y
    m = np.isfinite(d.circularity) & (y > 0)
    b, a = np.polyfit(d.circularity[m], np.log(y[m]), 1)
    xs = np.linspace(d.circularity.min(), d.circularity.max(), 100)
    ax.plot(xs, np.exp(a + b * xs), color=S.BRONZE, lw=S.SECONDARY, zorder=3)

    # three channels: glyph = architecture, lightness = weight decay, position = the two variables
    wd = d.weight_decay.replace(0.0, 0.05)
    lo, hi = np.log10(0.05), np.log10(3.0)
    shade = (np.log10(wd) - lo) / (hi - lo)
    for marker, sel in (("o", d.model == "transformer"), ("s", d.model == "mlp")):
        idx = sel.values
        ax.scatter(
            d.circularity[idx], y[idx], marker=marker, s=17,
            c=S.SEQUENTIAL(0.15 + 0.75 * shade[idx]), edgecolors=S.INK, linewidths=0.4, zorder=5,
        )

    ax.set_yscale("log")
    ax.set_xlim(0.05, 1.02)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticks([0.3, 1, 3, 10])
    ax.set_yticklabels(["0.3", "1", "3", "10"])
    ax.set_xlabel("terminal Fourier concentration")
    ax.set_ylabel(r"normalised $H_1$ ratio")
    S.range_frame(ax)
    S.annotate(ax, 0.025, 0.925, r"$\rho = 0.705$", size=8.2 * variant.scale)
    S.annotate(ax, 0.025, 0.855, r"$p = 7\times10^{-12}$", size=7.2 * variant.scale)

    # the degenerate strip: the same runs against final test accuracy
    for marker, sel in (("o", d.model == "transformer"), ("s", d.model == "mlp")):
        idx = sel.values
        ax2.scatter(
            d["test_acc__final"][idx], y[idx], marker=marker, s=17,
            c=S.SEQUENTIAL(0.15 + 0.75 * shade[idx]), edgecolors=S.INK, linewidths=0.4, zorder=5,
        )
    ax2.set_xlim(0.965, 1.035)
    ax2.set_xticks([1.0])
    ax2.set_xticklabels(["1.0"])
    ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("final\ntest acc.")
    ax2.spines["bottom"].set_bounds(0.97, 1.03)
    ax2.spines["bottom"].set_color(S.RULE)

    # the key, in the empty lower-right quadrant. Three channels are in play at once and
    # none of them can be direct-labelled on a scatter of seventy-one points.
    lx, ly, lw_, lh = 0.585, 0.045, 0.400, 0.235
    ax.add_patch(plt.Rectangle((lx, ly), lw_, lh, transform=ax.transAxes, facecolor="none",
                               edgecolor=S.RULE, lw=S.HAIRLINE, zorder=6))
    ax.text(lx + 0.028, ly + lh - 0.030, "run", transform=ax.transAxes, color=S.INK,
            family=S.SMALLCAPS, fontsize=6.8 * variant.scale, ha="left", va="top")
    for k, (mk, lab) in enumerate((("o", "transformer"), ("s", "MLP"))):
        ax.scatter([lx + 0.048], [ly + lh - 0.098 - 0.060 * k], marker=mk, s=15,
                   c=[S.SEQUENTIAL(0.55)], edgecolors=S.INK, linewidths=0.4,
                   transform=ax.transAxes, zorder=7, clip_on=False)
        ax.text(lx + 0.082, ly + lh - 0.098 - 0.060 * k, lab, transform=ax.transAxes,
                color=S.INK, fontsize=6.4 * variant.scale, va="center")
    for k in range(6):
        ax.add_patch(plt.Rectangle((lx + 0.222 + 0.026 * k, ly + 0.058), 0.026, 0.032,
                                   transform=ax.transAxes, facecolor=S.SEQUENTIAL(0.15 + 0.15 * k),
                                   edgecolor="none", zorder=7))
    ax.text(lx + 0.222, ly + 0.036, "0.1", transform=ax.transAxes, color=S.INK,
            fontsize=6.0 * variant.scale, ha="left", va="top")
    ax.text(lx + 0.378, ly + 0.036, "3.0", transform=ax.transAxes, color=S.INK,
            fontsize=6.0 * variant.scale, ha="right", va="top")
    ax.text(lx + 0.300, ly + 0.104, "weight decay", transform=ax.transAxes, color=S.INK,
            fontsize=6.4 * variant.scale, ha="center", va="bottom")

    if variant.name == "thesis":
        S.panel_letter(ax, "A")
        S.panel_letter(ax2, "B", dx_mm=4.5)

    return S.save(fig, "fig-4-6-circularity", variant)


# ---------------------------------------------------------------------------- D5.1


def _baseline_ratio(frame: pd.DataFrame, column: str) -> pd.Series:
    """A series divided by its own baseline median, so it reads against the null band.

    The window rule of thesis section 4.2, applied to the whole series rather than to its
    two windows: this is the same quantity the tables quote, drawn as it evolves.
    """
    tg = float(frame.t_g.iloc[0])
    step, value = frame.step.to_numpy(float), frame[column].to_numpy(float)
    base = np.nanmedian(value[(step >= 0.5 * tg) & (step <= 0.9 * tg)])
    return pd.Series(value / base, index=frame.index)


@renders("5.1", "basis")
def basis(variant: S.Variant) -> str:
    """What a change of basis costs a spectral statistic. Spec: D5-1-basis.md.

    Two rows because the figure makes two claims and only one of them is comfortable.
    Above: the residue-axis transform never leaves the noise the null models produce,
    while the same transform in the group's own basis rises. Below: what persistent
    homology, which was told neither basis, reports about the same runs.
    """
    S.use(variant)
    d = _load("fig-5-1-basis.csv")
    operations = [("mul", r"$a \times b$"), ("div", r"$a \div b$")]
    noise_lo, noise_hi = 0.119, 0.138  # terminal k = 5 concentration across the sixteen nulls

    fig = S.figure(S.FULL, S.RATIOS["standard"], variant)
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.828, bottom=0.100, top=0.892,
                          hspace=0.245, wspace=0.195)
    axes = {}
    for ci, (operation, name) in enumerate(operations):
        sub = d[d.operation == operation]
        tg = sorted(sub.groupby("run").t_g.first().dropna())
        end = sub.step.max()

        top = fig.add_subplot(gs[0, ci])
        bottom = fig.add_subplot(gs[1, ci])
        axes[(0, ci)], axes[(1, ci)] = top, bottom

        top.axhspan(noise_lo, noise_hi, facecolor=S.RULE, alpha=0.30, lw=0, zorder=0)
        # basis and terminal value are one label, set outside the frame where the series
        # ends. Named in the right column only: the two panels are a small multiple, so
        # naming both would be the same label twice.
        for column, dash, weight, basis_name in (
            ("fourier_concentration_k5", (0, (4, 2)), 1.0, "residue axis"),
            ("fourier_concentration_group_k5", "solid", 1.15, "discrete log"),
        ):
            for _, g in sub.groupby("run"):
                g = g.sort_values("step")
                top.plot(g.step, g[column], color=S.RULE, lw=0.45, zorder=2)
            median = sub.groupby("step")[column].median()
            top.plot(median.index, median.values, color=S.BRONZE, lw=weight, ls=dash, zorder=4)
            terminal = f"{median.iloc[-1]:.3f}"
            S.direct_label(top, end, median.iloc[-1],
                           f"{basis_name}   {terminal}" if ci == 1 else terminal,
                           S.BRONZE, dx=4, size=6.8 * variant.scale)
        top.set_ylim(0, 0.80)
        top.set_yticks([0, 0.25, 0.5, 0.75])

        S.null_band(bottom, 0.75, 1.32)
        # the ratio the thesis quotes for this condition, from the same table the figure
        # of section 4.4 is drawn from, so the two cannot drift apart
        q = _condition_row(operation=operation, train_fraction=0.5)
        S.value(bottom, 0.030, 0.905, f"{q.med:.2f}  [{q.lo:.2f}, {q.hi:.2f}]",
                r"plateau $\div$ baseline" if ci == 0 else "")
        ratios = []
        for _, g in sub.groupby("run"):
            g = g.sort_values("step")
            r = _baseline_ratio(g, "h1_max_persistence_normalised")
            ratios.append(pd.DataFrame({"step": g.step.values, "r": r.values}))
            bottom.plot(g.step, r.values, color=S.RULE, lw=0.45, zorder=2)
        median = pd.concat(ratios).groupby("step").r.median()
        bottom.plot(median.index, median.values, color=S.INK, lw=1.15, zorder=4)
        bottom.set_yscale("log")
        bottom.set_ylim(0.28, 3.6)
        bottom.set_yticks([0.3, 1, 3])
        bottom.set_yticklabels(["0.3", "1", "3"])
        bottom.minorticks_off()

        for ax in (top, bottom):
            _logx(ax, 100)
            ax.set_xlim(0, end)
            S.seed_comb(ax, tg, height=0.055)
            S.range_frame(ax)
        top.set_xticklabels([])
        bottom.set_xlabel("training step")
        S.panel_title(top, name, pad=8)
        if ci == 0:
            top.set_ylabel("Fourier concentration,  $k = 5$")
            bottom.set_ylabel(r"$H_1^{\max}/s$   $\div$ baseline")
        else:
            top.set_yticklabels([])
            bottom.set_yticklabels([])

    outer = d[d.operation == "div"].step.max()
    S.direct_label(axes[(1, 1)], outer, 1.0, r"$H_1$", S.INK, dx=4, size=7.2 * variant.scale)
    S.annotate(axes[(0, 0)], 0.030, noise_hi + 0.012, "null", colour=S.INK, style="normal",
               size=6.2 * variant.scale, transform=axes[(0, 0)].get_yaxis_transform())
    S.annotate(axes[(1, 0)], 0.185, 1.36, "null", colour=S.INK, style="normal",
               size=6.4 * variant.scale, transform=axes[(1, 0)].get_yaxis_transform())

    if variant.name == "thesis":
        for (ri, ci), ax in axes.items():
            S.panel_letter(ax, "ABCD"[2 * ri + ci], dx_mm=7.5 if ci == 0 else 3.0)

    return S.save(fig, "fig-5-1-basis", variant)


# ---------------------------------------------------------------------------- D5.2


@renders("5.2", "noncyclic")
def noncyclic(variant: S.Variant) -> str:
    """A task that groks with no circle to find. Spec: D5-2-noncyclic.md.

    Two columns because there are two facts and they need different horizontal axes. The
    spread in transition time is only legible against absolute steps; the persistence
    comparison is only legible once each seed is rescaled by its own transition, since
    five transitions six-fold apart otherwise smear the picture into nothing.

    The left column is split because the pre-transition plateau is the quantitative test
    of the leakage model of section 3.5, and at full scale it is a line on the axis.
    """
    S.use(variant)
    d = _load("fig-5-2-noncyclic.csv")
    band = float(d.null_lo.iloc[0]), float(d.null_hi.iloc[0])
    predicted = 0.6 * 840 / 14400  # section 3.5, from the 840 commuting pairs of S_5

    fig = S.figure(S.FULL, 0.585, variant)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.05, 1.0], left=0.068, right=0.912,
                          bottom=0.140, top=0.900, wspace=0.225, hspace=0.255)
    top = fig.add_subplot(gs[0, 0])
    strip = fig.add_subplot(gs[1, 0])
    right = fig.add_subplot(gs[:, 1])

    runs = sorted(d.run.unique(), key=lambda r: d.loc[d.run == r, "t_g"].iloc[0])
    censored = runs[-1]  # groks at 89,700 of a 100,000-step budget: no plateau window

    strip.axhline(predicted, color=S.BRONZE, lw=S.HAIRLINE, zorder=1)
    for run in runs:
        g = d[d.run == run].sort_values("step")
        colour = S.SLATE if run == censored else S.INK
        for ax in (top, strip):
            ax.plot(g.step, g.test_acc, color=colour, lw=0.9, zorder=3)

    for ax in (top, strip):
        _logx(ax, 100)
        ax.set_xlim(0, d.step.max())
        S.seed_comb(ax, sorted(d.groupby("run").t_g.first()), height=0.06)
    top.set_ylim(-0.03, 1.06)
    top.set_yticks([0, 0.5, 1.0])
    top.set_yticklabels(["0", "0.5", "1"])
    top.set_xticklabels([])
    top.set_ylabel("test accuracy")
    S.range_frame(top, y=(0, 1))
    S.panel_title(top, r"$S_5$   absolute steps", pad=7)
    S.annotate(top, 0.030, 0.86,
               "  ".join(f"{t / 1000:.1f}" for t in sorted(d.groupby("run").t_g.first())) + "k",
               colour=S.SIENNA, size=6.6 * variant.scale, style="normal")

    strip.set_ylim(0, 0.062)
    strip.set_yticks([0, 0.035])
    strip.set_yticklabels(["0", "0.035"])
    strip.set_xlabel("training step")
    S.range_frame(strip, y=(0, 0.06))
    measured = _claims()["s5_measured_plateau"]["median"]
    # no y-label: the strip is a zoom of the panel above it, sharing both quantity and
    # horizontal axis, and its two ticks already report the range
    S.value(strip, 0.028, 0.985, f"{measured:.3f}", "measured", colour=S.INK)
    S.value(strip, 0.290, 0.985, f"{predicted:.3f}", "predicted")

    S.null_band(right, *band)
    right.axvline(1.0, color=S.SIENNA, lw=S.HAIRLINE, zorder=1)
    for run in runs:
        g = d[d.run == run].sort_values("step").dropna(subset=["step_over_tg"])
        colour = S.SLATE if run == censored else S.INK
        ratio = _baseline_ratio(g, "h1_total_persistence_normalised")
        right.plot(g.step_over_tg, ratio.values, color=colour, lw=0.9, zorder=3)
        S.direct_label(right, g.step_over_tg.iloc[-1], ratio.iloc[-1], f"{ratio.iloc[-1]:.1f}",
                       colour, dx=4, size=6.6 * variant.scale)
        if run == censored:
            right.scatter([g.step_over_tg.iloc[-1]], [ratio.iloc[-1]], s=17, marker="o",
                          facecolors="none", edgecolors=S.SLATE, linewidths=0.8, zorder=5)

    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(0.06, 8.0)
    right.set_ylim(0.10, 120.0)
    right.set_xticks([0.1, 1, 5])
    right.set_xticklabels(["0.1", "1", "5"])
    right.set_yticks([0.1, 1, 10, 100])
    right.set_yticklabels(["0.1", "1", "10", "100"])
    right.minorticks_off()
    right.set_xlabel(r"step $\div\, t_g$")
    right.set_ylabel(r"total $H_1/s$   $\div$ baseline")
    S.range_frame(right)
    S.panel_title(right, r"$S_5$   rescaled by $t_g$", pad=7)
    S.annotate(right, 1.0, 0.045, r"$t_g$", colour=S.SIENNA, ha="left", style="normal",
               size=6.6 * variant.scale, transform=right.get_xaxis_transform())
    S.annotate(right, 0.030, band[1] * 1.10, "null", colour=S.INK, style="normal",
               size=6.2 * variant.scale, transform=right.get_yaxis_transform())

    if variant.name == "thesis":
        S.panel_letter(top, "A")
        S.panel_letter(strip, "B", dy_mm=-0.5)
        S.panel_letter(right, "C")

    return S.save(fig, "fig-5-2-noncyclic", variant)


# ---------------------------------------------------------------------------- D5.3


@renders("5.3", "headtohead")
def headtohead(variant: S.Variant) -> str:
    """What early-window topology adds. Spec: D5-3-headtohead.md.

    Two questions of the same features, and the harder one is the one that would matter.
    Whether a run groks at all is a classification problem the cheap observables already
    answer; when it groks is a regression problem nothing answers, and the fold spread is
    drawn rather than summarised because it is the reason the second panel says nothing.

    The lower strips are the claim by itself: the score of the cheap observables together,
    subtracted from the score of the same set with the topological features added.

    The four series cross twice and end within a tenth of each other, so there is no
    terminus to name them at --- the one case §4.6 keeps the ruled key block for. It also
    fills the corner the two short strips leave empty.
    """
    S.use(variant)
    d = _load("fig-5-3-headtohead.csv")
    windows = ["w1000", "w5000", "tc"]
    ticks = ["1,000", "5,000", r"$t_c$"]
    families = {
        "topology": ("topology", *S.SERIES["topology"]),
        "fourier": ("Fourier", *S.SERIES["fourier"]),
        "weight_norm": ("weight norm", *S.SERIES["weight"]),
        "lid": ("intrinsic dimension", *S.SERIES["lid"]),
    }
    tasks = [("classification", "area under the curve", (0.40, 0.98), 0.5, "chance", 0.985),
             ("regression", r"$R^2$", (-2.8, 0.9), 0.0, "the mean", 0.40)]

    fig = S.figure(S.FULL, 0.655, variant)
    outer = fig.add_gridspec(2, 1, height_ratios=[2.15, 1.0], left=0.072, right=0.985,
                             bottom=0.098, top=0.905, hspace=0.46)
    top = outer[0, 0].subgridspec(1, 2, wspace=0.215)
    bottom = outer[1, 0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.92], wspace=0.215)

    panels = []
    for ci, (task, ylabel, ylim, floor, floor_name, floor_x) in enumerate(tasks):
        ax = fig.add_subplot(top[0, ci])
        panels.append(ax)
        sub = d[d.task == task]
        ax.axhline(floor, color=S.RULE, lw=S.HAIRLINE, zorder=1)
        for key, (_name, colour, dash, glyph) in families.items():
            arm = sub[sub.feature_set == key].set_index("window").reindex(windows)
            x = np.arange(len(windows))
            ax.errorbar(x, arm.score_mean, yerr=arm.score_std, color=colour, lw=0.9,
                        ls="solid" if dash == (None, None) else (0, dash), marker=glyph,
                        ms=3.4, mew=0.0, elinewidth=S.HAIRLINE, capsize=1.8,
                        capthick=S.HAIRLINE, alpha=0.90, zorder=3)
        ax.set_xticks(np.arange(len(windows)))
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.35, 2.35)
        ax.set_ylim(*ylim)
        ax.set_xlabel("early window, steps")
        ax.set_ylabel(ylabel)
        S.range_frame(ax, x=(0, 2))
        S.panel_title(ax, task, pad=7)
        S.annotate(ax, floor_x, floor, floor_name, colour=S.RULE, style="normal", ha="right",
                   size=6.4 * variant.scale, va="top", transform=ax.get_yaxis_transform())

    for ci, (task, *_rest) in enumerate(tasks):
        ax = fig.add_subplot(bottom[0, ci])
        panels.append(ax)
        values = [float(d[(d.task == task) & (d.window == w)
                          & (d.feature_set == "baselines+topology")]
                        .increment_over_baselines.iloc[0]) for w in windows]
        y = np.arange(len(values))
        ax.axvline(0.0, color=S.INK, lw=0.6, zorder=2)
        for value, row in zip(values, y, strict=True):
            ax.plot([0, value], [row, row], color=S.BRONZE, lw=S.HAIRLINE, zorder=3)
        ax.scatter(values, y, s=19, marker="o", zorder=4, linewidths=0.8,
                   facecolors=[S.BRONZE if v > 0 else "none" for v in values],
                   edgecolors=S.BRONZE)
        ax.set_yticks(y)
        ax.set_yticklabels(ticks, fontsize=6.8 * variant.scale)
        ax.set_ylim(len(values) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(-1.15, 0.55)
        ax.set_xticks([-1.0, 0, 0.5])
        ax.set_xticklabels(["$-1$", "0", "0.5"])
        ax.spines["bottom"].set_color(S.RULE)
        S.panel_title(ax, task, pad=5)
        ax.set_xlabel("added by topology")

    box = fig.add_subplot(bottom[0, 2])
    rect = box.get_position()
    box.remove()

    def swatch(colour, dash, glyph):
        def draw(ax, y):
            ax.plot([0.07, 0.245], [y, y], color=colour, lw=0.9, clip_on=False,
                    ls="solid" if dash == (None, None) else (0, dash))
            ax.plot([0.1575], [y], marker=glyph, color=colour, ms=3.4, mew=0.0,
                    clip_on=False)
        return draw

    S.key(fig, (rect.x0, rect.y0, rect.width, rect.height),
          [(name, swatch(colour, dash, glyph))
           for name, colour, dash, glyph in families.values()],
          heading="feature set", note="bars: mean $\\pm$ s.d. over folds")

    if variant.name == "thesis":
        for ax, letter, dx in zip(panels, "ABCD", (7.5, 5.0, 7.5, 5.0), strict=True):
            S.panel_letter(ax, letter, dx_mm=dx, dy_mm=0.5)

    return S.save(fig, "fig-5-3-headtohead", variant)


# ---------------------------------------------------------------------------- D5.4


@renders("5.4", "pid")
def pid(variant: S.Variant) -> str:
    """The redundancy question, decomposed. Spec: D5-4-pid.md.

    A stacked bar, which this house style otherwise forbids, because here the whole is
    the joint mutual information and the four atoms genuinely partition it.

    The second panel is not decoration. Under the minimum-mutual-information redundancy
    function, redundancy is defined as the smaller of the two source-target mutual
    informations, so the unique atom of whichever source carries less is zero by
    construction rather than by measurement. That is the case here, and a figure showing
    the first panel alone would invite the reader to read a structural zero as a finding.

    Both panels are drawn on one bits axis, so the argument is made by alignment: the
    right-hand edge of the redundant atom falls on the end of the $H_1$ bar because the
    estimator defines it to. Two-thirds width, because one decomposition of a sixth of a
    bit does not fill a page and padding it out would say otherwise.
    """
    S.use(variant)
    d = _load("fig-5-4-pid.csv").set_index("atom")
    atoms = [("redundant", "redundant", S.RULE),
             ("unique_a", r"unique to $H_1$", S.INK),
             ("unique_b", "unique to Fourier", S.BRONZE),
             ("synergistic", "synergistic", S.SLATE)]
    total = float(d.total.iloc[0])
    redundant = float(d.bits["redundant"])
    span = (0.0, total * 1.115)

    fig = S.figure(S.TWO_THIRDS, 0.615, variant)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.80], left=0.128, right=0.905,
                          bottom=0.150, top=0.880, hspace=0.72)
    ax, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])

    left = 0.0
    for key, name, colour in atoms:
        width = float(d.bits[key])
        ax.barh([0], [width], left=left, height=0.62, color=colour, zorder=3,
                edgecolor=variant.ground or "white", linewidth=0.5)
        centre, drop = left + 0.5 * width, 7
        if width <= 0.002:  # a segment too thin to see, marked where it would be
            ax.plot([left, left], [-0.34, 0.34], color=colour, lw=1.2, zorder=4)
            centre, drop = left, 30
            ax.plot([left, left], [-0.34, -1.02], color=colour, lw=S.HAIRLINE, zorder=3)
        ax.annotate(f"{name}\n{width:.3f}", xy=(centre, -0.34), xytext=(0, -drop),
                    textcoords="offset points", ha="center", va="top", color=colour,
                    fontsize=6.8 * variant.scale, linespacing=1.5)
        left += width

    ax.set_ylim(-2.9, 0.8)
    ax.text(total * 1.02, 0.0, f"{total:.3f} bits", ha="left", va="center", color=S.INK,
            fontsize=7.0 * variant.scale)
    S.annotate(ax, -0.012, 0.0, "joint", colour=S.INK, ha="right", va="center",
               style="normal", size=6.8 * variant.scale, transform=ax.get_yaxis_transform())

    # why one atom is zero: the estimator's redundancy is the smaller of the two, so the
    # dashed rule dropped from the atom boundary lands on the end of the shorter bar
    for i, (key, name, colour) in enumerate((("mi_a", r"$H_1$", S.INK),
                                             ("mi_b", "Fourier", S.BRONZE))):
        value = float(d[key].iloc[0])
        ax2.barh([i], [value], height=0.46, color=colour, zorder=3)
        ax2.text(value + 0.0035, i, f"{value:.3f}", va="center", color=colour,
                 fontsize=6.8 * variant.scale)
        S.annotate(ax2, -0.012, i, name, colour=S.INK, ha="right", va="center",
                   style="normal", size=6.8 * variant.scale,
                   transform=ax2.get_yaxis_transform())
    ax2.axvline(redundant, color=S.RULE, lw=S.HAIRLINE, ls=(0, (2, 2)), zorder=2)
    ax2.set_ylim(1.72, -0.78)
    ax2.set_xlabel(r"$I(\,\cdot\,;\ \log t_g)$,  bits")

    for a in (ax, ax2):
        a.set_xlim(*span)
        a.set_yticks([])
        a.spines["left"].set_visible(False)
        a.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xticklabels([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax2.set_xticklabels(["0", "0.05", "0.10", "0.15"])
    S.range_frame(ax2, x=span, y=(1.72, -0.78))
    ax2.spines["left"].set_visible(False)

    S.annotate(ax, 1.0, 1.06, f"Gaussian MMI,  Williams\u2013Beer lattice,  "
               f"$n$ = {int(d.n.iloc[0])}", colour=S.INK, ha="right", va="bottom",
               style="normal", size=6.4 * variant.scale)

    if variant.name == "thesis":
        S.panel_letter(ax, "A", dx_mm=9.5, dy_mm=1.0)
        S.panel_letter(ax2, "B", dx_mm=9.5, dy_mm=1.0)

    return S.save(fig, "fig-5-4-pid", variant)


# ---------------------------------------------------------------------------- D5.5


@renders("5.5", "lag")
def lag(variant: S.Variant) -> str:
    """The signed lag, ordered by signature strength. Spec: D5-5-lag.md.

    The figure's claim is about dispersion as much as sign: conditions whose signature
    clears its null band produce tight, small, negative lags; conditions without one
    produce values scattered over two orders of magnitude on both sides, because the
    detector is being asked to time an event that did not occur.
    """
    S.use(variant)
    obs = "h1_max_persistence_normalised"
    d = _load("fig-5-5-lag.csv")
    d = d[(d.observable == obs) & d.delta.notna()]
    cond = pd.read_csv("results/processed/thesis/conditions.csv")

    keys = ["model", "operation", "modulus", "train_fraction", "weight_decay", "loss", "optimizer"]
    rows = []
    for key, g in d.groupby(keys):
        c = cond
        for k, v in zip(keys, key, strict=True):
            c = c[c[k] == v]
        if c.empty or not np.isfinite(c["h1_max_persistence_normalised__med"].iloc[0]):
            continue
        rows.append(
            {
                "label": _condition_label(key),
                "n": len(g),
                "median": float(g.delta.median()),
                "lo": float(g.delta.min()),
                "hi": float(g.delta.max()),
                "ratio": float(c["h1_max_persistence_normalised__med"].iloc[0]),
                "circ": float(c["circularity"].iloc[0]),
                "clears": c["h1_max_persistence_normalised__verdict"].iloc[0] == "above",
            }
        )
    t = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)

    nulls = _load("fig-5-5-lag.csv")
    nulls = nulls[(nulls.observable == obs) & nulls.label_permutation & nulls.t_top.notna()]

    # Standard plate: full column width, filled by two sparkline columns beside the forest
    # and the null strip beneath it (master §4.4).
    fig = S.figure(S.FULL, 0.72, variant)
    gs = fig.add_gridspec(
        2, 3, height_ratios=[len(t), 2.6], width_ratios=[1.0, 0.135, 0.135],
        left=0.245, right=0.975, bottom=0.105, top=0.925, hspace=0.10, wspace=0.055
    )
    ax = fig.add_subplot(gs[0, 0])
    axn = fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])

    lim = 1.35e5
    for a in (ax, axn):
        a.set_xscale("symlog", linthresh=3000, linscale=0.55)
        a.set_xlim(-lim, lim)
        a.axvspan(-lim, 0, facecolor=S.BRONZE, alpha=0.055, lw=0, zorder=0)
        a.axvspan(0, lim, facecolor=S.SLATE, alpha=0.055, lw=0, zorder=0)
        a.axvline(0, color=S.INK, lw=0.6, zorder=1)

    for i, r in t.iterrows():
        colour = S.BRONZE if r.clears else S.INK
        ax.plot([r.lo, r.hi], [i, i], color=colour, lw=S.DATA, solid_capstyle="butt", zorder=3)
        ax.scatter([r["median"]], [i], s=22, marker="o", zorder=4,
                   facecolors=colour if r.clears else "none", edgecolors=colour, linewidths=0.8)
    ax.axvline(-1000, color=S.BRONZE, lw=0.5, ls=(0, (1, 2)), zorder=2)

    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.label, fontsize=6.4 * variant.scale)
    ax.set_ylim(-0.8, len(t) - 0.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xticklabels([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    S.annotate(ax, 0.47, 1.015, "lags", colour=S.INK, ha="right", style="normal",
               size=7.2 * variant.scale)
    S.annotate(ax, 0.53, 1.015, "leads", colour=S.SLATE, ha="left", style="normal",
               size=7.2 * variant.scale)
    S.annotate(ax, -1000, -0.012, "Tang, $\\approx$1,000", colour=S.BRONZE, ha="right",
               va="top", size=6.0 * variant.scale, transform=ax.get_xaxis_transform())

    # the null strip: no t_g, so no lag — only the steps the detector chose to call a transition
    axn.scatter(nulls.t_top, np.zeros(len(nulls)), s=20, marker="o", facecolors="none",
                edgecolors=S.INK, linewidths=0.8, zorder=4)
    axn.set_ylim(-1.1, 1.1)
    axn.set_yticks([0])
    axn.set_yticklabels(["permuted labels"], fontsize=6.6 * variant.scale)
    axn.tick_params(axis="y", length=0)
    axn.spines["left"].set_visible(False)
    axn.set_xlabel("signed lag  $\\Delta = t_g - t_{\\mathrm{top}}$   (steps)")
    axn.set_xticks([-1e5, -1e4, 0, 1e4, 1e5])
    axn.set_xticklabels(["$-10^5$", "$-10^4$", "0", "$10^4$", "$10^5$"])
    axn.spines["bottom"].set_color(S.RULE)
    S.annotate(axn, 0.02, 0.80, "$t_{top}$ only", colour=S.INK, ha="left",
               style="normal",
               size=6.4 * variant.scale)

    # why a row is filled: the signature strength that orders the forest, as a sparkline
    # column (device 5). Rows with a signature have long bars and tight negative lags.
    axr.axvspan(0.75, 1.32, facecolor=S.RULE, alpha=0.30, lw=0, zorder=0)
    axr.barh(range(len(t)), t.ratio, height=0.42, zorder=3,
             color=[S.BRONZE if c else S.INK for c in t.clears])
    axr.set_xscale("log")
    axr.set_xlim(0.28, 12.0)
    axr.set_xticks([1, 10])
    axr.set_xticklabels(["1", "10"], fontsize=6.0 * variant.scale)
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks([])
    axr.minorticks_off()
    axr.spines["left"].set_visible(False)
    axr.spines["bottom"].set_color(S.RULE)
    S.annotate(axr, 0.5, 1.015, r"$H_1^{\max}/s$", colour=S.INK, ha="center", style="normal",
               size=6.4 * variant.scale)

    # the second ordering variable, tying this figure to D4.6 without a word
    axc.barh(range(len(t)), t.circ.fillna(0.0), height=0.42, zorder=3,
             color=[S.BRONZE if c else S.INK for c in t.clears])
    axc.set_xlim(0, 1.05)
    axc.set_xticks([0, 1])
    axc.set_xticklabels(["0", "1"], fontsize=6.0 * variant.scale)
    axc.set_ylim(ax.get_ylim())
    axc.set_yticks([])
    axc.minorticks_off()
    axc.spines["left"].set_visible(False)
    axc.spines["bottom"].set_color(S.RULE)
    S.annotate(axc, 0.5, 1.015, "circularity", colour=S.INK, ha="center", style="normal",
               size=6.4 * variant.scale)


    if variant.name == "thesis":
        S.panel_letter(ax, "A", dx_mm=36.0)
        S.panel_letter(axn, "B", dx_mm=36.0, dy_mm=1.0)

    return S.save(fig, "fig-5-5-lag", variant)


# ---------------------------------------------------------------------------- D6.1


@renders("6.1", "crocker")
def crocker(variant: S.Variant) -> str:
    """The trajectory as one surface. Spec: D6-1-crocker.md.

    A CROCKER plot is a (time x scale) field of Betti numbers, so the form is a surface
    and there is no alternative; the design work is entirely in not letting it look like
    a heatmap from a library.

    Two rows because the contraction of section 3.3.1 appears on the raw scale axis as
    the whole band of live features sliding downwards, and is absent from the normalised
    one. The rows do not share a vertical axis; they are in different units, which is the
    error the chapter is about.
    """
    S.use(variant)
    d = _load("fig-6-1-crocker.csv")
    regimes = [("reference", r"$p=113$, wd $0.1$"), ("canonical", r"$p=97$, wd $1.0$")]
    axes_rows = [("raw", "raw scale"), ("normalised", r"scale $\div\, s$")]
    cmap = S.sequential(floor=0.32)
    # the top of the scale is a handful of cells; framing on the maximum would render the
    # whole surface as paper. Clipped at the 95th percentile of the live cells, and the
    # colourbar carries the arrow that says so.
    live = d.betti1[d.betti1 > 0]
    vmax = float(np.percentile(live, 90))
    # ... and the ramp is compressed at the top, because the median live cell holds eleven
    # features against a ceiling of fifty: on a linear ramp the surface the chapter is
    # about would sit in the palest fifth of it.
    norm = PowerNorm(gamma=0.48, vmin=0.5, vmax=vmax)

    fig = S.figure(S.FULL, S.RATIOS["standard"], variant)
    gs = fig.add_gridspec(2, 2, left=0.090, right=0.980, bottom=0.215, top=0.885,
                          hspace=0.22, wspace=0.10)

    mesh = None
    for ci, (regime, subtitle) in enumerate(regimes):
        for ri, (axis, rowname) in enumerate(axes_rows):
            ax = fig.add_subplot(gs[ri, ci])
            cell = d[(d.regime == regime) & (d.axis == axis)]
            grid = cell.pivot(index="scale", columns="step", values="betti1")
            mesh = ax.pcolormesh(grid.columns.values, grid.index.values, grid.values,
                                 cmap=cmap, norm=norm, shading="nearest", rasterized=True)
            tg = cell.t_g.dropna()
            if not tg.empty:
                ax.axvline(float(tg.iloc[0]), color=S.SIENNA, lw=S.HAIRLINE, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(max(grid.columns.min(), 1), grid.columns.max())
            ax.minorticks_off()
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=1.8, width=S.HAIRLINE)
            if ri == 0:
                ax.set_xticklabels([])
                S.panel_title(ax, f"{regime}   {subtitle}", pad=7)
            else:
                ax.set_xlabel("training step")
            if ci == 0:
                ax.set_ylabel(rowname)
            else:
                ax.set_yticklabels([])
            if variant.name == "thesis":
                S.panel_letter(ax, "ABCD"[2 * ri + ci], dx_mm=8.5 if ci == 0 else 3.0,
                               dy_mm=0.5)

    bar = fig.add_axes([0.360, 0.075, 0.280, 0.022])
    colours = fig.colorbar(mesh, cax=bar, orientation="horizontal", extend="max",
                           ticks=[1, 5, 10, 20, 30, 40])
    colours.outline.set_visible(False)
    bar.set_xlabel(r"live $H_1$ features", labelpad=2)
    bar.tick_params(length=1.8, width=S.HAIRLINE, pad=1.5)
    # absence is paper, and paper is not on the ramp, so it gets its own swatch
    bar.add_patch(plt.Rectangle((-0.075, 0.0), 0.045, 1.0, transform=bar.transAxes,
                                facecolor=S.PAGE, edgecolor=S.RULE, lw=S.HAIRLINE,
                                clip_on=False, zorder=4))
    bar.text(-0.0525, -0.55, "0", transform=bar.transAxes, ha="center", va="top",
             color=S.INK, fontsize=6.4 * variant.scale)

    return S.save(fig, "fig-6-1-crocker", variant)


# ---------------------------------------------------------------------------- D5.6


@renders("5.6", "interventions")
def interventions(variant: S.Variant) -> str:
    """The interventions are architecture-sensitive. Spec: D5-6-interventions.md.

    Every seed is drawn, because the seed-level outcomes are the finding: a mean over
    five transformer seeds of which four diverge is a number describing no run.

    Two failures are distinguished and must stay distinguished. A run that *memorises*
    holds train accuracy at one and test accuracy at the floor to its budget; a run that
    *diverges* stops, and is drawn stopping, with an open marker at the last finite step.

    Four arms recur across both panels and all four end in the same corner of each, so
    they are named once in the ruled block --- which sits in the lower right, where the
    paired strip's own width leaves a corner that would otherwise be empty.
    """
    S.use(variant)
    d = _load("fig-5-6-interventions.csv")
    arms = {
        ("softmax_ce", "adamw", 0.001): ("weight decay", S.RULE, 0.7, "solid"),
        ("softmax_ce", "orthograd_adamw", 0.01): (r"$\perp$Grad,  $10^{-2}$", S.INK, 0.9, "solid"),
        ("stablemax_ce", "orthograd_adamw", 0.001):
            (r"StableMax $+\perp$Grad,  $10^{-3}$", S.BRONZE, 1.0, "solid"),
        ("stablemax_ce", "adamw", 0.01): (r"StableMax,  $10^{-2}$", S.SLATE, 0.9, (0, (4, 2))),
    }

    fig = S.figure(S.FULL, 0.700, variant)
    gs = fig.add_gridspec(1, 2, left=0.068, right=0.988, bottom=0.545, top=0.935, wspace=0.075)
    panels = {}
    for ci, model in enumerate(("mlp", "transformer")):
        ax = fig.add_subplot(gs[0, ci])
        panels[model] = ax
        sub = d[d.model == model]
        for key, (_name, colour, weight, dash) in arms.items():
            loss, optimizer, lr = key
            arm = sub[(sub.loss == loss) & (sub.optimizer == optimizer) & (sub.lr == lr)]
            if arm.empty:
                continue
            for run, g in arm.groupby("run"):
                g = g.sort_values("step").dropna(subset=["test_acc"])
                ax.plot(g.step, g.test_acc, color=colour, lw=weight, ls=dash, zorder=3,
                        alpha=0.85)
                if bool(arm[arm.run == run].diverged.iloc[0]):
                    ax.scatter([g.step.iloc[-1]], [g.test_acc.iloc[-1]], s=16, marker="o",
                               facecolors="none", edgecolors=colour, linewidths=0.8, zorder=5)
        _logx(ax, 100)
        ax.set_xlim(0, sub.step.max())
        ax.set_ylim(-0.03, 1.06)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("training step")
        S.range_frame(ax, y=(0, 1))
        S.panel_title(ax, "MLP" if model == "mlp" else "transformer", pad=7)
        if ci == 0:
            ax.set_yticklabels(["0", "0.5", "1"])
            ax.set_ylabel("test accuracy")
        else:
            ax.set_yticklabels([])

    # how each bundle ends, as a tally of named outcomes. Set in ink rather than in an
    # arm's colour: these are counts over the panel, and a coloured tally would claim to
    # belong to one series.
    for model, tallies in (("mlp", (("memorising", 3), ("diverged", 1))),
                           ("transformer", (("diverged", 4), ("memorising", 5)))):
        for row, (name, count) in enumerate(tallies):
            S.annotate(panels[model], 0.030, 0.640 - 0.115 * row, f"{name}  {count}",
                       colour=S.INK, size=6.8 * variant.scale, style="normal")

    # the paired comparison the chapter turns on: two MLP arms that grok equally fast
    lower = fig.add_gridspec(1, 3, width_ratios=[0.86, 0.125, 0.63], left=0.235, right=0.988,
                             bottom=0.120, top=0.325, wspace=0.070)
    strip = fig.add_subplot(lower[0, 0])
    bars = fig.add_subplot(lower[0, 1])
    pairs = [
        _condition_row(block="intervention", model="mlp", loss="stablemax_ce",
                       optimizer="orthograd_adamw"),
        _condition_row(block="intervention", model="mlp", loss="softmax_ce",
                       optimizer="orthograd_adamw"),
    ]
    labels = [r"StableMax $+\perp$Grad", r"$\perp$Grad"]
    colours = [S.BRONZE if r.verdict == "above" else S.INK for r in pairs]

    S.null_band(strip, 0.75, 1.32, horizontal=False)
    for i, (row, colour) in enumerate(zip(pairs, colours, strict=True)):
        strip.plot([row.lo, row.hi], [i, i], color=colour, lw=S.DATA, solid_capstyle="butt",
                   zorder=3)
        strip.scatter([row.med], [i], s=20, marker="o", zorder=4, linewidths=0.8,
                      facecolors=colour if row.verdict == "above" else "none", edgecolors=colour)
    strip.set_xscale("log")
    strip.set_xlim(0.55, 14.0)
    strip.set_ylim(1.7, -0.7)
    strip.set_yticks([0, 1])
    strip.set_yticklabels([f"{lab}    $t_g$ {r.t_g:,.0f}" for lab, r in zip(labels, pairs,
                                                                           strict=True)],
                          fontsize=6.8 * variant.scale)
    strip.tick_params(axis="y", length=0)
    strip.set_xticks([1, 2, 5, 10])
    strip.set_xticklabels(["1", "2", "5", "10"])
    strip.minorticks_off()
    strip.spines["left"].set_visible(False)
    strip.spines["bottom"].set_color(S.RULE)
    strip.set_xlabel(r"plateau $\div$ baseline,   normalised $H_1^{\max}$")

    S.sparkline(bars, [r.circularity for r in pairs], colours=colours, ticks=(0, 1),
                label="circularity", height=0.30)
    bars.set_ylim(strip.get_ylim())

    box = fig.add_subplot(lower[0, 2])
    rect = box.get_position()
    box.remove()

    def swatch(colour, weight, dash):
        def draw(ax, y):
            ax.plot([0.045, 0.20], [y, y], color=colour, lw=weight, ls=dash, clip_on=False)
        return draw

    def diverged(ax, y):
        ax.plot([0.1225], [y], marker="o", ms=3.6, mfc="none", mec=S.INK, mew=0.8,
                clip_on=False)

    S.key(fig, (rect.x0, rect.y0 - 0.042, rect.width, rect.height + 0.118),
          [(name, swatch(colour, weight, dash)) for name, colour, weight, dash in arms.values()]
          + [("diverged", diverged)],
          heading="arm")

    if variant.name == "thesis":
        S.panel_letter(panels["mlp"], "A")
        S.panel_letter(panels["transformer"], "B", dx_mm=3.0)
        S.panel_letter(strip, "C", dx_mm=27.0, dy_mm=0.5)

    return S.save(fig, "fig-5-6-interventions", variant)


# ---------------------------------------------------------------------------- D6.2


@renders("6.2", "velocity")
def velocity(variant: S.Variant) -> str:
    """Topological speed, and a silent null. Spec: D6-2-velocity.md.

    Three panels rather than two: the third is the permuted-label control and it exists
    to show nothing, which is the degenerate-panel device and half the argument.

    The rate is plotted, never the raw distance. The snapshot grid is logarithmic outside
    the dense window, so a consecutive-distance series confounds reorganisation with
    sampling interval, and this thesis has already had one detector artefact.
    """
    S.use(variant)
    d = _load("fig-6-2-velocity.csv")
    panels = [("reference", r"reference   $p=113$, wd $0.1$"),
              ("canonical", r"canonical   $p=97$, wd $1.0$"),
              ("permuted", "permuted labels   null")]

    fig = S.figure(S.FULL, 0.425, variant)
    gs = fig.add_gridspec(1, 3, left=0.076, right=0.992, bottom=0.168, top=0.868, wspace=0.095)

    lo = float(d[d.rate > 0].rate.min())
    hi = float(d.rate.max())
    for ci, (panel, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, ci])
        sub = d[d.panel == panel]
        for _, g in sub.groupby("run"):
            g = g.sort_values("step")
            ax.plot(g.step, g.rate, color=S.RULE, lw=0.5, zorder=2)
        median = sub.groupby("step").rate.median()
        ax.plot(median.index, median.values, color=S.INK, lw=1.15, zorder=4)

        tg = sub.t_g.dropna()
        if not tg.empty:
            ax.axvline(float(tg.median()), color=S.SIENNA, lw=S.HAIRLINE, zorder=3)
            S.annotate(ax, float(tg.median()), 0.02, r"  $t_g$", colour=S.SIENNA, ha="left",
                       style="normal", size=6.4 * variant.scale,
                       transform=ax.get_xaxis_transform())

        _logx(ax, 100)
        ax.set_yscale("log")
        ax.set_xlim(0, sub.step.max())
        ax.set_ylim(lo * 0.75, hi * 1.4)
        ax.set_yticks([1e-4, 1e-3, 1e-2])
        ax.minorticks_off()
        ax.set_xlabel("training step")
        S.range_frame(ax)
        S.panel_title(ax, title, pad=7)
        if ci == 0:
            ax.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$"])
            ax.set_ylabel(r"$W_{\mathrm{sw}}$ per step")
        else:
            ax.set_yticklabels([])
        if variant.name == "thesis":
            S.panel_letter(ax, "ABC"[ci], dx_mm=8.0 if ci == 0 else 3.0, dy_mm=1.0)

    return S.save(fig, "fig-6-2-velocity", variant)


# ---------------------------------------------------------------------------- D6.3


@renders("6.3", "phdim")
def phdim(variant: S.Variant) -> str:
    """The dimension of the optimisation path. Spec: D6-3-phdim.md.

    A negative is harder to draw than a positive and drawn badly it looks like a failed
    positive, so the form has to make flatness the visible subject. Five conditions in
    sequence with the null last, on a fixed generous vertical range: auto-scaling would
    magnify noise into structure, which is exactly the error the section warns about.

    The sixth panel states the claim being tested in its own currency. Birdal et al.
    report that this dimension tracks the generalisation gap; here the gap runs from zero
    to one and the dimension does not move.
    """
    S.use(variant)
    d = _load("fig-6-3-phdim.csv")
    conditions = [
        ("transformer_add113_f0.3_wd0.1_dense", r"reference   $p=113$, wd $0.1$"),
        ("transformer_add113_f0.3_wd1.0_dense", r"control   $p=113$, wd $1.0$"),
        ("transformer_add97_f0.3_wd1.0_dense", r"canonical   $p=97$, wd $1.0$"),
        ("transformer_compose120_f0.6_wd1.0_softmax_ce", r"$S_5$   no circle"),
        ("transformer_add97_permuted_dense", "permuted labels   null"),
    ]
    ylim = (0.95, 1.65)

    fig = S.figure(S.FULL, 0.735, variant)
    gs = fig.add_gridspec(2, 3, left=0.070, right=0.920, bottom=0.115, top=0.895,
                          hspace=0.40, wspace=0.105)
    order = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    terminal = {}
    for (prefix, title), (ri, ci) in zip(conditions, order, strict=True):
        ax = fig.add_subplot(gs[ri, ci])
        sub = d[d.run.str.startswith(prefix)]
        for _, g in sub.groupby("run"):
            g = g.sort_values("step")
            ax.plot(g.step, g.ph_dim, color=S.RULE, lw=0.5, zorder=2)
        median = sub.groupby("step").ph_dim.median()
        ax.plot(median.index, median.values, color=S.INK, lw=1.15, zorder=4)
        terminal[prefix] = float(median.iloc[-1])

        tg = sub.t_g.dropna()
        if not tg.empty:
            ax.axvline(float(tg.median()), color=S.SIENNA, lw=0.6, zorder=3)

        lo, hi = float(sub.step.min()), float(sub.step.max())
        ax.set_xscale("log")
        ax.set_xlim(lo * 0.85, hi * 1.05)
        ax.set_ylim(*ylim)
        ax.set_xticks([t for t in (1e3, 1e4, 1e5) if lo <= t <= hi])
        ax.set_yticks([1.0, 1.2, 1.4, 1.6])
        ax.minorticks_off()
        S.range_frame(ax, x=(lo, hi), y=ylim)
        S.panel_title(ax, title, pad=7)
        S.annotate(ax, 0.965, 0.86, f"{median.iloc[-1]:.2f}", ha="right",
                   size=7.4 * variant.scale)
        if ci == 0:
            ax.set_yticklabels(["1.0", "1.2", "1.4", "1.6"])
            ax.set_ylabel(r"$\dim_{\mathrm{PH}}$")
        else:
            ax.set_yticklabels([])
        if ri == 1:
            ax.set_xlabel("training step")
        if variant.name == "thesis":
            S.panel_letter(ax, "ABCDE"[order.index((ri, ci))],
                           dx_mm=8.0 if ci == 0 else 3.0, dy_mm=0.5)

    # the claim in Birdal's own currency: dimension against the gap it is said to track
    ax = fig.add_subplot(gs[1, 2])
    gaps = {"transformer_add113_f0.3_wd0.1_dense": 0.0,
            "transformer_add113_f0.3_wd1.0_dense": 0.0,
            "transformer_add97_f0.3_wd1.0_dense": 0.0,
            "transformer_compose120_f0.6_wd1.0_softmax_ce": 0.0,
            "transformer_add97_permuted_dense": 1.0}
    medians: dict[float, list[float]] = {}
    for prefix, gap in gaps.items():
        sub = d[d.run.str.startswith(prefix)]
        values = [float(g.sort_values("step").ph_dim.iloc[-1]) for _, g in sub.groupby("run")]
        ax.scatter(np.full(len(values), gap) + np.linspace(-0.05, 0.05, len(values)), values,
                   s=13, marker="o", facecolors="none", edgecolors=S.INK, linewidths=0.5,
                   zorder=3)
        ax.scatter([gap], [np.median(values)], s=24, marker="o", color=S.BRONZE, zorder=5,
                   linewidths=0)
        medians.setdefault(gap, []).extend(values)
    for gap, values in medians.items():
        S.direct_label(ax, gap, float(np.median(values)), f"{np.median(values):.2f}",
                       S.BRONZE, dx=7 if gap == 0 else -7, size=7.0 * variant.scale,
                       ha="left" if gap == 0 else "right")
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticks([1.0, 1.2, 1.4, 1.6])
    # this panel is a scatter in a grid of series, and it is two columns from the only
    # labelled vertical axis, so it carries its own on the side it is read from
    ax.set_yticklabels(["1.0", "1.2", "1.4", "1.6"])
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_ticks_position("right")
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel(r"$\dim_{\mathrm{PH}}$", rotation=-90, labelpad=11)
    ax.set_xlabel("generalisation gap")
    S.range_frame(ax, x=(0, 1), y=ylim)
    ax.spines["right"].set_bounds(*ylim)
    ax.spines["right"].set_color(S.RULE)
    S.panel_title(ax, "terminal dimension", pad=7)
    if variant.name == "thesis":
        S.panel_letter(ax, "F", dx_mm=3.0, dy_mm=0.5)

    return S.save(fig, "fig-6-3-phdim", variant)


# ------------------------------------------------------------------------------ cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--variant", choices=["thesis", "slide", "both"], default="thesis")
    parser.add_argument("--out", type=Path, help="output root (default: results/figures/generated)")
    args = parser.parse_args()
    if args.out:
        S.OUTPUT_ROOT = args.out

    variants = {"thesis": [S.THESIS], "slide": [S.SLIDE], "both": [S.THESIS, S.SLIDE]}[args.variant]
    wanted = args.only or sorted(RENDERERS)
    for number in wanted:
        fn = RENDERERS.get(number)
        if fn is None:
            print(f"  {number:5s} no renderer")
            continue
        for v in variants:
            try:
                path = fn(v)
            except FileNotFoundError as exc:
                print(f"  {number:5s} {fn.figure_name:14s} skipped — missing {exc}")
                break
            print(f"  {number:5s} {fn.figure_name:14s} {v.name:6s} -> {path}")


if __name__ == "__main__":
    import matplotlib as mpl  # noqa: F401  (used above for the colormap helper)

    main()
