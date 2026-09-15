"""Render the presentation figures from the same tidy CSVs the thesis figures are drawn from.

A thesis plate would need 217 mm to keep its type legible on a slide, so each slide figure is a
separate composition drawn at exactly the size its frame gives it. Data, style and devices come
from ``render`` and ``style``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.transforms import blended_transform_factory

from analysis.figures import render as R
from analysis.figures import style as S

# Beside the deck when it is there, into the repository otherwise.
_DECK = Path("../LaTeX/Presentation/figures/generated")
OUTPUT_ROOT = _DECK if _DECK.parent.parent.is_dir() else Path("results/figures/talk")

BAND = (0.75, 1.32)  # the normalised-maximum null band, §4.4
RENDERERS: dict[str, Callable] = {}


def draws(number: str, name: str):
    def register(fn):
        fn.number, fn.figure_name = number, name
        RENDERERS[number] = fn
        return fn

    return register


def _canvas(ratio: float = 64.0 / 148.0, *, plate: bool = False):
    S.use(S.TALK)
    width = S.TALK_PLATE_W if plate else S.TALK_W
    return S.figure(width, ratio, S.TALK)


WORDS = ("zero", "one", "two", "three", "four", "five",
         "six", "seven", "eight", "nine", "ten")


def _words(n: int) -> str:
    return WORDS[n]


def _pt(factor: float) -> float:
    return factor * S.TALK.scale


def _save(fig, name: str, out: Path | None) -> str:
    return S.save(fig, name, S.TALK, out_dir=out or OUTPUT_ROOT, svg=False, suffix="")


def _stages(out: Path | None, name: str, count: int, draw: Callable[[int], object]) -> str:
    """A figure that builds on the slide: one file per stage, the last under the plain name.

    Every stage runs the same drawing code with a stage number, so an intermediate state
    cannot disagree with the finished figure, and the finished figure is the one the deck
    and the checklist already know by name.
    """
    for stage in range(1, count + 1):
        suffix = "" if stage == count else f"-{chr(96 + stage)}"
        path = _save(draw(stage), f"{name}{suffix}", out)
    return path


def _accuracy_axis(ax, *, end: float, label: bool = True) -> None:
    R._logx(ax, 100)
    ax.set_xlim(0, end)
    ax.set_ylim(-0.03, 1.06)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"] if label else [])
    ax.set_xlabel("training step")
    S.range_frame(ax, y=(0, 1))


# I. the question


@draws("01", "phenomenon")
def phenomenon(out: Path | None) -> str:
    """Fitting and generalising, two orders of magnitude apart."""
    return _stages(out, "talk-01-phenomenon", 3, _phenomenon)


def _phenomenon(stage: int):
    """1 the run to the end of the plateau · 2 the transition · 3 the ratio between them."""
    d = R._load("fig-1-1-hero.csv")
    t_c, t_g = float(d.t_c.dropna().iloc[0]), float(d.t_g.dropna().iloc[0])
    acc = d.dropna(subset=["test_acc"]).sort_values("step")
    end = float(acc.step.max())
    # Stage one truncates the series, never the axis: the run so far, on the axis it will
    # be read against, so nothing moves when the rest of it arrives.
    shown = acc if stage > 1 else acc[acc.step <= 0.62 * t_g]

    fig = _canvas()
    ax = fig.add_axes([0.098, 0.155, 0.892, 0.815])

    ax.axvline(t_c, color=S.RULE, lw=S.HAIRLINE, zorder=1)
    if stage > 1:
        ax.axvspan(t_c, t_g, facecolor=S.INK, alpha=0.04, lw=0, zorder=0)
        ax.axvline(t_g, color=S.SIENNA, lw=S.HAIRLINE, zorder=1)
    ax.plot(shown.step, shown.train_acc, color=S.RULE, lw=1.0, zorder=2)
    ax.plot(shown.step, shown.test_acc, color=S.INK, lw=1.4, zorder=4)

    _accuracy_axis(ax, end=end)
    S.range_frame(ax, x=(0.0, end), y=(0, 1))  # the spine spans the run in every stage
    ax.set_ylabel("accuracy")
    S.direct_label(ax, 900, 1.00, "train", S.BRONZE, dy=-4, va="top", size=_pt(7.6))
    S.direct_label(ax, 900, 0.305, "test", S.INK, dy=4, va="bottom", size=_pt(7.6))

    trans = ax.get_xaxis_transform()
    ax.text(t_c * 1.35, 0.055, r"$t_c$ = 200", transform=trans, ha="left", va="bottom",
            color=S.INK, fontsize=_pt(7.2))
    if stage > 1:
        ax.text(t_g * 0.62, 0.055, r"$t_g$ = 28,600", transform=trans, ha="right", va="bottom",
                color=S.SIENNA, fontsize=_pt(7.2), zorder=6)

    if stage > 2:
        y = 0.660
        ax.annotate("", xy=(t_c, y), xytext=(t_g, y), xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle="<->", color=S.BRONZE, lw=S.HAIRLINE,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.text(t_c * 3.4, y + 0.030, r"$143\times$", transform=trans, ha="center",
                va="bottom", color=S.BRONZE, style="italic", fontsize=_pt(8.6))

    return fig


@draws("02", "leak")
def leak(out: Path | None) -> str:
    """The pre-grokking plateau is commutativity, and it is worth exactly the train fraction."""
    d = R._load("fig-1-1-hero.csv")
    t_g = float(d.t_g.dropna().iloc[0])
    acc = d.dropna(subset=["test_acc"]).sort_values("step")
    novel = d.dropna(subset=["test_acc_novel"]).sort_values("step")
    fraction = 0.30
    predicted = fraction * (1 - 1 / 97)  # the leak model on Z_97: every off-diagonal pair commutes
    plateau = R._claims()["modular_addition_measured_plateau"]["median"]

    fig = _canvas()
    ax = fig.add_axes([0.098, 0.155, 0.892, 0.815])

    ax.axhline(fraction, color=S.BRONZE, lw=S.HAIRLINE, ls=(0, (4, 2.5)), zorder=1)
    ax.axvline(t_g, color=S.SIENNA, lw=S.HAIRLINE, zorder=1)
    ax.plot(acc.step, acc.test_acc, color=S.INK, lw=1.4, zorder=4)
    ax.plot(novel.step, novel.test_acc_novel, color=S.INK, lw=1.1, ls=(0, (4, 2)), zorder=3)

    _accuracy_axis(ax, end=float(acc.step.max()))
    ax.set_ylabel("test accuracy")
    # named left of the two transient collapses, where each series is alone (master §4.7)
    S.direct_label(ax, 300, 0.305, "as reported", S.INK, dy=6, va="bottom", size=_pt(7.6))
    S.direct_label(ax, 300, 0.010, "novel pairs only", S.INK, dy=6, va="bottom", size=_pt(7.6))
    S.annotate(ax, 0.022, fraction + 0.028, rf"$f = {fraction:g}$", ha="left", size=_pt(7.6),
               transform=ax.get_yaxis_transform())

    S.value(ax, 0.030, 0.905, f"{plateau:.3f}", "measured plateau", colour=S.INK)
    S.value(ax, 0.230, 0.905, f"{predicted:.3f}", "predicted")

    return _save(fig, "talk-02-leak", out)


# II. measuring shape


@draws("03", "reproduction")
def reproduction(out: Path | None) -> str:
    """It reproduces, five seeds of five, across recipes and architectures."""
    d = R._load("fig-4-1-reproduction.csv")
    panels = [("canonical transformer", r"canonical   $p=97$"),
              ("reference transformer", r"reference   $p=113$"),
              ("canonical MLP", r"MLP   $p=97$")]

    fig = _canvas()
    gs = fig.add_gridspec(1, 3, left=0.098, right=0.990, bottom=0.200, top=0.860, wspace=0.075)

    for ci, (panel, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, ci])
        sub = d[d.panel == panel]
        tg = sorted(sub.groupby("run").t_g.first().dropna())

        for _, g in sub.groupby("run"):
            g = g.sort_values("step")
            ax.plot(g.step, g.train_acc, color=S.RULE, lw=0.5, zorder=2)
            ax.plot(g.step, g.test_acc, color=S.INK, lw=0.6, alpha=0.55, zorder=3)
        median = sub.groupby("step").test_acc.median()
        ax.plot(median.index, median.values, color=S.INK, lw=1.3, zorder=5)

        _accuracy_axis(ax, end=float(sub.step.max()), label=ci == 0)
        S.seed_comb(ax, tg, height=0.075)
        S.panel_title(ax, title, pad=7)
        R._seed_tally(ax, sub.run.nunique(), len(tg), size=_pt(7.0),
                      name="seeds grokked" if ci == 0 else "", pitch=0.036)
        if ci == 0:
            ax.set_ylabel("accuracy")
        if ci == 2:  # named on the panel whose curves are clean, master §4.7
            S.direct_label(ax, 1500, 1.000, "train", S.BRONZE, dy=-4, va="top", size=_pt(7.0))
            S.direct_label(ax, 1500, 0.000, "test", S.INK, dy=5, va="bottom", size=_pt(7.0))
        # the comb says which seeds grokked; the range says when, set clear of both
        S.annotate(ax, 0.045, 0.560, f"{min(tg) / 1000:.1f}–{max(tg) / 1000:.1f}k steps",
                   colour=S.SIENNA, style="normal", size=_pt(7.0))

    return _save(fig, "talk-03-reproduction", out)


def _signature_row(fig, gs, column: str, *, ylabel: str, scale_trace: bool) -> None:
    """One row of the raw/normalised comparison: the same observable in both regimes."""
    d = R._load("fig-4-2-signature.csv")
    titles = {"reference": r"reference   $p=113$, wd $0.1$",
              "canonical": r"canonical   $p=97$, wd $1.0$"}
    collapse = {"reference": r"scale falls $26\times$", "canonical": r"scale falls $155\times$"}
    ratios = {("reference", "h1_max_persistence"): "1.40  [0.70, 2.04]",
              ("canonical", "h1_max_persistence"): "0.91  [0.85, 1.21]",
              ("reference", "h1_max_persistence_normalised"): "2.90  [2.20, 5.77]",
              ("canonical", "h1_max_persistence_normalised"): "1.09  [0.88, 1.21]"}

    axes = []
    for ci, regime in enumerate(("reference", "canonical")):
        ax = fig.add_subplot(gs[0, ci])
        axes.append(ax)
        sub = d[d.panel == regime]
        tg = sorted(sub.groupby("run").t_g.first().dropna().unique())
        tg_med = float(np.median(tg))

        ax.axvspan(0.5 * tg_med, 0.9 * tg_med, facecolor=S.INK, alpha=0.045, lw=0, zorder=0)
        ax.axvspan(1.2 * tg_med, sub.step.max(), facecolor=S.INK, alpha=0.045, lw=0, zorder=0)
        for _, g in sub.groupby("run"):
            g = g.sort_values("step")
            ax.plot(g.step, g[column], color=S.RULE, lw=0.55, zorder=2)
        med = sub.groupby("step")[column].median()
        ax.plot(med.index, med.values, color=S.INK, lw=S.EMPHASIS, zorder=4)

        if scale_trace:
            trace = sub.groupby("step").pointcloud_scale.median()
            twin = ax.twinx()
            twin.plot(trace.index, trace.values, color=S.RULE, lw=0.9, ls=(0, (1, 2.2)),
                      zorder=1)
            twin.set_yscale("log")
            # the trace is context: keep it in the upper half, clear of the series the panel
            # is about and clear of the axis at the foot
            twin.set_ylim(trace.min() * 0.05, trace.max() * 2.2)
            twin.yaxis.set_visible(False)
            for side in twin.spines.values():
                side.set_visible(False)
            S.annotate(ax, 0.035, 0.055, collapse[regime], size=_pt(7.2))

        R._logx(ax, 100)
        S.seed_comb(ax, tg)
        ax.set_xlim(0, sub.step.max() * 1.02)
        ax.margins(y=0.26 if scale_trace else 0.10)
        ax.set_xlabel("training step")
        S.panel_title(ax, titles[regime], pad=7)
        S.value(ax, 0.975 if scale_trace else 0.035, 0.925, ratios[(regime, column)],
                r"plateau $\div$ baseline" if ci == 0 else "",
                ha="right" if scale_trace else "left")
        if ci == 0:
            ax.set_ylabel(ylabel)

    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for ci, ax in enumerate(axes):
        ax.set_ylim(lo, hi)
        if ci:
            ax.set_yticklabels([])
        S.range_frame(ax)
    if scale_trace:  # named in Bronze so the label ties to the dotted trace, master §4.6a
        S.annotate(axes[0], 0.040, 0.925, "cloud scale, $s$", size=_pt(7.2))
    tg = sorted(d[d.panel == "reference"].groupby("run").t_g.first().dropna().unique())
    axes[0].text(float(np.median(tg)) * 2.6, axes[0].get_ylim()[0], r"$t_g$", color=S.SIENNA,
                 fontsize=_pt(7.0), ha="left", va="bottom")


@draws("04", "scale")
def scale(out: Path | None) -> str:
    """Raw persistence is measured in units of a cloud that contracts by two orders."""
    fig = _canvas()
    gs = fig.add_gridspec(1, 2, left=0.112, right=0.986, bottom=0.200, top=0.860, wspace=0.075)
    _signature_row(fig, gs, "h1_max_persistence", ylabel=r"raw   $H_1^{\max}$", scale_trace=True)
    return _save(fig, "talk-04-scale", out)


@draws("04b", "flip")
def flip(out: Path | None) -> str:
    """What the scale correction buys: the same sixteen conditions, read twice."""
    return _stages(out, "talk-04b-flip", 2, _flip)


def _flip(stage: int):
    """1 the sixteen conditions on the raw reading · 2 each carried to the corrected one."""
    d = R._load("fig-4-4-robustness.csv")
    d = d[d.block == "condition"].sort_values("h1_max_persistence_normalised__med")
    raw, norm = "h1_max_persistence", "h1_max_persistence_normalised"
    bands = {c: (float(d[f"{c}__null_lo"].iloc[0]), float(d[f"{c}__null_hi"].iloc[0]))
             for c in (raw, norm)}

    fig = _canvas()
    ax = fig.add_axes([0.150, 0.180, 0.700, 0.660])

    # the two readings are two vertical axes; a condition is the line between them, so the
    # correction is the slope rather than a claim about it
    for x, column in ((0.0, raw), (1.0, norm)):
        lo, hi = bands[column]
        ax.add_patch(plt.Rectangle((x - 0.075, lo), 0.150, hi - lo, facecolor=S.RULE, alpha=0.22,
                               edgecolor="none", zorder=0))
        ax.plot([x, x], [0.045, 4.2], color=S.RULE, lw=S.HAIRLINE, zorder=0.5)

    cleared = 0
    for row in d.itertuples():
        a, b = getattr(row, f"{raw}__med"), getattr(row, f"{norm}__med")
        verdict = getattr(row, f"{norm}__verdict")
        above = verdict == "above"
        cleared += above
        if stage < 2:  # the raw reading alone, in its own verdict's colours
            first = S.SLATE if getattr(row, f"{raw}__verdict") == "below" else S.INK
            ax.scatter([0.0], [a], s=11, marker="o", zorder=4, linewidths=0.8,
                       edgecolors=first, facecolors="none")
            continue
        colour = S.BRONZE if above else (S.SLATE if verdict == "below" else S.INK)
        ax.plot([0.0, 1.0], [a, b], color=colour, lw=S.DATA if above else S.SECONDARY,
                alpha=1.0 if above else 0.55, zorder=3 if above else 2, solid_capstyle="round")
        for x, v in ((0.0, a), (1.0, b)):
            ax.scatter([x], [v], s=17 if above else 11, marker="o", zorder=4,
                       linewidths=0.8, edgecolors=colour,
                       facecolors=colour if above else "none")

    ax.set_yscale("log")
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(0.042, 4.2)
    ax.set_yticks([0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0.1", "0.3", "1", "3"])
    ax.minorticks_off()
    ax.set_xticks([])
    ax.set_ylabel(r"plateau $\div$ baseline")
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(S.RULE)

    # heads above, counts below the frame, and each band's range on its own outer side, so
    # nothing that names an axis sits where the lines between the two axes run
    for x, head, band, count, side in (
        (0.0, "raw", bands[raw], "none of sixteen clear", -1),
        (1.0, "normalised", bands[norm],
         f"{_words(cleared)} of sixteen clear" if stage > 1 else "", +1),
    ):
        ax.text(x, 4.55, head.upper(), ha="center", va="bottom", color=S.BRONZE,
                fontsize=_pt(7.6))
        ax.text(x, -0.085, count, ha="center", va="top", color=S.INK, fontsize=_pt(7.4),
                transform=ax.get_xaxis_transform())
        ax.text(x + side * 0.105, band[1], f"null {band[0]:.2f}–{band[1]:.2f}",
                ha="left" if side > 0 else "right", va="bottom", color=S.INK, alpha=0.62,
                fontsize=_pt(6.6))

    return fig


@draws("05", "signature")
def signature(out: Path | None) -> str:
    """Divided by a scale intrinsic to the same cloud, one regime rises and one does not."""
    fig = _canvas()
    gs = fig.add_gridspec(1, 2, left=0.112, right=0.986, bottom=0.200, top=0.860, wspace=0.075)
    _signature_row(fig, gs, "h1_max_persistence_normalised",
                   ylabel=r"normalised   $H_1^{\max}/s$", scale_trace=False)
    return _save(fig, "talk-05-signature", out)


# III. what we found

TALK_COLUMNS = ((-0.385, "left"), (-0.300, "left"), (-0.205, "left"), (-0.085, "right"),
                (-0.020, "right"))
TALK_HEADERS = ("arch", "op", "", "wd", "$n$")
# the lag forest carries two MLP intervention arms that the dose alone cannot tell apart
LAG_COLUMNS = ((-0.460, "left"), (-0.385, "left"), (-0.300, "left"), (-0.180, "right"),
               (-0.165, "left"))
LAG_HEADERS = ("arch", "op", "", "wd", "")


def _setting(operation: str, modulus: float, fraction: float) -> str:
    """What sets a condition apart from the canonical task, $p = 97$ at $f = 0.3$."""
    parts = [f"p{int(modulus)}"] if operation != "compose" and modulus != 97 else []
    if fraction != 0.3:
        parts.append(f"f{fraction:g}")
    return " ".join(parts)


def _short_fields(row) -> tuple[str, ...]:
    """Architecture, operation, what differs from the canonical task, dose, and seeds."""
    if row.block == "null model":
        return ("", R._OPERATION[row.operation], "", "", "")
    seeds = f"{int(row.n_runs)}" if row.n_grokked == row.n_runs else \
        f"{int(row.n_grokked)}/{int(row.n_runs)}"
    return ("MLP" if row.model == "mlp" else "tf", R._OPERATION[row.operation],
            _setting(row.operation, row.modulus, row.train_fraction),
            f"{row.weight_decay:g}", seeds)


@draws("06", "robustness")
def robustness(out: Path | None) -> str:
    """Five conditions clear the null band, ten sit inside it and one runs backwards."""
    return _stages(out, "talk-06-robustness", 2, _robustness)


def _robustness(stage: int):
    """1 the sixteen conditions against the null · 2 the circularity that separates them."""
    obs = "h1_max_persistence_normalised"
    d = R._load("fig-4-4-robustness.csv")
    conditions = d[d.block == "condition"].reset_index(drop=True)
    nulls = d[d.block == "null model"].reset_index(drop=True)

    fig = _canvas(73.0 / S.TALK_PLATE_W, plate=True)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[len(conditions), len(nulls) + 0.7], width_ratios=[1.0, 0.145],
        left=0.240, right=0.972, bottom=0.185, top=0.910, hspace=0.185, wspace=0.050)
    ax, axn = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    axc, axcn = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    def colours(frame):
        return [{"above": S.BRONZE, "below": S.SLATE}.get(v, S.INK)
                for v in frame[f"{obs}__verdict"]]

    def forest(a, frame):
        for i, r in frame.iterrows():
            colour = {"above": S.BRONZE, "below": S.SLATE}.get(r[f"{obs}__verdict"], S.INK)
            a.plot([r[f"{obs}__lo"], r[f"{obs}__hi"]], [i, i], color=colour, lw=S.DATA,
                   solid_capstyle="butt", zorder=3)
            a.scatter([r[f"{obs}__med"]], [i], s=17, marker="o", zorder=4, linewidths=0.8,
                      facecolors=colour if r[f"{obs}__verdict"] == "above" else "none",
                      edgecolors=colour)

    size = _pt(5.9)
    for a in (ax, axn):
        S.null_band(a, *BAND, horizontal=False)
        a.set_xscale("log")
        a.set_xlim(0.15, 7.0)
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
        a.minorticks_off()
    forest(ax, conditions)
    forest(axn, nulls)

    ax.set_yticks([])
    ax.set_ylim(len(conditions) - 0.4, -0.6)
    R._condition_columns(ax, [_short_fields(r) for _, r in conditions.iterrows()], size=size,
                         columns=TALK_COLUMNS, headers=TALK_HEADERS, y0=-0.78)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)

    axn.set_yticks([])
    axn.set_ylim(len(nulls) - 0.4, -0.9)
    R._condition_columns(axn, [_short_fields(r) for _, r in nulls.iterrows()], size=size,
                         columns=TALK_COLUMNS, headers=TALK_HEADERS, header=False)
    axn.set_xticks([0.2, 0.5, 1, 2, 5])
    axn.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    axn.set_xlabel(r"plateau $\div$ baseline,   normalised $H_1^{\max}$")
    axn.spines["bottom"].set_color(S.RULE)
    axn.spines["bottom"].set_bounds(float(d[f"{obs}__lo"].min()), float(d[f"{obs}__hi"].max()))
    axn.text(TALK_COLUMNS[0][0], -0.95, "null models", transform=axn.get_yaxis_transform(),
             ha="left", va="bottom", color=S.INK, alpha=0.68, family=S.SMALLCAPS,
             clip_on=False, fontsize=size)

    S.sparkline(axc, conditions.circularity, colours=colours(conditions), ticks=(0, 1),
                label="circularity", height=0.46)
    for row, value in enumerate(conditions.circularity):
        if not np.isfinite(value):
            axc.text(0.06, row, "--", transform=axc.get_yaxis_transform(), ha="left",
                     va="center", color=S.INK, alpha=0.55, clip_on=False, fontsize=size)
    S.sparkline(axcn, nulls.circularity, ticks=(0, 1), height=0.46)
    for a, source in ((axc, ax), (axcn, axn)):
        a.set_ylim(source.get_ylim())
    axc.set_xticklabels([])
    axc.spines["bottom"].set_visible(False)
    axc.tick_params(axis="x", length=0)

    rho = R._claims()["circularity_association"][f"{obs}__ratio"]["spearman_rho"]
    S.annotate(axc, 0.5, -0.055, rf"$\rho = {rho:.2f}$", ha="center", va="top", size=_pt(7.2))

    if stage < 2:  # the column is the answer, and it is worth a click of its own
        axc.set_visible(False)
        axcn.set_visible(False)

    return fig


@draws("07", "circularity")
def circularity(out: Path | None) -> str:
    """What separates the regimes is how circular the solution is, not whether it generalises."""
    return _stages(out, "talk-07-circularity", 2, _circularity)


def _circularity(stage: int):
    """1 seventy-one runs against circularity · 2 the same runs against final accuracy."""
    d = R._load("fig-4-6-circularity.csv").dropna(
        subset=["circularity", "h1_max_persistence_normalised__ratio"])
    y = d["h1_max_persistence_normalised__ratio"]

    fig = _canvas()
    gs = fig.add_gridspec(1, 2, width_ratios=[132, 22], left=0.086, right=0.980, bottom=0.285,
                          top=0.950, wspace=0.150)
    ax, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1], sharey=None)

    S.null_band(ax, *BAND)
    finite = np.isfinite(d.circularity) & (y > 0)
    b, a = np.polyfit(d.circularity[finite], np.log(y[finite]), 1)
    xs = np.linspace(d.circularity.min(), d.circularity.max(), 100)
    ax.plot(xs, np.exp(a + b * xs), color=S.BRONZE, lw=S.DATA, zorder=3)

    decay = d.weight_decay.replace(0.0, 0.05)
    shade = (np.log10(decay) - np.log10(0.05)) / (np.log10(3.0) - np.log10(0.05))
    for marker, selected in (("o", d.model == "transformer"), ("s", d.model == "mlp")):
        idx = selected.values
        for target, xcol in ((ax, d.circularity), (ax2, d["test_acc__final"])):
            target.scatter(xcol[idx], y[idx], marker=marker, s=22,
                           c=S.SEQUENTIAL(0.15 + 0.75 * shade[idx]), edgecolors=S.INK,
                           linewidths=0.4, zorder=5)

    for target in (ax, ax2):
        target.set_yscale("log")
        # the largest ratio is 13.78, so a top of 14 clipped its marker against the axes:
        # a limit has to clear the data by more than the radius of the glyph on it
        target.set_ylim(0.14, 20.0)
        target.set_yticks([0.3, 1, 3, 10])
        target.minorticks_off()
    ax.set_yticklabels(["0.3", "1", "3", "10"])
    ax.set_xlim(0.05, 1.02)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel("terminal Fourier concentration")
    ax.set_ylabel(r"normalised $H_1$ ratio")
    S.range_frame(ax)

    rho, pv = R._spearman(d.circularity, y)
    exponent = int(np.floor(np.log10(pv)))
    S.annotate(ax, 0.028, 0.905, rf"$\rho = {rho:.2f}$", size=_pt(9.0))
    S.annotate(ax, 0.028, 0.815, rf"$p = {pv / 10 ** exponent:.0f}\times 10^{{{exponent}}}$",
               size=_pt(7.6))

    ax2.set_xlim(0.965, 1.035)
    ax2.set_xticks([1.0])
    ax2.set_xticklabels(["1.0"])
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("final\ntest acc.")
    ax2.spines["bottom"].set_bounds(0.97, 1.03)
    ax2.spines["bottom"].set_color(S.RULE)

    S.annotate(ax, 0.030, BAND[1] * 1.10, "null", colour=S.INK, style="normal", size=_pt(6.8),
               transform=ax.get_yaxis_transform())

    def glyph(marker):
        def draw(a, y_):
            a.scatter([0.13], [y_], marker=marker, s=20, c=[S.SEQUENTIAL(0.55)],
                      edgecolors=S.INK, linewidths=0.4, clip_on=False)
        return draw

    def ramp(a, y_):
        for k in range(6):
            a.add_patch(plt.Rectangle((0.055 + 0.036 * k, y_ - 0.060), 0.036, 0.120,
                                      facecolor=S.SEQUENTIAL(0.15 + 0.15 * k), edgecolor="none"))

    # right edge flush with the end of the main axis, which sits at 0.799 of the figure
    S.key(fig, (0.559, 0.318, 0.240, 0.200),
          [("transformer", glyph("o")), ("MLP", glyph("s")), ("weight decay", ramp)],
          heading="run")

    if stage < 2:  # the panel with no variance in it lands harder on its own click
        ax2.set_visible(False)

    return fig


RECIPE_ROWS = (("n_layers", "depth", r"1 $\leftrightarrow$ 2 blocks"),
               ("lr", "learning rate", r"$10^{-3}$ $\leftrightarrow$ $3{\times}10^{-3}$"),
               ("lr+n_layers", "both", ""),
               ("act", "activation", r"ReLU $\leftrightarrow$ GELU"),
               ("batch_size", "batch", r"full $\leftrightarrow$ 512"),
               ("d_mlp", "width", r"512 $\leftrightarrow$ 256"),
               ("eps", r"Adam $\epsilon$", r"$10^{-8}$ $\leftrightarrow$ $10^{-6}$"))
RECIPE_CAUSES = ("n_layers", "lr", "lr+n_layers")


@draws("07b", "recipe")
def recipe(out: Path | None) -> str:
    """Two of six ingredients move circularity the way the regimes differ, from either end."""
    circularity = pd.read_csv(R.DATA.parent / "recipe_cells.csv").set_index(
        ["anchor", "factor"]).circularity
    anchors = {a: float(circularity[a, "—"]) for a in ("canonical", "reference")}

    fig = _canvas()
    ax = fig.add_axes([0.330, 0.170, 0.640, 0.660])
    top = ax.get_xaxis_transform()
    for name, x in anchors.items():
        ax.axvline(x, color=S.RULE, lw=S.HAIRLINE, zorder=1)
        ax.text(x, 1.110, name, transform=top, ha="center", va="bottom", color=S.INK,
                family=S.SMALLCAPS, fontsize=_pt(7.4))
        ax.text(x, 1.040, f"{x:.2f}", transform=top, ha="center", va="bottom", color=S.INK,
                style="italic", fontsize=_pt(6.8))

    # each ingredient starts one arrow at each anchor; a cause sends the two toward each other
    label = ax.get_yaxis_transform()
    for i, (factor, name, setting) in enumerate(RECIPE_ROWS):
        y = i + (0.55 if i >= len(RECIPE_CAUSES) else 0.0)
        cause = factor in RECIPE_CAUSES
        colour = S.BRONZE if cause else S.INK
        for dy, anchor in ((-0.15, "canonical"), (0.15, "reference")):
            start, end = anchors[anchor], float(circularity[anchor, factor])
            ax.annotate("", xy=(end, y + dy), xytext=(start, y + dy),
                        arrowprops=dict(arrowstyle="-|>", color=colour, alpha=1 if cause else 0.5,
                                        lw=S.DATA if cause else S.SECONDARY, shrinkA=0,
                                        shrinkB=0, mutation_scale=6))
            if cause:
                ahead = 0.014 if end > start else -0.014
                ax.text(end + ahead, y + dy, f"{end:.2f}", ha="left" if ahead > 0 else "right",
                        va="center", color=colour, style="italic", fontsize=_pt(6.4))
        ax.text(-0.505, y, name, transform=label, ha="left", va="center", color=S.INK,
                style="italic" if factor == "lr+n_layers" else "normal", fontsize=_pt(7.2))
        ax.text(-0.255, y, setting, transform=label, ha="left", va="center", color=S.INK,
                alpha=0.62, fontsize=_pt(6.4))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(len(RECIPE_ROWS) - 0.05, -0.55)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([])
    ax.set_xlabel("terminal circularity")
    S.range_frame(ax, x=(0.0, 1.0))
    ax.spines["left"].set_visible(False)
    return _save(fig, "talk-07b-recipe", out)


@draws("08", "basis")
def basis(out: Path | None) -> str:
    """A fixed-basis spectral statistic goes blind where persistent homology does not."""
    d = R._load("fig-5-1-basis.csv")
    noise = (0.119, 0.138)
    bases = (("fourier_concentration_group_k5", "solid", 1.35, "discrete log"),
             ("fourier_concentration_k5", (0, (4, 2)), 1.15, "residue axis"))

    fig = _canvas()
    gs = fig.add_gridspec(1, 2, left=0.112, right=0.986, bottom=0.200, top=0.860, wspace=0.075)
    axes = []
    for ci, (operation, name) in enumerate((("mul", r"$a \times b$"), ("div", r"$a \div b$"))):
        ax = fig.add_subplot(gs[0, ci])
        axes.append(ax)
        sub = d[d.operation == operation]
        tg = sorted(sub.groupby("run").t_g.first().dropna())

        ax.axhspan(*noise, facecolor=S.RULE, alpha=0.30, lw=0, zorder=0)
        for row, (column, dash, weight, label) in enumerate(bases):
            for _, g in sub.groupby("run"):
                g = g.sort_values("step")
                ax.plot(g.step, g[column], color=S.RULE, lw=0.5, zorder=2)
            median = sub.groupby("step")[column].median()
            ax.plot(median.index, median.values, color=S.BRONZE, lw=weight, ls=dash, zorder=4)
            # the key carries each basis's terminal concentration, so the mapping and the
            # measurement are one mark and neither lands on the seed spaghetti
            y = 0.940 - 0.105 * row
            ax.plot([0.045, 0.150], [y, y], transform=ax.transAxes, color=S.BRONZE, lw=weight,
                    ls=dash, clip_on=False, zorder=6)
            ax.text(0.175, y, label, transform=ax.transAxes, color=S.INK, va="center",
                    fontsize=_pt(7.2))
            ax.text(0.560, y, f"{median.iloc[-1]:.3f}", transform=ax.transAxes, color=S.BRONZE,
                    va="center", style="italic", fontsize=_pt(7.2))

        R._logx(ax, 100)
        ax.set_xlim(0, sub.step.max())
        ax.set_ylim(0, 0.80)
        ax.set_yticks([0, 0.25, 0.5, 0.75])
        ax.set_xlabel("training step")
        S.seed_comb(ax, tg, height=0.055)
        S.range_frame(ax)
        S.panel_title(ax, name, pad=7)
        if ci == 0:
            ax.set_ylabel("Fourier concentration,  $k = 5$")
        else:
            ax.set_yticklabels([])

    S.annotate(axes[0], 0.030, noise[1] + 0.014, "null", colour=S.INK, style="normal",
               size=_pt(6.8), transform=axes[0].get_yaxis_transform())
    return _save(fig, "talk-08-basis", out)


PID_ATOMS = (("redundant", "redundant", S.RULE),
             ("unique_a", r"unique to $H_1$", S.INK),
             ("unique_b", "unique to Fourier", S.BRONZE),
             ("synergistic", "synergistic", S.SLATE))


@draws("08b", "pid")
def pid(out: Path | None) -> str:
    """Pooled, nothing about the transition is unique to H1; in one regime the zero moves."""
    return _stages(out, "talk-08b-pid", 2, _pid)


def _pid(stage: int):
    """1 the pooled bank · 2 the canonical regime, where the decomposition reverses."""
    d = R._load("fig-5-4-pid.csv")
    d = d[d.estimator == "gaussian_mmi__ratio"]
    regimes = (("pooled", "pooled"), ("canonical", "canonical"))

    fig = _canvas()
    ax = fig.add_axes([0.205, 0.215, 0.765, 0.545])
    size = _pt(7.2)
    for x, (_, name, colour) in zip((0.0, 0.225, 0.480, 0.775), PID_ATOMS, strict=True):
        ax.add_patch(plt.Rectangle((x, 1.235), 0.020, 0.080, transform=ax.transAxes,
                                   clip_on=False, facecolor=colour, edgecolor="none"))
        ax.text(x + 0.032, 1.275, name, transform=ax.transAxes, ha="left", va="center",
                color=S.INK, fontsize=size)

    label = ax.get_yaxis_transform()
    for row, (regime, name) in enumerate(regimes[:stage]):
        block = d[d.regime == regime].set_index("atom")
        left = 0.0
        for atom, _, colour in PID_ATOMS:
            width = float(block.nats[atom])
            ax.barh([row], [width], left=left, height=0.46, color=colour, edgecolor=S.PAGE,
                    linewidth=0.6, zorder=3)
            if width <= 0.004:  # a structural zero is a gap in the stack, no taller than the bar
                ax.plot([left, left], [row - 0.23, row + 0.23], color=colour, lw=1.2, zorder=4)
            left += width
        ax.text(-0.025, row - 0.03, name, transform=label, ha="right", va="bottom",
                color=S.INK, family=S.SMALLCAPS, fontsize=size)
        ax.text(-0.025, row + 0.03, f"{int(block.n.iloc[0])} runs", transform=label,
                ha="right", va="top", color=S.INK, alpha=0.62, fontsize=_pt(6.6))

        unique = block.loc["unique_a"]
        if unique.nats <= 0.004:
            S.direct_label(ax, left, row, r"unique to $H_1$:  zero, by construction", S.INK,
                           dx=7, size=_pt(7.0))
        else:
            ax.text(float(block.nats["redundant"]) + unique.nats / 2, row,
                    rf"${unique.nats:+.3f}$ nats,   $p = {unique.null_p:.3f}$", ha="center",
                    va="center", color=S.PAGE, style="italic", fontsize=_pt(7.0), zorder=5)

    ax.set_xlim(0.0, 0.50)
    ax.set_ylim(1.55, -0.55)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4"])
    ax.set_yticks([])
    ax.set_xlabel(r"information about $\log t_g$,   nats")
    S.range_frame(ax, x=(0.0, 0.4))
    ax.spines["left"].set_visible(False)
    return fig


@draws("09", "noncyclic")
def noncyclic(out: Path | None) -> str:
    """Composition in S_5 groks, and the signature is present where no circle can be."""
    return _stages(out, "talk-09-noncyclic", 2, _noncyclic)


def _noncyclic(stage: int):
    """1 the seventeen seeds, in absolute and in rescaled time · 2 the region above every null."""
    d = R._load("fig-5-2-noncyclic.csv")
    band = float(d.null_lo.iloc[0]), float(d.null_hi.iloc[0])
    runs = sorted(d.run.unique(), key=lambda r: d.loc[d.run == r, "t_g"].iloc[0])
    censored = runs[-1]  # groks at 89,700 of a 100,000-step budget: no plateau window
    steps = sorted(d.groupby("run").t_g.first())

    fig = _canvas()
    gs = fig.add_gridspec(1, 2, left=0.104, right=0.938, bottom=0.200, top=0.860, wspace=0.260)
    left, right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    for run in runs:
        g = d[d.run == run].sort_values("step")
        colour = S.SLATE if run == censored else S.INK
        left.plot(g.step, g.test_acc, color=colour, lw=1.0, zorder=3)
    _accuracy_axis(left, end=float(d.step.max()))
    S.seed_comb(left, steps, height=0.06)
    left.set_ylabel("test accuracy")
    S.panel_title(left, r"$S_5$   absolute steps", pad=7)
    R._seed_tally(left, len(runs), len(steps), size=_pt(7.0), name="seeds grokked", y=0.905,
                  pitch=0.032)
    S.annotate(left, 0.050, 0.680,
               f"over {steps[0] / 1000:.1f}–{steps[-1] / 1000:.1f}k steps", colour=S.SIENNA,
               ha="left", va="center", style="normal", size=_pt(7.0))

    S.null_band(right, *band)
    right.axvline(1.0, color=S.SIENNA, lw=S.HAIRLINE, zorder=1)
    for run in runs:
        g = d[d.run == run].sort_values("step").dropna(subset=["step_over_tg"])
        colour = S.SLATE if run == censored else S.INK
        ratio = R._baseline_ratio(g, "h1_total_persistence_normalised").values
        right.plot(g.step_over_tg, ratio, color=colour, lw=1.0, zorder=3)
        if run == censored:  # an open end: this seed has no plateau to measure
            right.scatter([g.step_over_tg.iloc[-1]], [ratio[-1]], s=20, marker="o",
                          facecolors=S.PAGE, edgecolors=S.SLATE, linewidths=0.9, zorder=5)
    # a decade of air above the highest seed, where the second stage names what it marks
    if stage > 1:
        right.fill_between([1.0, 8.0], band[1], 90.0, color=S.BRONZE, alpha=0.10, lw=0,
                           zorder=0)
        right.text(1.12, 170.0, "sixteen of seventeen", ha="left", va="center",
                   color=S.BRONZE, style="italic", fontsize=_pt(6.8), clip_on=False)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(0.06, 8.0)
    right.set_ylim(0.10, 400.0)
    right.set_xticks([0.1, 1, 5])
    right.set_xticklabels(["0.1", "1", "5"])
    right.set_yticks([0.1, 1, 10, 100])
    right.set_yticklabels(["0.1", "1", "10", "100"])
    right.minorticks_off()
    right.set_xlabel(r"step $\div\, t_g$")
    right.set_ylabel(r"total $H_1/s$   $\div$ baseline")
    S.range_frame(right, y=(0.1, 100.0))
    S.panel_title(right, r"$S_5$   rescaled by $t_g$", pad=7)
    S.annotate(right, 1.0, 0.045, r"$t_g$", colour=S.SIENNA, ha="left", style="normal",
               size=_pt(7.0), transform=right.get_xaxis_transform())
    S.annotate(right, 0.030, band[0] * 0.70, "null", colour=S.INK, style="normal", va="top",
               size=_pt(6.8), transform=right.get_yaxis_transform())
    return fig


def _lag_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each timed condition's signed lag and signature, and the permuted-label transitions."""
    obs = "h1_max_persistence_normalised"
    d = R._load("fig-5-5-lag.csv")
    timed = d[(d.observable == obs) & d.delta.notna()]
    cond = pd.read_csv("results/processed/thesis/conditions.csv")
    keys = ["model", "operation", "modulus", "train_fraction", "weight_decay", "loss", "optimizer"]

    rows = []
    for key, g in timed.groupby(keys):
        c = cond
        for name, value in zip(keys, key, strict=True):
            c = c[c[name] == value]
        if c.empty or not np.isfinite(c[f"{obs}__med"].iloc[0]):
            continue
        model, operation, modulus, fraction, decay, loss, optimizer = key
        arm = ("" if loss == "softmax_ce" else "smax")
        arm = (arm + ("" if optimizer == "adamw" else r" $\perp$G")).strip()
        rows.append({"fields": ("MLP" if model == "mlp" else "tf", R._OPERATION[operation],
                                _setting(operation, modulus, fraction), f"{decay:g}", arm),
                     "median": float(g.delta.median()), "lo": float(g.delta.min()),
                     "hi": float(g.delta.max()), "ratio": float(c[f"{obs}__med"].iloc[0]),
                     "clears": c[f"{obs}__verdict"].iloc[0] == "above"})
    t = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
    nulls = d[(d.observable == obs) & d.label_permutation & d.t_top.notna()]
    return t, nulls


LAG_LIMIT = 1.35e5


def _lag_axis(a, *, tinted: bool) -> None:
    """Signed lag on a symmetric log axis, the lagging half tinted bronze and the leading slate."""
    a.set_xscale("symlog", linthresh=3000, linscale=0.55)
    a.set_xlim(-LAG_LIMIT, LAG_LIMIT)
    if tinted:  # an empty tinted strip reads as a box waiting to be filled
        a.axvspan(-LAG_LIMIT, 0, facecolor=S.BRONZE, alpha=0.045, lw=0, zorder=0)
        a.axvspan(0, LAG_LIMIT, facecolor=S.SLATE, alpha=0.045, lw=0, zorder=0)
        a.axvline(0, color=S.INK, lw=0.6, zorder=1)


def _lag_ticks(a) -> None:
    a.set_xlabel(r"signed lag  $\Delta = t_g - t_{\mathrm{top}}$   (steps)")
    a.set_xticks([-1e5, -1e4, 0, 1e4, 1e5])
    a.set_xticklabels(["$-10^5$", "$-10^4$", "0", "$10^4$", "$10^5$"])
    a.spines["bottom"].set_color(S.RULE)


@draws("10", "lag")
def lag(out: Path | None) -> str:
    """Topology lags where a signature exists; the large apparent leads are where none does."""
    return _stages(out, "talk-10-lag", 2, _lag)


def _lag(stage: int):
    """1 the conditions, grouped by whether they carry a signature · 2 the permuted control."""
    t, nulls = _lag_table()
    # the claim is about the two groups, so the groups are the rows' only labels; which row is
    # which condition is backup B12
    blocks = (("with a signature", t[t.clears], S.BRONZE),
              ("without one", t[~t.clears], S.INK))
    gap = 1.4

    fig = _canvas(73.0 / S.TALK_PLATE_W, plate=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[len(t) + gap, 2.2], left=0.215, right=0.965,
                          bottom=0.190, top=0.915, hspace=0.090)
    ax, axn = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _lag_axis(ax, tinted=True)
    _lag_axis(axn, tinted=stage > 1)

    label = ax.get_yaxis_transform()
    y = 0.0
    for name, block, colour in blocks:
        first = y
        for r in block.sort_values("ratio", ascending=False).itertuples():
            ax.plot([r.lo, r.hi], [y, y], color=colour, lw=S.DATA, solid_capstyle="butt",
                    zorder=3)
            ax.scatter([r.median], [y], s=16, marker="o", zorder=4, linewidths=0.8,
                       facecolors=colour if r.clears else "none", edgecolors=colour)
            y += 1
        middle = (first + y - 1) / 2
        ax.text(-0.025, middle, name, transform=label, ha="right", va="bottom", color=colour,
                family=S.SMALLCAPS, fontsize=_pt(7.4))
        ax.text(-0.025, middle, f"{len(block)} conditions", transform=label, ha="right",
                va="top", color=S.INK, alpha=0.62, fontsize=_pt(6.8))
        y += gap
    ax.axvline(-1000, color=S.BRONZE, lw=0.5, ls=(0, (1, 2)), zorder=2)
    ax.set_ylim(y - gap - 0.4, -0.8)
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis="both", length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    S.annotate(ax, 0.47, 1.012, "lags", colour=S.INK, ha="right", style="normal", size=_pt(7.6))
    S.annotate(ax, 0.53, 1.012, "leads", colour=S.SLATE, ha="left", style="normal", size=_pt(7.6))

    if stage > 1:  # a detector with no transition to find is the second half of the claim
        axn.scatter(nulls.t_top, np.zeros(len(nulls)), s=18, marker="o", facecolors="none",
                    edgecolors=S.INK, linewidths=0.8, zorder=4)
        axn.text(-0.025, 0.0, "permuted labels", transform=axn.get_yaxis_transform(),
                 ha="right", va="center", color=S.INK, family=S.SMALLCAPS, fontsize=_pt(7.4))
    axn.set_ylim(-1.1, 1.1)
    axn.set_yticks([])
    axn.spines["left"].set_visible(False)
    _lag_ticks(axn)
    return fig


@draws("10f", "lag-forest")
def lag_forest(out: Path | None) -> str:
    """Every timed condition, row by row: the backup behind the grouped timing slide."""
    return _save(_lag_forest(), "talk-10-lag-forest", out)


def _lag_forest():
    t, nulls = _lag_table()

    fig = _canvas(73.0 / S.TALK_PLATE_W, plate=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[len(t), 2.4], width_ratios=[1.0, 0.125],
                          left=0.292, right=0.972, bottom=0.190, top=0.915, hspace=0.090,
                          wspace=0.050)
    ax, axn = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[0, 1])
    _lag_axis(ax, tinted=True)
    _lag_axis(axn, tinted=True)

    for i, r in t.iterrows():
        colour = S.BRONZE if r.clears else S.INK
        ax.plot([r.lo, r.hi], [i, i], color=colour, lw=S.DATA, solid_capstyle="butt",
                zorder=3)
        ax.scatter([r["median"]], [i], s=16, marker="o", zorder=4, linewidths=0.8,
                   facecolors=colour if r.clears else "none", edgecolors=colour)
    ax.axvline(-1000, color=S.BRONZE, lw=0.5, ls=(0, (1, 2)), zorder=2)

    size = _pt(5.7)
    ax.set_yticks([])
    ax.set_ylim(-0.8, len(t) - 0.2)
    R._condition_columns(ax, list(t.fields), size=size, columns=LAG_COLUMNS,
                         headers=LAG_HEADERS)
    ax.tick_params(axis="both", length=0)
    ax.set_xticklabels([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    S.annotate(ax, 0.47, 1.012, "lags", colour=S.INK, ha="right", style="normal", size=_pt(7.6))
    S.annotate(ax, 0.53, 1.012, "leads", colour=S.SLATE, ha="left", style="normal", size=_pt(7.6))

    axn.scatter(nulls.t_top, np.zeros(len(nulls)), s=18, marker="o", facecolors="none",
                edgecolors=S.INK, linewidths=0.8, zorder=4)
    axn.set_ylim(-1.1, 1.1)
    axn.set_yticks([])
    axn.text(LAG_COLUMNS[0][0], 0.0, "permuted labels", ha="left", va="center",
             transform=axn.get_yaxis_transform(), color=S.INK, clip_on=False, fontsize=size)
    axn.spines["left"].set_visible(False)
    _lag_ticks(axn)

    axr.axvspan(*BAND, facecolor=S.RULE, alpha=0.30, lw=0, zorder=0)
    axr.barh(range(len(t)), t.ratio, height=0.46, zorder=3,
             color=[S.BRONZE if c else S.INK for c in t.clears])
    axr.set_xscale("log")
    axr.set_xlim(0.28, 12.0)
    axr.set_xticks([1, 10])
    axr.set_xticklabels(["1", "10"], fontsize=size)
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks([])
    axr.minorticks_off()
    axr.spines["left"].set_visible(False)
    axr.spines["bottom"].set_color(S.RULE)
    S.annotate(axr, 0.5, 1.012, r"$H_1^{\max}/s$", colour=S.INK, ha="center", style="normal",
               size=size)

    return fig


INTERVENTION_ARMS = (
    (("stablemax_ce", "orthograd_adamw", 1e-3), r"SMax $+\perp$G, $10^{-3}$", S.BRONZE, 1.3,
     "solid"),
    (("softmax_ce", "orthograd_adamw", 1e-3), r"$\perp$Grad, $10^{-3}$", S.SLATE, 1.3,
     (0, (4, 2))),
    (("softmax_ce", "orthograd_adamw", 1e-2), r"$\perp$Grad, $10^{-2}$", S.INK, 1.1, "solid"),
)


@draws("11", "interventions")
def interventions(out: Path | None) -> str:
    """Three arms of one MLP, none decayed, all grok; two carry a signature."""
    d = R._load("fig-5-6-interventions.csv")
    mlp = d[d.model == "mlp"]

    fig = _canvas()
    ax = fig.add_axes([0.075, 0.200, 0.335, 0.660])
    strip = fig.add_axes([0.670, 0.200, 0.200, 0.660])

    arms = ((("softmax_ce", "adamw", 1e-3), "weight decay", S.RULE, 0.9, "solid"),
            *INTERVENTION_ARMS)
    for (loss, optimizer, lr), _name, colour, weight, dash in arms:
        arm = mlp[(mlp.loss == loss) & (mlp.optimizer == optimizer) & (mlp.lr == lr)]
        for _, g in arm.groupby("run"):
            g = g.sort_values("step").dropna(subset=["test_acc"])
            ax.plot(g.step, g.test_acc, color=colour, lw=weight, ls=dash, zorder=3, alpha=0.9)
    _accuracy_axis(ax, end=float(mlp.step.max()))
    ax.set_ylabel("test accuracy")
    # the three arms are keyed at their rows on the right; the baseline is named where it rises
    ax.text(2500, 0.20, "weight decay", ha="center", va="center", color=S.INK, alpha=0.62,
            fontsize=_pt(6.4))

    # each row reads left to right: the arm's line, its name, its interval, its two numbers
    rows = [R._condition_row(block="intervention", model="mlp", loss=loss, optimizer=optimizer,
                             lr=lr) for (loss, optimizer, lr), *_ in INTERVENTION_ARMS]
    across = blended_transform_factory(fig.transFigure, strip.transData)
    S.null_band(strip, *BAND, horizontal=False)
    for i, (row, (_, name, colour, weight, dash)) in enumerate(
            zip(rows, INTERVENTION_ARMS, strict=True)):
        mark = S.BRONZE if row.verdict == "above" else S.INK
        strip.plot([row.lo, row.hi], [i, i], color=mark, lw=S.EMPHASIS, solid_capstyle="butt",
                   zorder=3)
        strip.scatter([row.med], [i], s=26, marker="o", zorder=4, linewidths=0.9,
                      facecolors=mark if row.verdict == "above" else "none", edgecolors=mark)
        strip.plot([0.452, 0.490], [i, i], transform=across, color=colour, lw=weight, ls=dash,
                   clip_on=False)
        strip.text(0.502, i, name, transform=across, ha="left", va="center", color=S.INK,
                   fontsize=_pt(7.0))
        strip.text(0.887, i, f"{row.med:.2f}", transform=across, ha="left", va="center",
                   color=mark, style="italic", fontsize=_pt(7.2))
        strip.text(0.945, i, f"{row.circularity:.2f}", transform=across, ha="left",
                   va="center", color=S.INK, fontsize=_pt(7.2))
    for x, head in ((0.887, "ratio"), (0.945, "circ.")):
        strip.text(x, -0.72, head, transform=across, ha="left", va="bottom", color=S.INK,
                   alpha=0.68, family=S.SMALLCAPS, fontsize=_pt(6.6))
    strip.text(1.0, 2.35, "null", ha="center", va="center", color=S.INK, fontsize=_pt(6.8))
    strip.set_xscale("log")
    strip.set_xlim(0.55, 24.0)
    strip.set_ylim(2.6, -0.6)
    strip.set_yticks([])
    strip.set_xticks([1, 3, 10])
    strip.set_xticklabels(["1", "3", "10"])
    strip.minorticks_off()
    strip.spines["left"].set_visible(False)
    strip.spines["bottom"].set_color(S.RULE)
    strip.set_xlabel(r"normalised $H_1^{\max}$ ratio")

    # both titles at one height, above the column heads
    for x, title in ((0.2425, "MLP"), (0.715, "the same architecture, three times")):
        fig.text(x, 0.935, title, ha="center", va="center", color=S.INK, family=S.SMALLCAPS,
                 fontsize=plt.rcParams["axes.titlesize"])
    return _save(fig, "talk-11-interventions", out)


@draws("12", "phdim")
def phdim(out: Path | None) -> str:
    """The dimension of the optimisation path says nothing about the generalisation gap."""
    d = R._load("fig-6-3-phdim.csv")
    ylim = (0.90, 1.65)
    series = [("transformer_add113_f0.3_wd0.1_dense", "reference", S.INK, "solid"),
              ("transformer_add97_permuted_dense", "permuted labels", S.BRONZE, (0, (4, 2)))]

    fig = _canvas()
    gs = fig.add_gridspec(1, 2, left=0.106, right=0.980, bottom=0.200, top=0.860, wspace=0.230)
    ax, scatter = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    spans = []
    for row, (prefix, name, colour, dash) in enumerate(series):
        sub = d[d.run.str.startswith(prefix)]
        spans.append((float(sub.step.min()), float(sub.step.max())))
        for _, g in sub.groupby("run"):
            g = g.sort_values("step")
            ax.plot(g.step, g.ph_dim, color=S.RULE, lw=0.5, zorder=2)
        median = sub.groupby("step").ph_dim.median()
        ax.plot(median.index, median.values, color=colour, lw=S.EMPHASIS, ls=dash, zorder=4)
        # the terminal value outside the frame, where no seed's curve runs under it
        ax.text(1.02, float(median.iloc[-1]), f"{median.iloc[-1]:.2f}", color=colour,
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=_pt(7.2))
        tg = sub.t_g.dropna()
        if not tg.empty:
            ax.axvline(float(tg.median()), color=S.SIENNA, lw=0.6, zorder=3)
        # the key sits low and right of t_g, under curves that have all settled above 1.1
        y = 0.135 - 0.085 * row
        ax.plot([0.470, 0.550], [y, y], transform=ax.transAxes, color=colour, lw=S.EMPHASIS,
                ls=dash, clip_on=False, zorder=6)
        ax.text(0.570, y, name, transform=ax.transAxes, color=S.INK, va="center",
                fontsize=_pt(7.0))

    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    names = {2e3: r"$2{\times}10^{3}$", 5e3: r"$5{\times}10^{3}$", 1e4: r"$10^{4}$",
             5e4: r"$5{\times}10^{4}$"}
    ticks = [t for t in names if lo <= t <= hi]
    ax.set_xscale("log")
    ax.set_xlim(lo * 0.85, hi * 1.05)
    ax.set_ylim(*ylim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([names[t] for t in ticks])
    ax.minorticks_off()
    ax.set_yticks([1.0, 1.2, 1.4, 1.6])
    ax.set_yticklabels(["1.0", "1.2", "1.4", "1.6"])
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\dim_{\mathrm{PH}}$")
    S.range_frame(ax, x=(lo, hi), y=ylim)  # the axis at the frame's foot, under the key
    S.annotate(ax, float(d[d.run.str.startswith(series[0][0])].t_g.dropna().median()) * 1.15,
               0.96, r"$t_g$", colour=S.SIENNA, ha="left", va="top", style="normal",
               size=_pt(7.0), transform=ax.get_xaxis_transform())

    terminal = (d.sort_values("step").groupby("run")
                .agg(dim=("ph_dim", lambda s: float(s.dropna().tail(5).median())),
                     gap=("generalisation_gap", "first"),
                     fits=("fits_train_set", "first"))
                .dropna(subset=["dim", "gap"]))
    fitting = terminal[terminal.fits.astype(bool)]
    inside, above = fitting[fitting.dim <= ylim[1]], fitting[fitting.dim > ylim[1]]
    scatter.scatter(inside.gap, inside.dim, s=17, marker="o", facecolors="none",
                    edgecolors=S.INK, linewidths=0.6, zorder=3)
    if len(above):
        scatter.scatter(above.gap, np.full(len(above), ylim[1] - 0.014), s=17, marker="^",
                        facecolors="none", edgecolors=S.INK, linewidths=0.6, zorder=4,
                        clip_on=False)
        S.annotate(scatter, 0.975, 0.925,
                   "off scale:  " + ", ".join(f"{v:.1f}" for v in sorted(above.dim)),
                   colour=S.INK, ha="right", va="top", style="normal", size=_pt(6.6))
    by_condition = fitting.assign(cond=fitting.index.str.replace(r"_s\d+$", "", regex=True))
    for _, g in by_condition.groupby("cond"):
        if g.dim.median() <= ylim[1]:
            scatter.scatter([g.gap.median()], [g.dim.median()], s=30, marker="o",
                            color=S.BRONZE, zorder=5, linewidths=0)
    rho, _ = R._spearman(fitting.gap, fitting.dim)
    S.value(scatter, 0.972, 0.740, rf"$\rho = {rho:+.3f}$", f"n = {len(fitting)}", ha="right")

    scatter.set_xlim(-0.12, 1.12)
    scatter.set_ylim(*ylim)
    scatter.set_xticks([0.0, 0.5, 1.0])
    scatter.set_xticklabels(["0", "0.5", "1"])
    scatter.set_yticks([1.0, 1.2, 1.4, 1.6])
    scatter.set_yticklabels([])  # the left panel's scale, shared
    scatter.set_xlabel("generalisation gap")
    S.range_frame(scatter, x=(0, 1), y=ylim)
    S.panel_title(scatter, "terminal dimension against the gap", pad=7)

    return _save(fig, "talk-12-phdim", out)


@draws("13", "crocker")
def crocker(out: Path | None) -> str:
    """The run as one surface, on the axis the contraction has been divided out of."""
    d = R._load("fig-6-1-crocker.csv")
    d = d[d.axis == "normalised"]
    cmap = S.sequential(floor=0.32)
    vmax = float(np.percentile(d.betti1[d.betti1 > 0], 90))
    norm = PowerNorm(gamma=0.48, vmin=0.5, vmax=vmax)

    fig = _canvas()
    gs = fig.add_gridspec(1, 2, left=0.112, right=0.980, bottom=0.395, top=0.905, wspace=0.085)
    mesh = None
    for ci, (regime, subtitle) in enumerate((("reference", r"$p=113$, wd $0.1$"),
                                             ("canonical", r"$p=97$, wd $1.0$"))):
        ax = fig.add_subplot(gs[0, ci])
        cell = d[d.regime == regime]
        grid = cell.pivot(index="scale", columns="step", values="betti1")
        mesh = ax.pcolormesh(grid.columns.values, grid.index.values, grid.values, cmap=cmap,
                             norm=norm, shading="nearest", rasterized=True)
        tg = cell.t_g.dropna()
        if not tg.empty:
            ax.axvline(float(tg.iloc[0]), color=S.SIENNA, lw=S.HAIRLINE, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(max(grid.columns.min(), 1), grid.columns.max())
        # ticks chosen from inside the view, and the view restored: set_yticks autoscales out
        # to include a tick that lies beyond the data, which flattens the surface
        lo, hi = ax.get_ylim()
        ticks = [t for t in (0.2, 0.5, 1.0, 2.0, 5.0) if lo < t < hi]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_ylim(lo, hi)
        ax.minorticks_off()
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=2.0, width=S.HAIRLINE)
        ax.set_xlabel("training step")
        S.panel_title(ax, f"{regime}   {subtitle}", pad=7)
        if ci == 0:
            ax.set_ylabel(r"scale $\div\, s$")
        else:
            ax.set_yticks([])

    bar = fig.add_axes([0.390, 0.145, 0.250, 0.028])
    colours = fig.colorbar(mesh, cax=bar, orientation="horizontal", extend="max",
                           ticks=[1, 5, 10, 20, 30, 40])
    colours.outline.set_visible(False)
    bar.set_xlabel(r"live $H_1$ features", labelpad=2)
    bar.tick_params(length=2.0, width=S.HAIRLINE, pad=1.5)
    bar.add_patch(plt.Rectangle((-0.075, 0.0), 0.045, 1.0, transform=bar.transAxes,
                                facecolor=S.PAGE, edgecolor=S.RULE, lw=S.HAIRLINE,
                                clip_on=False, zorder=4))
    bar.text(-0.0525, -0.65, "0", transform=bar.transAxes, ha="center", va="top",
             color=S.INK, fontsize=_pt(6.6))

    return _save(fig, "talk-13-crocker", out)


# One task, addition mod 97, and test accuracy 1.0 in all three; seed 0 of each condition
RING_RUNS = (
    ("mlp_add97_f0.3_wd1.0_softmax_ce_s0", {"model": "mlp"}, r"MLP,  wd $1.0$"),
    ("transformer_add97_f0.3_wd1.0_softmax_ce_s0",
     {"model": "transformer", "operation": "add", "modulus": 97, "train_fraction": 0.3,
      "weight_decay": 1.0}, r"transformer,  wd $1.0$"),
    ("transformer_add97_f0.3_wd3.0_softmax_ce_s0", {"weight_decay": 3.0},
     r"transformer,  wd $3.0$"),
)


@draws("14", "rings")
def rings(out: Path | None) -> str:
    """Three solutions to one task at one accuracy, and the ratio that tells them apart."""
    from analysis.figures.dial import ring, terminal_embedding

    S.use(S.TALK)
    width = 0.8 * S.TALK_W  # the slide's inner measure, which the claim and the rules share
    fig = S.figure(width, 36.0 / width, S.TALK)
    for i, (run, where, name) in enumerate(RING_RUNS):
        row = R._condition_row(block="condition", **where)
        xy = ring(terminal_embedding(run))["xy"]
        centre = (2 * i + 1) / 6
        ax = fig.add_axes([centre - 0.105, 0.300, 0.210, 0.660])
        # threaded in residue order, 0 -> 1 -> ... -> 0: a clean circle traces its rim
        loop = np.vstack([xy, xy[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color=S.INK, lw=0.35, alpha=0.55, zorder=2)
        ax.scatter(xy[:, 0], xy[:, 1], s=3.0, color=S.INK, linewidths=0, zorder=3)
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_aspect("equal")
        ax.axis("off")
        colour = {"above": S.BRONZE, "below": S.SLATE}.get(row.verdict, S.INK)
        fig.text(centre, 0.180, f"{row.med:.2f}", ha="center", va="center", color=colour,
                 style="italic", fontsize=_pt(12.0))
        fig.text(centre, 0.050, name, ha="center", va="center", color=S.INK, alpha=0.62,
                 fontsize=_pt(6.6))
    return _save(fig, "talk-14-rings", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", type=Path, help=f"output root (default: {OUTPUT_ROOT})")
    args = parser.parse_args()

    for number in args.only or sorted(RENDERERS):
        fn = RENDERERS.get(number)
        if fn is None:
            print(f"  {number:4s} no renderer")
            continue
        try:
            path = fn(args.out)
        except FileNotFoundError as exc:
            print(f"  {number:4s} {fn.figure_name:14s} skipped — missing {exc}")
            continue
        print(f"  {number:4s} {fn.figure_name:14s} -> {path}")


if __name__ == "__main__":
    main()
