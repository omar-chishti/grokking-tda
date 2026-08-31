"""Render the presentation figures from the same tidy CSVs the thesis figures are drawn from.

A thesis plate is 158 mm wide at 8 pt; placed on a 160 mm slide so its labels read at body size it
would have to be 217 mm across. So the slide figures are separate compositions -- one claim, fewer
panels, larger type -- drawn at exactly the size the frame gives them, so nothing is ever scaled.
Everything below the composition (data, style, devices) comes from ``render`` and ``style``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm

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


def _pt(factor: float) -> float:
    return factor * S.TALK.scale


def _save(fig, name: str, out: Path | None) -> str:
    return S.save(fig, name, S.TALK, out_dir=out or OUTPUT_ROOT, svg=False, suffix="")


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
    d = R._load("fig-1-1-hero.csv")
    t_c, t_g = float(d.t_c.dropna().iloc[0]), float(d.t_g.dropna().iloc[0])
    acc = d.dropna(subset=["test_acc"]).sort_values("step")

    fig = _canvas()
    ax = fig.add_axes([0.098, 0.155, 0.892, 0.815])

    ax.axvspan(t_c, t_g, facecolor=S.INK, alpha=0.04, lw=0, zorder=0)
    ax.axvline(t_c, color=S.RULE, lw=S.HAIRLINE, zorder=1)
    ax.axvline(t_g, color=S.SIENNA, lw=S.HAIRLINE, zorder=1)
    ax.plot(acc.step, acc.train_acc, color=S.RULE, lw=1.0, zorder=2)
    ax.plot(acc.step, acc.test_acc, color=S.INK, lw=1.4, zorder=4)

    _accuracy_axis(ax, end=float(acc.step.max()))
    ax.set_ylabel("accuracy")
    S.direct_label(ax, 900, 1.00, "train", S.BRONZE, dy=-4, va="top", size=_pt(7.6))
    S.direct_label(ax, 900, 0.305, "test", S.INK, dy=4, va="bottom", size=_pt(7.6))

    trans = ax.get_xaxis_transform()
    ax.text(t_c * 1.35, 0.055, r"$t_c$ = 200", transform=trans, ha="left", va="bottom",
            color=S.INK, fontsize=_pt(7.2))
    ax.text(t_g * 0.62, 0.055, r"$t_g$ = 28,600", transform=trans, ha="right", va="bottom",
            color=S.SIENNA, fontsize=_pt(7.2), zorder=6)

    y = 0.660
    ax.annotate("", xy=(t_c, y), xytext=(t_g, y), xycoords=trans, textcoords=trans,
                arrowprops=dict(arrowstyle="<->", color=S.BRONZE, lw=S.HAIRLINE,
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.text(t_c * 3.4, y + 0.030, r"$143\times$", transform=trans, ha="center", va="bottom",
            color=S.BRONZE, style="italic", fontsize=_pt(8.6))

    return _save(fig, "talk-01-phenomenon", out)


@draws("02", "leak")
def leak(out: Path | None) -> str:
    """The pre-grokking plateau is commutativity, and it is worth exactly the train fraction."""
    d = R._load("fig-1-1-hero.csv")
    t_g = float(d.t_g.dropna().iloc[0])
    acc = d.dropna(subset=["test_acc"]).sort_values("step")
    novel = d.dropna(subset=["test_acc_novel"]).sort_values("step")
    fraction = 0.30
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
    S.value(ax, 0.230, 0.905, f"{fraction:.3f}", "train fraction")

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


@draws("05", "signature")
def signature(out: Path | None) -> str:
    """Divided by a scale intrinsic to the same cloud, one regime rises and one does not."""
    fig = _canvas()
    gs = fig.add_gridspec(1, 2, left=0.112, right=0.986, bottom=0.200, top=0.860, wspace=0.075)
    _signature_row(fig, gs, "h1_max_persistence_normalised",
                   ylabel=r"normalised   $H_1^{\max}/s$", scale_trace=False)
    return _save(fig, "talk-05-signature", out)


# III. what we found

TALK_COLUMNS = ((-0.355, "left"), (-0.240, "left"), (-0.060, "right"))
TALK_HEADERS = ("arch", "op", "wd")
# the lag plate carries two MLP intervention arms that the first three fields cannot tell apart
LAG_COLUMNS = ((-0.430, "left"), (-0.318, "left"), (-0.180, "right"), (-0.164, "left"))
LAG_HEADERS = (*TALK_HEADERS, "")


def _short_fields(row) -> tuple[str, ...]:
    """Architecture, operation and dose. The modulus and fraction are detail a room cannot read."""
    if row.block == "null model":
        return ("", R._OPERATION[row.operation], "")
    censored = "" if (row.n_runs == 5 and row.n_grokked == 5) else \
        f"$^{{{int(row.n_grokked)}/{int(row.n_runs)}}}$"
    return ("MLP" if row.model == "mlp" else "tf", R._OPERATION[row.operation],
            f"{row.weight_decay:g}{censored}")


@draws("06", "robustness")
def robustness(out: Path | None) -> str:
    """Five conditions clear the null band, ten sit inside it and one runs backwards."""
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

    return _save(fig, "talk-06-robustness", out)


@draws("07", "circularity")
def circularity(out: Path | None) -> str:
    """What separates the regimes is how circular the solution is, not whether it generalises."""
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
        target.set_ylim(0.14, 14.0)
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

    S.key(fig, (0.585, 0.318, 0.240, 0.200),
          [("transformer", glyph("o")), ("MLP", glyph("s")), ("weight decay", ramp)],
          heading="run")

    return _save(fig, "talk-07-circularity", out)


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


@draws("09", "noncyclic")
def noncyclic(out: Path | None) -> str:
    """Composition in S_5 groks, and the signature is present where no circle can be."""
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
    S.annotate(left, 0.050 + 0.032 * len(runs), 0.905,
               f"over {steps[0] / 1000:.1f}–{steps[-1] / 1000:.1f}k steps", colour=S.SIENNA,
               ha="left", va="center", style="normal", size=_pt(7.0))

    S.null_band(right, *band)
    right.axvline(1.0, color=S.SIENNA, lw=S.HAIRLINE, zorder=1)
    ends = []
    for run in runs:
        g = d[d.run == run].sort_values("step").dropna(subset=["step_over_tg"])
        colour = S.SLATE if run == censored else S.INK
        ratio = R._baseline_ratio(g, "h1_total_persistence_normalised")
        right.plot(g.step_over_tg, ratio.values, color=colour, lw=1.0, zorder=3)
        ends.append([float(g.step_over_tg.iloc[-1]), float(ratio.iloc[-1]), colour])
        if run == censored:
            right.scatter([g.step_over_tg.iloc[-1]], [ratio.iloc[-1]], s=20, marker="o",
                          facecolors="none", edgecolors=S.SLATE, linewidths=0.9, zorder=5)
    # two seeds finish within a few per cent of each other, so the labels are pushed apart in
    # log space before they are drawn: a pair that overprints reports one number, not two
    ends.sort(key=lambda e: e[1])
    drawn = 0.0
    for x, value, colour in ends:
        y = max(value, drawn * 1.34)
        drawn = y
        S.direct_label(right, x, y, f"{value:.1f}", colour, dx=4, size=_pt(7.0))
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
               size=_pt(7.0), transform=right.get_xaxis_transform())
    S.annotate(right, 0.030, band[1] * 1.12, "null", colour=S.INK, style="normal",
               size=_pt(6.8), transform=right.get_yaxis_transform())

    return _save(fig, "talk-09-noncyclic", out)


@draws("10", "lag")
def lag(out: Path | None) -> str:
    """Topology lags where a signature exists; the large apparent leads are where none does."""
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
        model, operation, *_rest, decay, loss, optimizer = key
        arm = ("" if loss == "softmax_ce" else "smax")
        arm = (arm + ("" if optimizer == "adamw" else r" $\perp$G")).strip()
        rows.append({"fields": ("MLP" if model == "mlp" else "tf",
                                R._OPERATION[operation], f"{decay:g}", arm),
                     "median": float(g.delta.median()), "lo": float(g.delta.min()),
                     "hi": float(g.delta.max()), "ratio": float(c[f"{obs}__med"].iloc[0]),
                     "clears": c[f"{obs}__verdict"].iloc[0] == "above"})
    t = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
    nulls = d[(d.observable == obs) & d.label_permutation & d.t_top.notna()]

    fig = _canvas(73.0 / S.TALK_PLATE_W, plate=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[len(t), 2.4], width_ratios=[1.0, 0.125],
                          left=0.292, right=0.972, bottom=0.190, top=0.915, hspace=0.090,
                          wspace=0.050)
    ax, axn = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[0, 1])

    limit = 1.35e5
    for a in (ax, axn):
        a.set_xscale("symlog", linthresh=3000, linscale=0.55)
        a.set_xlim(-limit, limit)
        a.axvspan(-limit, 0, facecolor=S.BRONZE, alpha=0.045, lw=0, zorder=0)
        a.axvspan(0, limit, facecolor=S.SLATE, alpha=0.045, lw=0, zorder=0)
        a.axvline(0, color=S.INK, lw=0.6, zorder=1)

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
    axn.set_xlabel(r"signed lag  $\Delta = t_g - t_{\mathrm{top}}$   (steps)")
    axn.set_xticks([-1e5, -1e4, 0, 1e4, 1e5])
    axn.set_xticklabels(["$-10^5$", "$-10^4$", "0", "$10^4$", "$10^5$"])
    axn.spines["bottom"].set_color(S.RULE)
    S.annotate(axn, 0.02, 0.78, "$t_{top}$ only", colour=S.INK, ha="left", style="normal",
               size=size)

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

    return _save(fig, "talk-10-lag", out)


@draws("11", "interventions")
def interventions(out: Path | None) -> str:
    """Two interventions, one architecture, both grok; only one produces a signature."""
    d = R._load("fig-5-6-interventions.csv")
    arms = {
        ("softmax_ce", "adamw", 0.001): ("weight decay", S.RULE, 0.9, "solid"),
        ("softmax_ce", "orthograd_adamw", 0.01): (r"$\perp$Grad", S.INK, 1.1, "solid"),
        ("stablemax_ce", "orthograd_adamw", 0.001):
            (r"StableMax $+\perp$Grad", S.BRONZE, 1.3, "solid"),
    }

    fig = _canvas()
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.90], left=0.104, right=0.982,
                          bottom=0.200, top=0.860, wspace=0.180)
    ax = fig.add_subplot(gs[0, 0])
    strip = fig.add_subplot(gs[0, 1])

    mlp = d[d.model == "mlp"]
    for (loss, optimizer, lr), (_name, colour, weight, dash) in arms.items():
        arm = mlp[(mlp.loss == loss) & (mlp.optimizer == optimizer) & (mlp.lr == lr)]
        for _, g in arm.groupby("run"):
            g = g.sort_values("step").dropna(subset=["test_acc"])
            ax.plot(g.step, g.test_acc, color=colour, lw=weight, ls=dash, zorder=3, alpha=0.9)
    _accuracy_axis(ax, end=float(mlp.step.max()))
    ax.set_ylabel("test accuracy")
    S.panel_title(ax, "MLP", pad=7)
    for row, (name, colour, weight, dash) in enumerate(arms.values()):
        y = 0.640 - 0.105 * row
        ax.plot([0.500, 0.600], [y, y], transform=ax.transAxes, color=colour, lw=weight,
                ls=dash, clip_on=False, zorder=6)
        ax.text(0.625, y, name, transform=ax.transAxes, color=S.INK, va="center",
                fontsize=_pt(7.0))

    pairs = [R._condition_row(block="intervention", model="mlp", loss="stablemax_ce",
                              optimizer="orthograd_adamw"),
             R._condition_row(block="intervention", model="mlp", loss="softmax_ce",
                              optimizer="orthograd_adamw")]
    labels = [r"StableMax $+\perp$Grad", r"$\perp$Grad"]
    S.null_band(strip, *BAND, horizontal=False)
    for i, row in enumerate(pairs):
        colour = S.BRONZE if row.verdict == "above" else S.INK
        strip.plot([row.lo, row.hi], [i, i], color=colour, lw=S.EMPHASIS,
                   solid_capstyle="butt", zorder=3)
        strip.scatter([row.med], [i], s=26, marker="o", zorder=4, linewidths=0.9,
                      facecolors=colour if row.verdict == "above" else "none", edgecolors=colour)
        # each arm is named at its own marker: two rows whose names sit at the far left and
        # whose markers sit an order of magnitude apart read as unrelated
        # each arm is named at its own marker; the alignment follows where in the frame the
        # marker sits, or a caption centred near an edge runs off it
        at = (np.log(row.med) - np.log(0.55)) / (np.log(24.0) - np.log(0.55))
        ha = "left" if at < 0.32 else ("right" if at > 0.68 else "center")
        strip.text(row.med, i - 0.30, labels[i], ha=ha, va="bottom", color=S.INK,
                   fontsize=_pt(7.2))
        # the grokking steps are the left panel's job; this one carries the geometry
        S.value(strip, row.med, i + 0.32, f"{row.med:.2f}",
                f"circularity {row.circularity:.2f}", colour=colour, ha=ha, va="top",
                transform=strip.transData)
    strip.set_xscale("log")
    strip.set_xlim(0.55, 24.0)
    strip.set_ylim(2.05, -0.95)
    strip.set_yticks([])
    strip.set_xticks([1, 3, 10])
    strip.set_xticklabels(["1", "3", "10"])
    strip.minorticks_off()
    strip.spines["left"].set_visible(False)
    strip.spines["bottom"].set_color(S.RULE)
    strip.set_xlabel(r"plateau $\div$ baseline,   normalised $H_1^{\max}$")
    S.annotate(strip, 1.0, 0.975, "null", colour=S.INK, ha="center", va="top", style="normal",
               size=_pt(6.8), transform=strip.get_xaxis_transform())
    S.panel_title(strip, "the same architecture, twice", pad=7)

    return _save(fig, "talk-11-interventions", out)


@draws("12", "phdim")
def phdim(out: Path | None) -> str:
    """The dimension of the optimisation path says nothing about the generalisation gap."""
    d = R._load("fig-6-3-phdim.csv")
    ylim = (0.95, 1.65)
    series = [("transformer_add113_f0.3_wd0.1_dense", "reference", S.INK, "solid"),
              ("transformer_add97_permuted_dense", "permuted labels", S.BRONZE, (0, (4, 2)))]

    fig = _canvas()
    gs = fig.add_gridspec(1, 2, left=0.106, right=0.980, bottom=0.200, top=0.860, wspace=0.210)
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
        S.direct_label(ax, median.index[-1], float(median.iloc[-1]), f"  {median.iloc[-1]:.2f}",
                       colour, dx=2, size=_pt(7.2))
        tg = sub.t_g.dropna()
        if not tg.empty:
            ax.axvline(float(tg.median()), color=S.SIENNA, lw=0.6, zorder=3)
        y = 0.945 - 0.095 * row
        ax.plot([0.330, 0.430], [y, y], transform=ax.transAxes, color=colour, lw=S.EMPHASIS,
                ls=dash, clip_on=False, zorder=6)
        ax.text(0.455, y, name, transform=ax.transAxes, color=S.INK, va="center",
                fontsize=_pt(7.0))

    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    ax.set_xscale("log")
    ax.set_xlim(lo * 0.85, hi * 1.30)
    ax.set_ylim(*ylim)
    ax.set_xticks([t for t in (2e3, 5e3, 1e4, 5e4) if lo <= t <= hi])
    ax.set_xticklabels([r"$2{\times}10^{3}$", r"$5{\times}10^{3}$", r"$10^{4}$",
                        r"$5{\times}10^{4}$"][-len([t for t in (2e3, 5e3, 1e4, 5e4)
                                                     if lo <= t <= hi]):])
    ax.minorticks_off()
    ax.set_yticks([1.0, 1.2, 1.4, 1.6])
    ax.set_yticklabels(["1.0", "1.2", "1.4", "1.6"])
    ax.minorticks_off()
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\dim_{\mathrm{PH}}$")
    S.range_frame(ax, x=(lo, hi), y=ylim)
    S.annotate(ax, float(d[d.run.str.startswith(series[0][0])].t_g.dropna().median()) * 1.15,
               0.045, r"$t_g$", colour=S.SIENNA, ha="left", style="normal", size=_pt(7.0),
               transform=ax.get_xaxis_transform())

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
        S.direct_label(scatter, float(above.gap.min()), ylim[1] - 0.014,
                       "off scale " + ", ".join(f"{v:.1f}" for v in sorted(above.dim)), S.INK,
                       dx=0, dy=-9, ha="center", va="top", size=_pt(6.8))
    by_condition = fitting.assign(cond=fitting.index.str.replace(r"_s\d+$", "", regex=True))
    for _, g in by_condition.groupby("cond"):
        if g.dim.median() <= ylim[1]:
            scatter.scatter([g.gap.median()], [g.dim.median()], s=30, marker="o",
                            color=S.BRONZE, zorder=5, linewidths=0)
    rho, _ = R._spearman(fitting.gap, fitting.dim)
    S.value(scatter, 0.972, 0.760, rf"$\rho = {rho:+.3f}$", f"n = {len(fitting)}", ha="right")

    scatter.set_xlim(-0.12, 1.12)
    scatter.set_ylim(*ylim)
    scatter.set_xticks([0.0, 0.5, 1.0])
    scatter.set_xticklabels(["0", "0.5", "1"])
    scatter.set_yticks([1.0, 1.2, 1.4, 1.6])
    scatter.set_yticklabels(["1.0", "1.2", "1.4", "1.6"])
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
