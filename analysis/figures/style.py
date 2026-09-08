"""The house drawing style for generated thesis figures (master §4)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from grokking_tda.plotting.style import BRONZE, INK, PAGE, RULE, SAGE, SIENNA, SLATE

from . import fonts

SMALLCAPS = fonts.install()

# Beside the manuscript when it is there, into the repository otherwise: an unconditional
# default once *created* ``../LaTeX/Final Thesis/`` in a clone with no thesis beside it
_THESIS = Path("../LaTeX/Final Thesis/figures/generated")
OUTPUT_ROOT = Path(
    os.environ.get("GTDA_FIGURE_DIR")
    or (_THESIS if _THESIS.parent.parent.is_dir() else Path("results/figures/generated"))
)

# the palette, series encodings and geometry every figure in the set is drawn to

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "thesis-seq", ["#F5EFE6", "#E7D9C3", "#D3BC9B", "#B99A6E", "#9A7748", "#6E5330"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "thesis-div",
    ["#2E4B54", "#4E7480", "#8FAAB0", "#E9E4DC", "#D3BC9B", "#9A7748", "#6E5330"],
)

# colour + dash + glyph, always all three (master §4.1)
SERIES = {
    "topology": (INK, (None, None), "o"),
    "fourier": (BRONZE, (4, 2), "s"),
    "weight": (SLATE, (5, 2, 1, 2), "^"),
    "lid": (SAGE, (1, 2), "D"),
    "combined": (RULE, (None, None), None),
}

MM = 1 / 25.4
FULL, TWO_THIRDS, HALF = 158.0, 105.0, 76.0  # column widths, in millimetres
RATIOS = {"wide": 0.5, "standard": 2 / 3, "square": 1.0, "portrait": 5 / 4}

HAIRLINE, SECONDARY, DATA, EMPHASIS = 0.4, 0.7, 1.0, 1.2


@dataclass(frozen=True)
class Variant:
    name: str
    scale: float
    ground: str | None


THESIS = Variant("thesis", 1.0, None)
SLIDE = Variant("slide", 1.65, PAGE)
# The talk canvas is the slide's figure box itself, so a rendered figure is placed at 1:1 and
# its type lands at the size it was drawn. Widths in millimetres; see analysis/figures/talk.py.
TALK = Variant("talk", 1.25, PAGE)
TALK_W, TALK_PLATE_W = 148.0, 152.0


def use(variant: Variant = THESIS) -> Variant:
    s = variant.scale
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.transparent": variant.ground is None,
            "figure.facecolor": variant.ground or "none",
            "axes.facecolor": "none",
            "font.family": "serif",
            "font.serif": ["ETbb", "Palatino", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0 * s,
            "axes.labelsize": 8.0 * s,
            "axes.titlesize": 8.5 * s,
            "xtick.labelsize": 7.5 * s,
            "ytick.labelsize": 7.5 * s,
            "legend.fontsize": 7.5 * s,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": HAIRLINE,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.4 * s,
            "ytick.major.size": 2.4 * s,
            "xtick.major.width": HAIRLINE,
            "ytick.major.width": HAIRLINE,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "legend.frameon": False,
            "figure.autolayout": False,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
        }
    )
    return variant


def figure(width_mm: float, ratio: float, variant: Variant = THESIS):
    return plt.figure(
        figsize=(width_mm * MM, width_mm * ratio * MM),
        facecolor=variant.ground or "none",
    )


def range_frame(ax, x=None, y=None, *, join: bool = True) -> None:
    if x is None:
        x = ax.get_xlim()
    if y is None:
        y = ax.get_ylim()
    x, y = sorted(x), sorted(y)
    ax.spines["bottom"].set_bounds(*x)
    ax.spines["left"].set_bounds(*y)
    if join:
        # seat each spine at whichever end of the other's range lies nearer that axis's
        # origin, so an inverted axis puts its rule at the foot of the panel
        x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
        ax.spines["bottom"].set_position(("data", min(y, key=lambda v: abs(v - y0))))
        ax.spines["left"].set_position(("data", min(x, key=lambda v: abs(v - x0))))
    ax.spines["bottom"].set_color(RULE)
    ax.spines["left"].set_color(RULE)


def direct_label(ax, x, y, text, colour=INK, *, dx=2.0, dy=0.0, size=None, style="normal",
                 va="center", ha="left"):
    ax.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=(dx, dy),
        textcoords="offset points",
        color=colour,
        va=va,
        ha=ha,
        fontsize=size or mpl.rcParams["font.size"] * 0.94,
        style=style,
        annotation_clip=False,
    )


def annotate(ax, x, y, text, *, colour=INK, size=None, ha="left", va="bottom",
             style="italic", transform=None, rotation=0):
    ax.text(
        x, y, text, transform=transform if transform is not None else ax.transAxes,
        color=colour, ha=ha, va=va, style=style, rotation=rotation, clip_on=False,
        fontsize=size or mpl.rcParams["font.size"] * 0.94,
    )


def panel_letter(ax, letter: str, *, dx_mm: float = 7.5, dy_mm: float = 2.0) -> None:
    """A parenthesised small capital outside the panel. ``dx_mm`` belongs to the column."""
    fig = ax.get_figure()
    box = ax.get_position()
    w, h = fig.get_size_inches()
    fig.text(
        box.x0 - (dx_mm * MM) / w,
        box.y1 + (dy_mm * MM) / h,
        f"({letter})",
        color=INK,
        ha="left",
        va="bottom",
        family=SMALLCAPS,
        fontsize=mpl.rcParams["font.size"] * 1.14,
    )


def panel_title(ax, text: str, *, pad: float = 8.0, colour: str = INK) -> None:
    ax.set_title(text, color=colour, pad=pad, loc="center", family=SMALLCAPS)


def value(ax, x, y, number: str, name: str = "", *, colour=INK, ha="left", va="bottom",
          transform=None, gap: float = 1.35) -> None:
    axes = transform if transform is not None else ax.transAxes
    size = mpl.rcParams["font.size"]
    ax.text(x, y, number, transform=axes, color=colour, ha=ha, va=va, style="italic",
            clip_on=False, fontsize=size * 0.94)
    if name:
        ax.annotate(name, xy=(x, y), xycoords=axes, xytext=(0, -gap * size), va="top",
                    textcoords="offset points", color=INK, alpha=0.62, ha=ha,
                    clip_on=False, annotation_clip=False, fontsize=size * 0.84)


def key(fig, rect, entries, *, heading: str = "") -> None:
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(RULE)
        side.set_linewidth(HAIRLINE)
    size = mpl.rcParams["font.size"]
    top = 1.0
    if heading:
        ax.text(0.055, 0.955, heading, va="top", ha="left", family=SMALLCAPS, color=INK,
                fontsize=size * 0.88)
        top = 0.80
    rows = len(entries)
    step = (top - 0.06) / max(rows, 1)
    for i, (label, draw) in enumerate(entries):
        y = top - step * (i + 0.5)
        draw(ax, y)
        ax.text(0.30, y, label, va="center", ha="left", color=INK, fontsize=size * 0.88)


def sequential(zero_as_page: bool = True, floor: float = 0.0):
    """The house ramp, absence as paper: pair with ``vmin`` above zero so ``set_under`` fires."""
    cmap = SEQUENTIAL
    if floor > 0.0:
        cmap = LinearSegmentedColormap.from_list(
            "thesis-seq-floored", SEQUENTIAL(np.linspace(floor, 1.0, 256))
        )
    cmap = cmap.copy()
    if zero_as_page:
        cmap.set_under(PAGE)
    return cmap


def null_band(ax, lo: float, hi: float, *, horizontal: bool = True) -> None:
    span = ax.axhspan if horizontal else ax.axvspan
    span(lo, hi, facecolor=RULE, alpha=0.18, edgecolor="none", zorder=0)
    line = ax.axhline if horizontal else ax.axvline
    line(1.0, color=RULE, lw=HAIRLINE, zorder=0.5)


def seed_comb(ax, steps, *, height=0.045, colour=SIENNA) -> None:
    for s in steps:
        ax.axvline(s, ymin=0.0, ymax=height, color=colour, lw=0.6, zorder=3, clip_on=False)


def sparkline(ax, values, *, colours=None, band=None, log: bool = False, kind: str = "bar",
              ticks=(0.0, 1.0), label: str = "", height: float = 0.42, offset: float = 0.0,
              size: float = 11.0) -> None:
    """A narrow column beside a forest (device 5): bars where zero means something, else dots."""
    n = len(values)
    rows = np.arange(n) + offset
    if band is not None:
        ax.axvspan(*band, facecolor=RULE, alpha=0.30, lw=0, zorder=0)
    if kind == "bar":
        ax.barh(rows, values, height=height, zorder=3,
                color=colours if colours is not None else INK)
    else:
        ax.scatter(values, rows, s=size, marker="o", zorder=3, linewidths=0.0,
                   color=colours if colours is not None else INK)
    if log:
        ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=mpl.rcParams["font.size"] * 0.8)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    if label:
        ax.text(0.5, 1.006, label, transform=ax.transAxes, ha="center", va="bottom",
                color=INK, family=SMALLCAPS, fontsize=mpl.rcParams["font.size"] * 0.86)


def save(fig, name: str, variant: Variant = THESIS, out_dir=None, svg: bool = True,
         suffix: str | None = None) -> str:
    """``suffix`` overrides the variant tag, for a set whose names already carry the variant."""
    root = Path(out_dir) if out_dir else OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if suffix is None:
        suffix = "" if variant.name == "thesis" else f"-{variant.name}"
    path = root / f"{name}{suffix}.pdf"
    fig.savefig(path, format="pdf", transparent=variant.ground is None)
    if svg:
        (root / "svg").mkdir(exist_ok=True)
        fig.savefig(root / "svg" / f"{name}{suffix}.svg", format="svg",
                    transparent=variant.ground is None)
    plt.close(fig)
    return str(path)
