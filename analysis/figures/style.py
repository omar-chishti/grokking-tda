"""The house drawing style for generated thesis figures.

The palette, the type scale, the millimetre geometry, the line weights, and the helpers
that make range frames and direct labelling cheap enough that nobody reaches for a legend.

The body serif is ETbb and matplotlib can see it, so figure type matches the manuscript
exactly. That single fact is most of the difference between a designed figure and a
screenshot of a plotting library. ``fonts`` adds the one thing matplotlib cannot ask an
OpenType face for: real small capitals, for panel letters and panel titles. Figures inside
those labels come out old-style to sit with the small caps, as they do on the author's
plates; everywhere a number is a *measurement* --- ticks, values, annotations --- the type
stays lining, because a column of tick labels has to align and a reader has to compare
digits.

Two variants share one code path: ``thesis`` at 158/105/76 mm column widths, and ``slide``
at the same geometry with the type scaled and the ground made explicit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import fonts

SMALLCAPS = fonts.install()

# Write beside the manuscript when the manuscript is there, and into the repository
# otherwise. The rule matters because the previous unconditional default *created*
# ``../LaTeX/Final Thesis/`` — a clone with no thesis beside it wrote into its own
# parent directory. Overridable with ``--out`` or GTDA_FIGURE_DIR.
_THESIS = Path("../LaTeX/Final Thesis/figures/generated")
OUTPUT_ROOT = Path(
    os.environ.get("GTDA_FIGURE_DIR")
    or (_THESIS if _THESIS.parent.parent.is_dir() else Path("results/figures/generated"))
)

# --- palette (master §4.1) --------------------------------------------------------------
INK = "#1F1B16"
BRONZE = "#8C6A43"
RULE = "#C8B69B"
PAGE = "#FBFAF7"
SIENNA = "#96402A"
SLATE = "#3A5C66"
SAGE = "#8B9179"

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

# --- geometry (master §4.3) -------------------------------------------------------------
MM = 1 / 25.4
FULL, TWO_THIRDS, HALF = 158.0, 105.0, 76.0
RATIOS = {"wide": 0.5, "standard": 2 / 3, "square": 1.0, "portrait": 5 / 4}

# --- line weights -----------------------------------------------------------------------
HAIRLINE, SECONDARY, DATA, EMPHASIS = 0.4, 0.7, 1.0, 1.2


@dataclass(frozen=True)
class Variant:
    name: str
    scale: float
    ground: str | None


THESIS = Variant("thesis", 1.0, None)
SLIDE = Variant("slide", 1.65, PAGE)


def use(variant: Variant = THESIS) -> Variant:
    """Install the house style. Call once at the top of every figure function."""
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
    """A figure of an exact millimetre width. Never ``bbox_inches='tight'`` (master §5.1)."""
    return plt.figure(
        figsize=(width_mm * MM, width_mm * ratio * MM),
        facecolor=variant.ground or "none",
    )


def range_frame(ax, x=None, y=None) -> None:
    """Trim the spines to the extent of the data, so the axis reports the range."""
    if x is None:
        x = ax.get_xlim()
    if y is None:
        y = ax.get_ylim()
    ax.spines["bottom"].set_bounds(*sorted(x))
    ax.spines["left"].set_bounds(*sorted(y))
    ax.spines["bottom"].set_color(RULE)
    ax.spines["left"].set_color(RULE)


def direct_label(ax, x, y, text, colour=INK, *, dx=2.0, dy=0.0, size=None, style="normal",
                 va="center", ha="left"):
    """Name a series where it terminates, in its own colour. Offsets are in points."""
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


def annotate(ax, x, y, text, *, colour=BRONZE, size=None, ha="left", va="bottom",
             style="italic", transform=None, rotation=0):
    """The one number a reader should take away, set on the drawing at the thing it refers to.

    Coordinates are axes fractions by default, so a value just outside ``[0, 1]`` places the
    text just outside the panel. Pass ``transform=ax.get_xaxis_transform()`` to anchor to a
    data position on x and an axes fraction on y.
    """
    ax.text(
        x, y, text, transform=transform if transform is not None else ax.transAxes,
        color=colour, ha=ha, va=va, style=style, rotation=rotation, clip_on=False,
        fontsize=size or mpl.rcParams["font.size"] * 0.94,
    )


def panel_letter(ax, letter: str, *, dx_mm: float = 7.5, dy_mm: float = 2.0) -> None:
    """A parenthesised small capital, set outside the panel's upper-left corner.

    Placed in *figure* coordinates from the axes bounding box, not in axes coordinates, so
    every letter in a multi-panel figure aligns to the same optical position whether or not
    its panel carries a y-label. Small capitals rather than weight: a panel letter is a
    label on the object and should not compete with the data for the eye.
    """
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


def panel_title(ax, text: str, *, pad: float = 8.0, colour: str = BRONZE) -> None:
    """Name a panel of a small multiple, above its data area, in the accent.

    Small capitals, so that a title naming a condition reads as a label rather than as the
    start of a sentence --- which is the failure §4.5 of the master document is about.
    """
    ax.set_title(text, color=colour, pad=pad, loc="center", family=SMALLCAPS)


def value(ax, x, y, number: str, name: str = "", *, colour=BRONZE, ha="left", va="bottom",
          transform=None, gap: float = 1.35) -> None:
    """A measured number with the quantity it measures set beneath it.

    A value alone in a panel corner is decoration: the reader has to go to the caption to
    learn what was measured, and by then the number has left the figure. The name is set
    smaller and lighter than the number, but not so light that it stops being readable ---
    a name nobody can read is worse than no name, since it still spends the ink.
    """
    axes = transform if transform is not None else ax.transAxes
    size = mpl.rcParams["font.size"]
    ax.text(x, y, number, transform=axes, color=colour, ha=ha, va=va, style="italic",
            clip_on=False, fontsize=size * 0.94)
    if name:
        ax.annotate(name, xy=(x, y), xycoords=axes, xytext=(0, -gap * size), va="top",
                    textcoords="offset points", color=INK, alpha=0.62, ha=ha,
                    clip_on=False, annotation_clip=False, fontsize=size * 0.84)


def key(fig, rect, entries, *, heading: str = "", note: str = "") -> None:
    """The hairline legend block (device 6): a ruled rectangle, a small-caps heading, rows.

    Used only where direct labelling genuinely cannot work --- a glyph family that recurs
    across every point of a scatter, or four series that all terminate in the same corner.
    Each entry is ``(label, draw)``, where ``draw`` receives an axes spanning the swatch
    column and the row's vertical centre in axes coordinates.
    """
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
    step = (top - (0.20 if note else 0.06)) / max(rows, 1)
    for i, (label, draw) in enumerate(entries):
        y = top - step * (i + 0.5)
        draw(ax, y)
        ax.text(0.30, y, label, va="center", ha="left", color=INK, fontsize=size * 0.88)
    if note:
        ax.text(0.055, 0.055, note, va="bottom", ha="left", color=RULE, fontsize=size * 0.80)


def sequential(zero_as_page: bool = True, floor: float = 0.0):
    """The house ramp, with absence rendered as paper rather than as the palest stop.

    Pair with ``vmin`` just above zero so that ``set_under`` catches the empty cells:
    a count of none and a count of one must not differ by a shade. ``floor`` drops the
    palest stops, which is what a field of mostly small values needs --- the ramp's first
    fifth is within a few percent of the page, so a surface that lives there reads as
    blank paper however carefully the data was computed.
    """
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
    """The range a statistic takes when nothing is happening (device 3)."""
    span = ax.axhspan if horizontal else ax.axvspan
    span(lo, hi, facecolor=RULE, alpha=0.18, edgecolor="none", zorder=0)
    line = ax.axhline if horizontal else ax.axvline
    line(1.0, color=RULE, lw=HAIRLINE, zorder=0.5)


def seed_comb(ax, steps, *, height=0.045, colour=SIENNA) -> None:
    """Each seed's transition as a short hairline along the axis (device 1)."""
    for s in steps:
        ax.axvline(s, ymin=0.0, ymax=height, color=colour, lw=0.6, zorder=3, clip_on=False)


def sparkline(ax, values, *, colours=None, band=None, log: bool = False, kind: str = "bar",
              ticks=(0.0, 1.0), label: str = "", height: float = 0.42, offset: float = 0.0,
              size: float = 11.0) -> None:
    """A narrow column beside a forest, carrying one variable per row (device 5).

    Bars where the quantity has a meaningful zero, dots where it does not: a bar drawn
    from the left edge of a logarithmic axis encodes its own axis limit, not its value.

    No y-axis of its own: the rows are the forest's rows, so a second set of labels
    would be a second reading of the same thing.
    """
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


def save(fig, name: str, variant: Variant = THESIS, out_dir=None, svg: bool = True) -> str:
    """Write vector output, text kept as text.

    PDF for the manuscript; SVG alongside it for the talk and for any web use, where the
    text stays live (``svg.fonttype = "none"``) so it re-renders in the viewer's ETbb.
    """
    root = Path(out_dir) if out_dir else OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    suffix = "" if variant.name == "thesis" else f"-{variant.name}"
    path = root / f"{name}{suffix}.pdf"
    fig.savefig(path, format="pdf", transparent=variant.ground is None)
    if svg:
        (root / "svg").mkdir(exist_ok=True)
        fig.savefig(root / "svg" / f"{name}{suffix}.svg", format="svg",
                    transparent=variant.ground is None)
    plt.close(fig)
    return str(path)
