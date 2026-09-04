"""Vector-first matplotlib style for run figures, in the house palette."""

from __future__ import annotations

import logging

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# fontTools logs every glyph it subsets at INFO when embedding fonts into PDFs.
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# The house palette, stated once. `analysis/figures/style.py` imports it: the two had been
# written out separately and had already drifted, SAGE holding a different colour in each.
INK = "#1F1B16"  # text / primary series
BRONZE = "#8C6A43"  # warm bronze, secondary series / emphasis
RULE = "#C8B69B"  # light rule / faint series
SAGE = "#8B9179"  # muted cool counterpoint (third series)
STONE = "#8E8276"  # warm grey (fourth series)
SLATE = "#3A5C66"  # cool accent, the fifth series in the manuscript set
SIENNA = "#96402A"  # burnt sienna: marks the grokking step on time axes
PAGE = "#FBFAF7"  # the ground the figures sit on

DIM_COLORS = {0: STONE, 1: BRONZE, 2: SAGE}  # H0 / H1 / H2 in persistence diagrams

SEQUENTIAL = LinearSegmentedColormap.from_list("thesis", ["#F5F1EA", BRONZE, INK])


def use_vector_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["ETbb", "Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.prop_cycle": cycler(color=[INK, BRONZE, SAGE, STONE, RULE]),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": RULE,
            "grid.alpha": 0.4,
            "legend.frameon": False,
            "figure.autolayout": True,
        }
    )
