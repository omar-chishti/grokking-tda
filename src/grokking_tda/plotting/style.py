"""Vector-first matplotlib style for thesis figures, in the thesis house palette.

``pdf.fonttype = 42`` embeds editable TrueType fonts (not bitmaps), and we keep
everything vector so figures scale cleanly in the manuscript. Colours mirror the
LaTeX house palette (``LaTeX/shared/styles/palette.tex``): ThesisInk ``#1F1B16``,
ThesisAccent (bronze) ``#8C6A43``, ThesisRule ``#C8B69B`` — with two muted
companions for multi-series plots. Serif font preferences fall back gracefully
where ETbb is not installed.
"""

from __future__ import annotations

import logging

import matplotlib as mpl
from cycler import cycler

# fontTools logs every glyph it subsets at INFO when embedding fonts into PDFs;
# that floods our console, so quiet it (and matplotlib's own chatter).
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# --- the thesis palette -------------------------------------------------------
INK = "#1F1B16"  # text / primary series
ACCENT = "#8C6A43"  # warm bronze, secondary series / emphasis
RULE = "#C8B69B"  # light rule / faint series
SAGE = "#5E6B5A"  # muted cool counterpoint (third series)
STONE = "#8E8276"  # warm grey (fourth series)

GROK_COLOR = "#96402A"  # burnt sienna: marks the grokking step on time axes
DIM_COLORS = {0: STONE, 1: ACCENT, 2: SAGE}  # H0 / H1 / H2 in persistence diagrams


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
            "axes.prop_cycle": cycler(color=[INK, ACCENT, SAGE, STONE, RULE]),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": RULE,
            "grid.alpha": 0.4,
            "legend.frameon": False,
            "figure.autolayout": True,
        }
    )
