"""Topology figures: observables over training, and persistence diagrams."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grokking_tda.plotting.style import DIM_COLORS as _DIM_COLORS
from grokking_tda.plotting.style import GROK_COLOR, use_vector_style
from grokking_tda.tda.summaries import finite_bars


def plot_observables_over_time(
    observables: pd.DataFrame,
    out_path: str | Path,
    markers: dict[str, int | None] | None = None,
) -> Path:
    """One panel per observable, value vs step, with optional transition markers."""
    use_vector_style()
    columns = [c for c in observables.columns if c != "step"]
    obs = observables.sort_values("step")
    n = max(len(columns), 1)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 2.8), squeeze=False)
    for ax, column in zip(axes[0], columns, strict=False):
        ax.plot(obs["step"], obs[column], marker="o", ms=3)
        ax.set_title(column, fontsize=8)
        ax.set_xlabel("step")
        ax.set_xscale("symlog")
        for label, value in (markers or {}).items():
            if value is not None:
                ax.axvline(value, color=GROK_COLOR, ls="--", lw=1, label=label)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_persistence_diagram(diagrams: dict[int, np.ndarray], out_path: str | Path) -> Path:
    """Birth-death scatter for each homology dimension (infinite bars clamped to top)."""
    use_vector_style()
    fig, ax = plt.subplots(figsize=(4, 4))
    bars = [b for b in map(finite_bars, diagrams.values()) if b.size]
    top = max((b[:, 1].max() for b in bars), default=1.0) * 1.1
    ax.plot([0, top], [0, top], color="grey", lw=0.8, zorder=0)
    for dim, dgm in diagrams.items():
        if dgm is None or dgm.size == 0:
            continue
        births, deaths = dgm[:, 0].copy(), dgm[:, 1].copy()
        deaths[~np.isfinite(deaths)] = top  # clamp essential classes to the top edge
        ax.scatter(births, deaths, s=18, c=_DIM_COLORS.get(dim, "k"), label=f"H{dim}", alpha=0.8)
    ax.set(xlabel="birth", ylabel="death", xlim=(0, top), ylim=(0, top))
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
