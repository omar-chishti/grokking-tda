"""CROCKER plot — a (time x scale) heatmap of Betti numbers over the trajectory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grokking_tda.plotting.style import SEQUENTIAL, SIENNA, use_vector_style


def plot_crocker(
    matrix: np.ndarray,
    steps: np.ndarray,
    scales: np.ndarray,
    out_path: str | Path,
    *,
    homology_dim: int = 1,
    grokking_step: int | None = None,
) -> Path:
    use_vector_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    vmax = max(int(matrix.max()), 1) if matrix.size else 1
    mesh = ax.pcolormesh(
        scales, steps, matrix, shading="nearest", cmap=SEQUENTIAL, vmin=0, vmax=vmax
    )
    ax.set_yscale("symlog")
    ax.set(xlabel="filtration scale", ylabel="training step")
    if grokking_step is not None:
        ax.axhline(grokking_step, color=SIENNA, ls="--", lw=1, label="grokking")
        ax.legend(loc="upper right")
    fig.colorbar(mesh, ax=ax, label=rf"$\beta_{{{homology_dim}}}$ (alive features)")
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
