"""Train/test accuracy and loss curves — the canonical grokking picture."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from grokking_tda.plotting.style import GROK_COLOR, use_vector_style


def plot_training_curves(
    metrics: pd.DataFrame, out_path: str | Path, transition: int | None = None
) -> Path:
    """Two-panel accuracy/loss vs step (symlog x to include step 0)."""
    use_vector_style()
    m = metrics.sort_values("step")
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(9, 3.2))

    ax_acc.plot(m["step"], m["train_acc"], label="train")
    ax_acc.plot(m["step"], m["test_acc"], label="test")
    ax_acc.set(xlabel="step", ylabel="accuracy", ylim=(-0.02, 1.02))
    ax_acc.set_xscale("symlog")
    ax_acc.legend(loc="lower right")

    ax_loss.plot(m["step"], m["train_loss"], label="train")
    ax_loss.plot(m["step"], m["test_loss"], label="test")
    ax_loss.set(xlabel="step", ylabel="loss")
    ax_loss.set_xscale("symlog")
    ax_loss.set_yscale("log")
    ax_loss.legend(loc="upper right")

    if transition is not None:
        for ax in (ax_acc, ax_loss):
            ax.axvline(transition, color=GROK_COLOR, ls="--", lw=1)

    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
