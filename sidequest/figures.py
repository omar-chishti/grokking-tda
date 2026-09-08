"""The four operator figures of R24, in the house system (Documentation/Figures/S-operator.md).

They live here rather than in ``analysis/figures`` because the side quest is not thesis work and
must not reach the manuscript's build. The drawing style is shared, since a figure that came out
looking like an import would defeat the point of having a house style at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from analysis.figures import style
from analysis.figures.style import BRONZE, INK, RULE, SIENNA, SLATE

PROCESSED = Path("results-sidequest/processed")
HERO = "mlp_addsub97_f0.3_wd1.0_sq-addsub_s4"
MARKED_RESIDUE = 24  # shown as b and -b on the two panels, to check the reflection by eye
NULL = "transformer_addsub113_f0.3_wd0.1_PERM_sq-addsub_s0"
REFERENCE = "transformer_addsub113_f0.3_wd0.1_sq-addsub_s2"
CHANCE = np.sqrt(2.0)  # two uncorrelated centred loops of equal norm

# The grokking step of each transformer seed, from the training curves.
TG = {"s0": 13000, "s1": 12500, "s2": 12000, "s3": 12000, "s4": 10000}


def _canvas(ratio: float, variant) -> object:
    """The house canvas, at the width the destination places it: a slide is not a column."""
    width = style.TALK_PLATE_W if variant is style.TALK else style.FULL
    return style.figure(width, ratio, variant)


def _save(fig, name: str, variant, out_dir) -> str:
    """The deck names its figures without a variant tag, since each is placed at 1:1."""
    if variant is style.TALK:
        return style.save(fig, f"talk-{name}", variant, out_dir, svg=False, suffix="")
    return style.save(fig, name, variant, out_dir)


def _thread(ax, loop: pd.DataFrame, colour: str, *, width: float, alpha: float = 1.0) -> None:
    """The loop as its own chords: p vertices joined in b-order and closed.

    At winding k this is the star polygon {p/k}, so nothing curved is drawn and the circle
    appears as the envelope of the chords -- the construction of the title-page emblem.
    """
    xy = loop.sort_values("b")[["x", "y"]].to_numpy()
    closed = np.vstack([xy, xy[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=colour, lw=width, alpha=alpha,
            solid_joinstyle="round", zorder=3)


def s1_mirrored_thread(variant=style.THESIS, out_dir=None) -> str:
    """S1. One dial, read forwards to add and backwards to subtract."""
    style.use(variant)
    loops = pd.read_csv(PROCESSED / "loops.csv")
    run = loops[loops["run"] == HERO]
    a0 = int(run["a0"].iloc[0])
    run = run[run["a0"] == a0]

    # A column takes the plate on two rows; a slide is wider than it is tall and takes the same
    # three panels in one. The claim is unchanged, so only the packing is.
    if variant is style.TALK:
        fig = _canvas(0.34, variant)
        grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.7), wspace=0.30,
                                left=0.02, right=0.87, top=0.88, bottom=0.20)
        threads = [fig.add_subplot(grid[0, i]) for i in (0, 1)]
        unrolled = fig.add_subplot(grid[0, 2])
    else:
        fig = _canvas(0.72, variant)
        grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.62), hspace=0.34, wspace=0.10,
                                left=0.09, right=0.88, top=0.95, bottom=0.10)
        threads = [fig.add_subplot(grid[0, i]) for i in (0, 1)]
        unrolled = fig.add_subplot(grid[1, :])

    modulus = int(run["b"].max()) + 1
    xy = run[["x", "y"]].to_numpy()
    centre = xy.mean(0)
    radius = float(np.hypot(*(xy - centre).T).mean())
    lim = radius * 1.28

    b0 = run[(run["operator"] == "add") & (run["b"] == 0)][["x", "y"]].to_numpy()[0]
    mirror = (b0 - centre) / np.linalg.norm(b0 - centre) * radius * 1.24

    for ax, operator, colour, name in ((threads[0], "add", INK, "$a + b$"),
                                       (threads[1], "sub", BRONZE, "$a - b$")):
        frame = run[run["operator"] == operator].sort_values("b")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(mpl.patches.Circle(centre, radius, fill=False, color=RULE,
                                        lw=style.HAIRLINE, ls=(0, (1, 3)), zorder=0))
        # the fixed line of the reflection, which is the shift the anchors agree on
        ax.plot(*np.array([centre - mirror, centre + mirror]).T,
                color=RULE, lw=style.HAIRLINE, zorder=1)
        _thread(ax, frame, colour, width=style.HAIRLINE * 1.5, alpha=0.72)
        # device 11: the group's own step, drawn once, at emphasis weight -- the same mark on
        # both panels, and the geometry it lands in is the comparison
        p0, p1 = frame[["x", "y"]].to_numpy()[:2]
        ax.annotate("", xy=p1, xytext=p0, zorder=4, arrowprops={
            "arrowstyle": "-|>,head_width=0.22,head_length=0.48", "color": colour,
            "lw": style.EMPHASIS * 1.6, "shrinkA": 0, "shrinkB": 0})
        ax.plot(*p0, "o", color=colour, ms=2.6, zorder=5)
        # one residue and its negative, marked on each panel: under the reflection the two land
        # in the same place, which is the claim of B.3 at a single point
        marked = MARKED_RESIDUE if operator == "add" else modulus - MARKED_RESIDUE
        q = frame[frame["b"] == marked][["x", "y"]].to_numpy()[0]
        ax.plot(*q, "o", mfc="none", mec=colour, ms=6.4, mew=style.DATA, zorder=5)
        style.direct_label(ax, *q, f"${'' if operator == 'add' else '-'}{MARKED_RESIDUE}$",
                           colour, dx=4.5, size=7.0)
        ax.set_xlim(centre[0] - lim, centre[0] + lim)
        ax.set_ylim(centre[1] - lim, centre[1] + lim)
        style.panel_title(ax, name, colour=colour, pad=2.0)
        style.panel_letter(ax, "a" if operator == "add" else "b", dx_mm=2.0, dy_mm=-1.0)

    # (c) the degree itself: the angle each loop accumulates, which is a straight line of
    # slope +/- k and needs no gate, no plane and no threshold to be read
    sense = 0.0
    for operator, colour in (("add", INK), ("sub", BRONZE)):
        frame = run[run["operator"] == operator].sort_values("b")[["x", "y"]].to_numpy()
        turns = np.unwrap(np.arctan2(*(frame - frame.mean(0)).T[::-1])) / (2 * np.pi)
        turns = turns - turns[0]
        # a principal axis carries no sign of its own, so addition is the one drawn rising
        sense = sense or np.sign(turns[-1])
        turns = turns * sense
        unrolled.plot(np.arange(len(turns)), turns, color=colour, lw=style.DATA)
        style.direct_label(unrolled, len(turns) - 1, turns[-1],
                           "$a + b$" if operator == "add" else "$a - b$", colour, dx=3.0)

    null = loops[loops["run"] == NULL]
    null = null[null["a0"] == null["a0"].iloc[0]]
    for operator in ("add", "sub"):
        frame = null[null["operator"] == operator].sort_values("b")[["x", "y"]].to_numpy()
        turns = np.unwrap(np.arctan2(*(frame - frame.mean(0)).T[::-1])) / (2 * np.pi)
        unrolled.plot(np.linspace(0, modulus - 1, len(turns)), (turns - turns[0]) * sense,
                      color=RULE, lw=style.SECONDARY, zorder=1)
    style.direct_label(unrolled, modulus - 1, 0.4, "permuted\nlabels", RULE, dx=3.0)

    unrolled.set_xlim(0, modulus - 1)
    unrolled.set_ylim(-17, 17)
    unrolled.set_yticks([-16, 0, 16])
    unrolled.set_xticks([0, modulus - 1])
    unrolled.set_xlabel("$b$")
    unrolled.set_ylabel("turns accumulated")
    style.range_frame(unrolled, (0, modulus - 1), (-16, 16))
    style.panel_letter(unrolled, "c", dx_mm=8.0)
    style.value(unrolled, 0.035, 0.92, "−1.000", "winding antisymmetry")
    style.value(unrolled, 0.035, 0.26, "0.21", "reflection residual")
    return _save(fig, "s1-mirrored-thread", variant, out_dir)


def s2_shift_landscape(variant=style.THESIS, out_dir=None) -> str:
    """S2. The re-indexing the model picks, and the p-1 it refuses."""
    style.use(variant)
    land = pd.read_csv(PROCESSED / "shift_landscape.csv")
    fig = _canvas(0.36, variant)
    axes = fig.subplots(1, 2, sharey=True,
                        gridspec_kw={"wspace": 0.07, "bottom": 0.22, "top": 0.86,
                                     "left": 0.09, "right": 0.98})

    for ax, run, title in ((axes[0], REFERENCE, "reference"), (axes[1], NULL, "permuted labels")):
        frame = land[land["run"] == run]
        modulus = int(frame["shift"].max()) + 1
        # centred, so the answer sits at the middle of the field rather than its left edge
        frame = frame.assign(centred=((frame["shift"] + modulus // 2) % modulus) - modulus // 2)
        style.null_band(ax, CHANCE - 0.035, CHANCE + 0.035)
        ax.axvline(0.0, color=SIENNA, lw=style.HAIRLINE, zorder=1)
        for family, colour in (("rotated", SLATE), ("reflected", BRONZE)):
            f = frame[frame["family"] == family]
            for _, anchor in f.groupby("a0"):
                a = anchor.sort_values("centred")
                ax.plot(a["centred"], a["residual"], color=colour, lw=style.HAIRLINE, alpha=0.11)
            median = f.groupby("centred")["residual"].median().sort_index()
            ax.plot(median.index, median.to_numpy(), color=colour, lw=style.DATA, zorder=3)
        ax.set_xlim(-modulus // 2, modulus // 2)
        ax.set_ylim(0, 1.62)
        ax.set_yticks([0, 1, CHANCE])
        ax.set_yticklabels(["0", "1", "$\\sqrt{2}$"])
        ax.set_xticks([-modulus // 2, 0, modulus // 2])
        ax.set_xlabel("candidate shift $s$")
        style.panel_title(ax, title)
        style.range_frame(ax, (-modulus // 2, modulus // 2), (0, 1.6))
        style.panel_letter(ax, "ab"[ax is axes[1]], dx_mm=6.0)

    axes[0].set_ylabel("relative residual")
    median = land[(land["run"] == REFERENCE) & (land["family"] == "reflected")]
    at_zero = median[median["shift"] == 0]["residual"].median()
    if variant is not style.TALK:  # on a slide the takeaway line carries these
        style.value(axes[0], 0.06, 0.16, f"{at_zero:.3f}", "residual at $s = 0$")
        style.value(axes[0], 0.42, 0.16, "16/16", "anchors agree")
    at_null = land[(land["run"] == NULL) & (land["family"] == "rotated")]
    if variant is not style.TALK:
        style.value(axes[1], 0.06, 0.30,
                    f"{at_null[at_null['shift'] == 0]['residual'].median():.3f}",
                    "best rotation, $s = 0$", colour=SLATE)
    style.value(axes[0], 0.06, 0.62, "reflected", "", colour=BRONZE)
    style.value(axes[0], 0.06, 0.48, "rotated", "", colour=SLATE)
    return _save(fig, "s2-shift-landscape", variant, out_dir)


def s3_leak_split(variant=style.THESIS, out_dir=None) -> str:
    """S3. The leak is in the data; only attention takes it."""
    style.use(variant)
    leak = pd.read_csv(PROCESSED / "leak_by_operator.csv")
    leak = leak[~leak["run"].str.contains("PERM")]
    wide = leak.pivot_table(index=["run", "step"], columns="operator",
                            values="test_acc").reset_index()

    fig = _canvas(0.36, variant)
    axes = fig.subplots(1, 2, sharey=True,
                        gridspec_kw={"wspace": 0.07, "bottom": 0.22, "top": 0.86,
                                     "left": 0.09, "right": 0.98})
    for ax, prefix, modulus, title in ((axes[0], "transformer", 113, "transformer"),
                                       (axes[1], "mlp", 97, "multilayer perceptron")):
        frame = wide[wide["run"].str.startswith(prefix)]
        # the registered prediction: a held-out sum is answered iff its transpose was memorised,
        # so a memorising model sits at the train fraction
        ax.axhline(0.30, color=BRONZE, alpha=0.35, lw=style.SECONDARY, zorder=0)
        style.direct_label(ax, 2.6e4, 0.30, "$f$", BRONZE, dy=2.0, va="bottom", size=7.0)
        ax.axhline(1 / modulus, color=RULE, lw=style.HAIRLINE, zorder=1)
        for _, run in frame.groupby("run"):
            run = run.sort_values("step")
            ax.plot(run["step"], run["add"], color=INK, lw=style.SECONDARY, alpha=0.85)
            ax.plot(run["step"], run["sub"], color=BRONZE, lw=style.SECONDARY, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlim(700, 3e4)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("training step")
        style.panel_title(ax, title)
        style.range_frame(ax, (700, 3e4), (0, 1.0))
        style.panel_letter(ax, "ab"[ax is axes[1]], dx_mm=6.0)
        if variant is not style.TALK:
            style.direct_label(ax, 2.6e4, 1 / modulus, "$1/p$", RULE, dy=-3.0, va="top", size=7.0)
        style.seed_comb(ax, sorted(frame.groupby("run").apply(
            lambda r: r.loc[r["add"].ge(0.9).idxmax(), "step"] if r["add"].max() > 0.9 else np.nan)
            .dropna()))

    axes[0].set_ylabel("held-out accuracy")
    # the memorisation plateau, defined as the text defines it: after train convergence and
    # before the transition, so the figure and the numbers beside it cannot disagree
    plateau = wide[wide["run"].str.startswith("transformer")]
    plateau = plateau[(plateau["step"] >= 1500) & (plateau["add"] < 0.5)]
    if variant is not style.TALK:
        style.value(axes[0], 0.05, 0.90, f"{plateau['add'].median():.3f}", "plateau, $a + b$")
        style.value(axes[0], 0.05, 0.66, f"{plateau['sub'].median():.3f}", "plateau, $a - b$")
    style.direct_label(axes[0], 6.5e3, plateau["add"].median(), "$a + b$", INK,
                       ha="right", dy=5.0)
    style.direct_label(axes[0], 6.5e3, plateau["sub"].median(), "$a - b$", BRONZE,
                       ha="right", dy=5.0)
    return _save(fig, "s3-leak-split", variant, out_dir)


def _fraction_of_own_change(step, y, rising):
    """A series rescaled to the fraction of its own total change, which is how §11.1 times them.

    Two series of different units and opposite sense cannot be compared on one axis; two series
    each expressed as *how far through its own transition it is* can, and the horizontal gap
    between them is then the lag itself rather than an artefact of scaling.
    """
    base = np.median(y[: max(3, len(y) // 5)])
    final = np.median(y[-max(3, len(y) // 5):])
    scaled = (y - base) / (final - base + 1e-12)
    return step, np.clip(scaled, -0.1, 1.1) if rising else np.clip(scaled, -0.1, 1.1)


def s4_timing(variant=style.THESIS, out_dir=None) -> str:
    """S4. The reflection forms after the transition, not before it."""
    style.use(variant)
    fine = pd.read_csv(Path("results-sidequest/processed-fine") / "orientation_trace.csv")
    coarse = pd.read_csv(PROCESSED / "orientation_trace.csv")
    nulls = coarse[coarse["run"].str.contains("PERM")]

    fig = _canvas(0.40, variant)
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.22, "bottom": 0.20, "top": 0.86,
                                           "left": 0.08, "right": 0.98})

    ax = axes[0]
    style.null_band(ax, CHANCE - 0.05, CHANCE + 0.05)
    for _, run in nulls.groupby("run"):
        run = run.sort_values("step")
        ax.plot(run["step"], run["residual_reflected_median"], color=RULE,
                lw=style.SECONDARY, zorder=2)
    for _, run in fine.groupby("run"):
        run = run.sort_values("step")
        ax.plot(run["step"], run["residual_reflected_median"], color=BRONZE,
                lw=style.DATA, alpha=0.85, zorder=3)
    ax.set_xlim(0, 40000)
    ax.set_ylim(1.62, 0)  # inverted: the reflection forming reads as a rise
    ax.set_xticks([0, 20000, 40000])
    ax.set_yticks([0, 1, CHANCE])
    ax.set_yticklabels(["0", "1", "$\\sqrt{2}$"])
    ax.set_ylabel("relative residual")
    style.seed_comb(ax, list(TG.values()))
    style.range_frame(ax, (0, 40000), (0, 1.6))
    style.panel_title(ax, "the whole of training")
    style.direct_label(ax, 38000, 0.06, "reference", BRONZE, ha="right", dy=6.0)
    style.direct_label(ax, 38000, 1.46, "permuted labels", RULE, ha="right", dy=-5.0)

    # (b) both series as the fraction of their own transition, which is what §11.1 times and the
    # only way a lag between an accuracy and a residual can be a horizontal distance on one axis
    ax = axes[1]
    crossings: dict[str, list[float]] = {}
    for run, frame in fine.groupby("run"):
        frame = frame.sort_values("step")
        observed = pd.read_csv(Path("results-sidequest/raw") / run / "analysis" / "observables.csv")
        observed = observed.dropna(subset=["test_acc"]).sort_values("step")
        series = {
            "accuracy": (_fraction_of_own_change(observed["step"].to_numpy(float),
                                                 observed["test_acc"].to_numpy(float), True), INK),
            "reflection": (_fraction_of_own_change(
                frame["step"].to_numpy(float),
                frame["residual_reflected_median"].to_numpy(float), False), BRONZE),
        }
        for name, ((x, y), colour) in series.items():
            ax.plot(x, y, color=colour, lw=style.SECONDARY, alpha=0.85)
            for level in (0.25, 0.75):
                crossings.setdefault(f"{name}{level}", []).append(float(np.interp(level, y, x)))
    ax.axhline(0.5, color=RULE, lw=style.HAIRLINE, zorder=0)
    ax.set_xlim(6000, 20500)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xticks([6000, 13000, 20000])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("fraction of own transition")
    style.range_frame(ax, (6000, 20000), (0, 1))
    style.panel_title(ax, "the transition")
    # each named on its own curve, and at different heights, so two labels a thousand steps
    # apart do not have to share a line
    style.direct_label(ax, np.median(crossings["accuracy0.75"]), 0.75, "test accuracy", INK,
                       dx=-3.0, ha="right")
    style.direct_label(ax, np.median(crossings["reflection0.25"]), 0.25, "reflection", BRONZE,
                       dx=3.0)
    if variant is not style.TALK:  # on a slide the takeaway line carries these
        # low on the right, the one quarter of the panel both series have left by then: on the
        # left they ran into the curve labels, and lower still the second name met the axis
        style.value(ax, 0.58, 0.42, "+890", "median lag")
        style.value(ax, 0.58, 0.22, "+382 to +3,698", "quantile sweep")

    for ax, letter in zip(axes, "ab", strict=True):
        ax.set_xlabel("training step")
        style.panel_letter(ax, letter, dx_mm=7.0)
    return _save(fig, "s4-timing", variant, out_dir)


FIGURES = {"s1": s1_mirrored_thread, "s2": s2_shift_landscape,
           "s3": s3_leak_split, "s4": s4_timing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=sorted(FIGURES))
    ap.add_argument("--variant", choices=["thesis", "talk"], default="thesis")
    ap.add_argument("--out", type=Path, default=Path("results-sidequest/figures"))
    args = ap.parse_args()
    variant = {"thesis": style.THESIS, "talk": style.TALK}[args.variant]
    for name in args.only:
        print(" ", FIGURES[name](variant, args.out))


if __name__ == "__main__":
    main()
