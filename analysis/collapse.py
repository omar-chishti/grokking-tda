"""Is the signature a function of dimensional collapse, and where is it not? (sections 4.5, 5.3)

Section 4.5 establishes that the topological signature tracks how circular the learned embedding
becomes, and measures that with a Fourier concentration. The measure has a boundary the chapter does
not state: Fourier concentration along the residue axis is only meaningful when the residues are the
group, so every permutation-group run is absent from the association. That is the one condition the
argument most needs, because section 5.3 reports the largest total-persistence ratio in the bank on
a task with no circle at all. As written, the two sections do not meet.

The spectrum meets them. Effective rank of the centred embedding is defined for any task, takes no
basis and counts no loops, and it orders the modular conditions the same way circularity does. So
the general form of the boundary is dimensional collapse, of which circularity is the cyclic-group
special case, and the question section 5.3 should be asking becomes quantitative: does the
non-abelian task carry *more* loop structure than its own collapse accounts for?

The law is fitted on the modular runs alone --- log2 persistence ratio against log2 effective-rank
ratio --- and every condition is then scored by its mean residual from it, in units of the fit's own
residual spread. Conditions are ranked, so the headline is a rank rather than a t-statistic: with
eighteen conditions, first place is worth p = 1/18 and nothing stronger is claimed.

Usage (from ``Code/``)::

    uv run python -m analysis.collapse
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis import cli
from analysis.bank import HEADLINE_OBSERVABLE, load_bank

RATIO = f"{HEADLINE_OBSERVABLE}__ratio"
COLLAPSE = "effective_rank__ratio"
NON_CYCLIC = "compose"  # the operation whose solution admits no circle


def frame(root, out) -> pd.DataFrame:
    """Grokking runs carrying both a persistence ratio and a spectrum, one row each."""
    bank, _ = load_bank(root)
    spectra = pd.read_csv(out / "normalisation.csv")[["run", COLLAPSE]]
    frame = bank.merge(spectra, on="run", how="inner")
    frame = frame[frame.t_g.notna()]
    if "replicate" in frame:
        frame = frame[~frame.replicate]
    frame = frame[(frame[RATIO] > 0) & (frame[COLLAPSE] > 0)].dropna(subset=[RATIO, COLLAPSE])
    frame["log_ratio"] = np.log2(frame[RATIO])
    frame["log_collapse"] = np.log2(frame[COLLAPSE])
    frame["condition"] = (
        frame.model + " " + frame.operation + " p" + frame.modulus.astype(str)
        + " f" + frame.train_fraction.astype(str) + " wd" + frame.weight_decay.astype(str)
        + " " + frame.loss
    )
    return frame


def main() -> None:
    args = cli.parser(__doc__).parse_args()
    data = frame(args.root, args.out)
    cyclic = data.operation != NON_CYCLIC

    fit = stats.linregress(data.log_collapse[cyclic], data.log_ratio[cyclic])
    data["residual"] = data.log_ratio - (fit.intercept + fit.slope * data.log_collapse)
    spread = data.residual[cyclic].std()
    data["residual_sd"] = data.residual / spread

    by_condition = (
        data.groupby("condition")
        .agg(n=("run", "size"), operation=("operation", "first"),
             collapse=(COLLAPSE, "median"), ratio=(RATIO, "median"),
             excess=("residual_sd", "mean"))
        .sort_values("excess", ascending=False)
    )
    by_condition.insert(0, "rank", np.arange(1, len(by_condition) + 1))

    args.out.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out / "collapse_runs.csv", index=False)
    by_condition.to_csv(args.out / "collapse_conditions.csv")

    print(
        f"the law, fitted on {int(cyclic.sum())} cyclic-task runs:\n"
        f"  log2(persistence ratio) = {fit.intercept:+.3f} {fit.slope:+.3f} * log2(rank ratio)"
        f"   r = {fit.rvalue:.3f}, p = {fit.pvalue:.1e}, slope se {fit.stderr:.3f}\n"
        f"  residual spread {spread:.3f} log2 units\n"
    )
    for label, mask in (("cyclic tasks", cyclic), ("all tasks", pd.Series(True, data.index))):
        rho, p = stats.spearmanr(data.log_collapse[mask], data.log_ratio[mask])
        print(f"  {label:14s} n={int(mask.sum()):3d}  Spearman rho = {rho:+.3f} (p = {p:.1e})")

    print("\nexcess loop structure over the law, by condition:")
    for name, row in by_condition.iterrows():
        mark = "  <- non-cyclic" if row.operation == NON_CYCLIC else ""
        print(f"  {int(row['rank']):2d}. {name:<52s} n={int(row.n):2d}  {row.excess:+.2f} sd{mark}")

    non_cyclic = by_condition[by_condition.operation == NON_CYCLIC]
    if not non_cyclic.empty:
        place, total = int(non_cyclic["rank"].iloc[0]), len(by_condition)
        excess = non_cyclic.excess.iloc[0]
        print(
            f"\nthe non-cyclic task ranks {place} of {total} conditions at {excess:+.2f} sd "
            f"(p = {place / total:.3f} against a uniform rank)"
        )
    print(f"\nwritten to {args.out}/collapse_conditions.csv")


if __name__ == "__main__":
    main()
