"""Significance across the robustness grid, corrected as one family.

One of the two things the thesis protocol promises and nothing else computes; the other,
the partial information decomposition, outgrew this module and lives in ``analysis/pid.py``,
which estimates it per regime under two redundancy functions with intervals and nulls.

**Significance with multiplicity.** Each condition gets a permutation p-value: how often the
pooled null runs — permuted labels and the polynomial, which fit their training data and
never generalise — produce a median ratio as large as the condition's. Those p-values are
corrected across the whole grid, because the grid is one family of tests fixed in advance by
the run manifests. Both corrections are reported beside the raw p-values, and the headline
is the conservative one: every condition is tested against **the same sixteen null runs**, so
the tests share a denominator and positive regression dependence cannot be argued from the
construction — a null bank that happens to sit high depresses every condition at once.
Benjamini-Yekutieli is valid under that dependence and Benjamini-Hochberg is not, so BY is
what the grid is read at and BH is shown for comparison.

    uv run python -m analysis.redundancy --root results/raw --out results/processed/thesis
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import CONDITION_KEYS, RATIO_OBSERVABLES, load_bank, null_runs
from grokking_tda.evaluation import benjamini_hochberg, benjamini_yekutieli

N_PERMUTATIONS = 20_000
Q = 0.1  # false-discovery rate, pre-registered in thesis section 3.7


def permutation_pvalues(
    bank: pd.DataFrame, column: str, *, seed: int = 0
) -> pd.DataFrame:
    """Per-condition p-value against the pooled null runs, for one ratio observable."""
    nulls = np.asarray(null_runs(bank)[column], dtype=float)
    nulls = nulls[np.isfinite(nulls)]
    rng = np.random.default_rng(seed)
    tested = bank[~bank.dense] if "dense" in bank else bank
    # A null condition must not be tested against a distribution it helps define: it would
    # be compared with itself, and any rejection would be circular rather than a discovery.
    tested = tested.drop(index=null_runs(bank).index, errors="ignore")
    rows = []
    for key, sub in tested.groupby(CONDITION_KEYS, dropna=False):
        values = np.asarray(sub[column], dtype=float)
        values = values[np.isfinite(values)]
        row = dict(zip(CONDITION_KEYS, key, strict=True))
        row["n"] = int(values.size)
        if values.size == 0 or nulls.size < 4:
            row["observed"], row["p"] = float("nan"), float("nan")
            rows.append(row)
            continue
        observed = float(np.median(values))
        # Draw pseudo-conditions of the same size from the nulls: a small condition must
        # not look significant merely because a median over few seeds is noisy.
        draws = np.median(rng.choice(nulls, size=(N_PERMUTATIONS, values.size)), axis=1)
        row["observed"] = observed
        row["p"] = float((np.sum(draws >= observed) + 1) / (N_PERMUTATIONS + 1))
        rows.append(row)
    frame = pd.DataFrame(rows)
    raw = frame["p"].to_numpy()
    bh_rejected, bh_adjusted = benjamini_hochberg(raw, q=Q)
    by_rejected, by_adjusted = benjamini_yekutieli(raw, q=Q)
    frame["p_bh"], frame["significant_bh"] = bh_adjusted, bh_rejected
    frame["p_by"], frame["significant_by"] = by_adjusted, by_rejected
    frame["observable"] = column
    return frame




def main() -> None:
    parser = cli.parser(__doc__)
    args = parser.parse_args()

    bank, fourier_k = load_bank(args.root)
    args.out.mkdir(parents=True, exist_ok=True)

    frames = [
        permutation_pvalues(bank, f"{column}__ratio")
        for column in RATIO_OBSERVABLES
        if f"{column}__ratio" in bank
    ]
    significance = pd.concat(frames, ignore_index=True)
    significance.to_csv(args.out / "significance.csv", index=False)

    headline = "h1_max_persistence_normalised__ratio"
    table = significance[significance.observable == headline].sort_values("p")
    print(f"fourier k selected: {fourier_k}")
    print(f"\nrobustness grid at q={Q:g}, on {headline}:")
    for _, r in table[table.significant_bh | table.significant_by].iterrows():
        label = f"{r.model} {r.operation}{r.modulus:g} f{r.train_fraction:g} wd{r.weight_decay:g}"
        mark = "BY" if r.significant_by else "BH"
        print(
            f"  {label:<46} ratio {r.observed:5.2f}  p={r.p:.4f}"
            f"  BH {r.p_bh:.4f}  BY {r.p_by:.4f}  [{mark}]"
        )
    print(
        f"  ({int(table.significant_by.sum())} of {len(table)} survive Benjamini-Yekutieli; "
        f"{int(table.significant_bh.sum())} survive Benjamini-Hochberg)"
    )

    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
