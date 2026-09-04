"""Significance across the robustness grid, corrected as one family at BY (§3.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import (
    CONDITION_KEYS,
    RATIO_OBSERVABLES,
    condition_label,
    load_bank,
    null_runs,
)
from grokking_tda.evaluation import benjamini_hochberg, benjamini_yekutieli
from grokking_tda.observable import OBSERVABLE_DIRECTION

N_RESAMPLES = 20_000
Q = 0.1  # false-discovery rate, pre-registered in §3.7


def alternative_for(column: str) -> str:
    """Which tail the claim is in, from the observable's own declared direction.

    ``auto`` becomes two-sided rather than being resolved from the data: every resolution
    available here — the sign of the null band, the direction of the observed ratios — would
    pick the tail using the sample under test.
    """
    observable = column.removesuffix("__ratio")
    if observable not in OBSERVABLE_DIRECTION:
        # defaulting would silently move a published row from one tail to two
        raise KeyError(f"{observable} declares no direction; is its module imported?")
    return {"rising": "greater", "falling": "less"}.get(
        OBSERVABLE_DIRECTION[observable], "two-sided"
    )


def _tail_p(draws: np.ndarray, observed: float, alternative: str) -> float:
    upper = (np.sum(draws >= observed) + 1) / (N_RESAMPLES + 1)
    lower = (np.sum(draws <= observed) + 1) / (N_RESAMPLES + 1)
    if alternative == "greater":
        return float(upper)
    if alternative == "less":
        return float(lower)
    return float(min(1.0, 2 * min(upper, lower)))


def null_configurations(bank: pd.DataFrame, column: str) -> list[np.ndarray]:
    frame = null_runs(bank)
    keys = frame[CONDITION_KEYS].astype(str).agg("|".join, axis=1)
    out = []
    for _, sub in frame.groupby(keys):
        values = np.asarray(sub[column], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            out.append(values)
    return out


def null_medians(
    groups: list[np.ndarray], n: int, *, rng: np.random.Generator, draws: int
) -> np.ndarray:
    picked = rng.integers(len(groups), size=draws)
    out = np.empty(draws)
    for i, g in enumerate(picked):
        values = groups[g]
        out[i] = np.median(values[rng.integers(values.size, size=n)])
    return out


def resampled_pvalues(bank: pd.DataFrame, column: str, *, seed: int = 0) -> pd.DataFrame:
    groups = null_configurations(bank, column)
    alternative = alternative_for(column)
    rng = np.random.default_rng(seed)
    tested = bank[~bank.replicate] if "replicate" in bank else bank
    # a null must not be tested against a distribution it helps define
    tested = tested.drop(index=null_runs(bank).index, errors="ignore")
    rows = []
    for key, sub in tested.groupby(CONDITION_KEYS, dropna=False):
        values = np.asarray(sub[column], dtype=float)
        values = values[np.isfinite(values)]
        row = dict(zip(CONDITION_KEYS, key, strict=True))
        row["n"] = int(values.size)
        row["alternative"] = alternative
        if values.size == 0 or len(groups) < 2:
            row["observed"], row["p"] = float("nan"), float("nan")
            rows.append(row)
            continue
        observed = float(np.median(values))
        # same size, so a small condition cannot look significant on a noisy median
        draws = null_medians(groups, values.size, rng=rng, draws=N_RESAMPLES)
        row["observed"] = observed
        row["p"] = _tail_p(draws, observed, alternative)
        rows.append(row)
    frame = pd.DataFrame(rows)
    raw = frame["p"].to_numpy()
    # Every condition is tested against the same null bank, so a bank that happens to sit high
    # depresses all of them at once. BY is valid under that dependence and BH is not, so the
    # grid is read at BY and BH is reported beside it.
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
        resampled_pvalues(bank, f"{column}__ratio")
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
        label = condition_label(r)
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
