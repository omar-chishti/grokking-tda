"""False-discovery control across the robustness grid, under a stated dependence assumption.

The robustness map tests one hypothesis — does the signature rise — over a grid of seeds,
primes, fractions, operations and architectures. Reporting each cell at its own alpha
would guarantee false positives at that many tests, so the grid is corrected as a family.
The grid is fixed in advance by the run manifests, which is what makes the correction
meaningful rather than a choice made after seeing which cells survived.

False-discovery rather than family-wise control, because controlling the *proportion* of
false discoveries keeps power that Bonferroni would spend on a grid this size.

**Which procedure, and why it matters here.** Benjamini-Hochberg controls the FDR under
independence or positive regression dependence (PRDS); Benjamini-Yekutieli controls it
under *arbitrary* dependence, at the cost of a log-factor in power. The cells of this grid
are not independent: every condition's p-value is computed against **the same sixteen null
runs**, so the test statistics share a denominator and their errors are common, not
merely correlated through the design. A shared null bank induces dependence of a sign that
is not obviously positive — a null bank that happens to sit high makes every condition
look *less* significant at once — so PRDS cannot be argued from the construction, and BY
is the defensible default. Both are provided, and the caller reports which it used.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values, q: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rejected, adjusted)`` for a family of p-values at FDR ``q``.

    ``adjusted`` are BH-adjusted p-values (monotone, capped at 1), so a reader can apply a
    different threshold without re-running the procedure. NaN p-values are carried through
    as non-rejected with NaN adjustment: a cell that could not be tested is not a discovery
    and must not shrink the denominator.
    """
    return _step_up(p_values, q=q, penalty=1.0)


def benjamini_yekutieli(p_values, q: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """``(rejected, adjusted)`` under arbitrary dependence between the tests.

    Identical to Benjamini-Hochberg with every threshold divided by the harmonic number
    ``c(n) = sum_{i=1..n} 1/i``, which is what buys validity when the dependence structure
    is unknown. On the robustness grid the conditions share a null bank, so this is the
    procedure whose assumption the design actually satisfies.
    """
    n = int(np.isfinite(np.asarray(p_values, dtype=float)).sum())
    penalty = float(np.sum(1.0 / np.arange(1, n + 1))) if n else 1.0
    return _step_up(p_values, q=q, penalty=penalty)


def _step_up(p_values, *, q: float, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """The shared step-up procedure; ``penalty`` is 1 for BH and ``c(n)`` for BY."""
    p = np.asarray(p_values, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)
    adjusted = np.full(p.shape, np.nan)

    finite = np.isfinite(p)
    if not finite.any():
        return rejected, adjusted

    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    n = ranked.size

    # Step-up: the largest k with p_(k) <= k/n * q, then reject everything up to it.
    thresholds = (np.arange(1, n + 1) / (n * penalty)) * q
    below = np.where(ranked <= thresholds)[0]
    cut = below.max() if below.size else -1

    reject_sorted = np.zeros(n, dtype=bool)
    if cut >= 0:
        reject_sorted[: cut + 1] = True

    # Adjusted p-values are the running minimum of n/k * p_(k) taken from the top down,
    # which enforces the monotonicity the raw ratios do not have.
    adjusted_sorted = np.minimum.accumulate(
        (n * penalty / np.arange(n, 0, -1)) * ranked[::-1]
    )[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    out_reject = np.zeros(n, dtype=bool)
    out_adjusted = np.empty(n)
    out_reject[order] = reject_sorted
    out_adjusted[order] = adjusted_sorted

    rejected[finite] = out_reject
    adjusted[finite] = out_adjusted
    return rejected, adjusted
