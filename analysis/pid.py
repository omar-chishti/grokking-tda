"""Partial information decomposition, per regime and with uncertainty (section 5.5).

The pooled decomposition in ``analysis/redundancy.py`` reports four bare point estimates
over the whole bank. Three things are wrong with that and only one is currently in the
text.

**It is pooled.** Section 4.5 establishes that the regimes differ in exactly the structure
being decomposed, so a single decomposition averages over the distinction the chapter is
about. Each regime is decomposed separately here, and the pooled figure kept for
comparison.

**The weaker source's unique atom is zero by construction.** Under minimum-mutual-
information redundancy, redundancy is ``min_i I(S_i; T)``, so whichever source carries
less can never be credited with anything of its own --- and $H_1$ carries less. That
limitation is already stated in the chapter, but the stronger fact is not: Barrett showed
that for jointly Gaussian variables with univariate sources and target, essentially every
proposed redundancy function *collapses* to MMI, so the zero cannot be escaped by choosing
differently within the Gaussian model. Escaping it requires leaving that model, so the
Williams-Beer decomposition is computed on quantile-binned variables alongside, where
redundancy is taken over target values rather than over their average and a source that
resolves a different part of the range can be credited. Both are reported. A zero under
*both* is a measurement; a zero under MMI alone is an artefact of the estimator.

**The atoms are point estimates.** Each gets a cluster bootstrap and a permutation null
here. The bootstrap resamples *configurations*, not runs, because the five seeds of a
recipe are near-duplicates --- which is also the real correction to the effective sample
size. The caveat currently in section 5.5 attributes the shortfall to autocorrelation
between checkpoints within runs; that describes a different analysis, since the unit here
is the run. The reported design effect is what the clustering actually costs.

Usage (from ``Code/``)::

    uv run python -m analysis.pid
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import CONDITION_KEYS, load_bank
from grokking_tda.evaluation import gaussian_pid, williams_beer_pid

N_BOOT = 2_000
N_PERMUTATIONS = 2_000
ATOMS = ("redundant", "unique_a", "unique_b", "synergistic", "total")

SOURCE_A = "h1_max_persistence_normalised__ratio"
SOURCE_B = "circularity"
TARGET = "log10 t_g"

ESTIMATORS = {"gaussian_mmi": gaussian_pid, "williams_beer": williams_beer_pid}

# The regimes section 4.5 separates; "pooled" keeps the whole bank for comparison.
REGIMES = {
    "reference": lambda d: (d.model == "transformer")
    & (d.modulus == 113)
    & (d.weight_decay == 0.1),
    "canonical": lambda d: (d.model == "transformer")
    & (d.modulus == 97)
    & (d.operation == "add"),
    "mlp": lambda d: d.model == "mlp",
}


def design_effect(groups: np.ndarray, values: np.ndarray) -> dict:
    """Effective sample size under clustering by configuration.

    ``n_eff = n / (1 + (m - 1) rho)`` with ``m`` the mean cluster size and ``rho`` the
    intraclass correlation, estimated from the one-way variance components. Seeds of one
    recipe are near-duplicates, so this --- not autocorrelation in training time --- is
    what the nominal count overstates.
    """
    frame = pd.DataFrame({"g": groups, "v": values}).dropna()
    n, n_groups = len(frame), frame.g.nunique()
    if n_groups < 2 or n == n_groups:
        return {"n": n, "n_configurations": n_groups, "icc": 0.0, "n_effective": float(n)}

    means = frame.groupby("g").v.transform("mean")
    within = float(((frame.v - means) ** 2).sum() / max(n - n_groups, 1))
    between = float(frame.groupby("g").v.mean().var(ddof=1))
    icc = 0.0 if between + within <= 0 else max(0.0, between / (between + within))
    m = n / n_groups
    return {
        "n": n,
        "n_configurations": n_groups,
        "icc": icc,
        "n_effective": float(n / (1.0 + (m - 1.0) * icc)),
    }


def decompose(frame: pd.DataFrame, estimator, *, seed: int = 0) -> dict:
    """Atoms, a cluster bootstrap interval on each, and a permutation null."""
    a = frame[SOURCE_A].to_numpy(float)
    b = frame[SOURCE_B].to_numpy(float)
    t = np.log10(frame["t_g"].to_numpy(float))
    groups = frame["group"].to_numpy()

    point = estimator(a, b, t)
    if not np.isfinite(point.get("total", np.nan)):
        return {"atoms": point, **design_effect(groups, a)}

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique_groups}

    draws, null = [], []
    for _ in range(N_BOOT):
        picked = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        rows = np.concatenate([index[g] for g in picked])
        draws.append(estimator(a[rows], b[rows], t[rows]))
    for _ in range(N_PERMUTATIONS):
        null.append(estimator(a, b, rng.permutation(t)))

    out = {"atoms": {}, **design_effect(groups, a)}
    for atom in ATOMS:
        boot = np.array([d.get(atom, np.nan) for d in draws], dtype=float)
        boot = boot[np.isfinite(boot)]
        shuffled = np.array([d.get(atom, np.nan) for d in null], dtype=float)
        shuffled = shuffled[np.isfinite(shuffled)]
        out["atoms"][atom] = {
            "estimate": point.get(atom, float("nan")),
            "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
            if boot.size
            else [float("nan")] * 2,
            "null_median": float(np.median(shuffled)) if shuffled.size else float("nan"),
            "null_p": float((np.sum(shuffled >= point.get(atom, np.nan)) + 1) / (shuffled.size + 1))
            if shuffled.size
            else float("nan"),
        }
    for key in ("mi_a", "mi_b"):
        if key in point:
            out[key] = point[key]
    return out


def main() -> None:
    ap = cli.parser(__doc__)
    args = ap.parse_args()

    bank, _ = load_bank(args.root)
    grokked = bank[bank.grokked & bank.t_g.notna()]
    if "dense" in grokked:
        grokked = grokked[~grokked.dense]
    grokked = grokked.dropna(subset=[SOURCE_A, SOURCE_B]).copy()
    grokked["group"] = grokked[CONDITION_KEYS].astype(str).agg("|".join, axis=1)

    subsets = {"pooled": grokked}
    for name, predicate in REGIMES.items():
        subsets[name] = grokked[predicate(grokked)]

    results = {
        "target": TARGET,
        "source_a": SOURCE_A,
        "source_b": SOURCE_B,
        "n_bootstrap": N_BOOT,
        "n_permutations": N_PERMUTATIONS,
        "regimes": {},
    }
    for name, frame in subsets.items():
        if len(frame) < 12:
            print(f"  {name:10s} skipped ({len(frame)} runs)")
            continue
        results["regimes"][name] = {
            estimator_name: decompose(frame, estimator)
            for estimator_name, estimator in ESTIMATORS.items()
        }
        block = results["regimes"][name]
        eff = block["gaussian_mmi"]
        print(
            f"\n{name} — {eff['n']} runs across {eff['n_configurations']} configurations, "
            f"ICC {eff['icc']:.2f}, effective n {eff['n_effective']:.1f}"
        )
        for estimator_name, block in results["regimes"][name].items():
            atoms = block.get("atoms", {})
            if "redundant" not in atoms or not isinstance(atoms["redundant"], dict):
                continue
            parts = " ".join(
                f"{atom[:4]} {atoms[atom]['estimate']:+.3f}"
                f"[{atoms[atom]['ci'][0]:+.3f},{atoms[atom]['ci'][1]:+.3f}]"
                for atom in ("redundant", "unique_a", "unique_b", "synergistic")
            )
            print(f"  {estimator_name:14s} {parts}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pid.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {args.out}/pid.json")


if __name__ == "__main__":
    main()
