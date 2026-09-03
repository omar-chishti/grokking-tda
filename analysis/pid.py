"""Partial information decomposition, per regime and with uncertainty (§5.5)."""

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

SOURCE_B = "circularity"
TARGET = "log10 t_g"

# The reported pairing sets a ratio against a terminal level, which are not the same kind of
# quantity, so it is also run with both sources terminal
SOURCE_A = "h1_max_persistence_normalised__ratio"
SOURCES_A = {
    "ratio": SOURCE_A,
    "terminal": "h1_max_persistence_normalised__plateau",
}

# The vectorised source of §5.4: the diagram as a landscape and an image rather than as one
# number. Three components, because the Gaussian estimator spends a degree of freedom on each
# and the smallest regime carries twenty-odd runs.
VECTOR_COMPONENTS = 3
VECTOR_SOURCE = "vector"

ESTIMATORS = {"gaussian_mmi": gaussian_pid, "williams_beer": williams_beer_pid}
# Resampling creates ties, which a binned estimator reads as dependence, so only the
# permutation null is reported for the Williams-Beer atoms
BOOTSTRAPPED = {"gaussian_mmi"}

# The regimes §4.5 separates; "pooled" keeps the whole bank for comparison.
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
    """Effective sample size under clustering by configuration; seeds are near-duplicates."""
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


def permute_within_clusters(
    target: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle targets between configurations, so only the association is destroyed."""
    unique = np.unique(groups)
    pools = {g: target[groups == g] for g in unique}
    out = np.empty_like(target)
    for source, destination in zip(unique, rng.permutation(unique), strict=True):
        rows = np.flatnonzero(groups == destination)
        pool = pools[source]
        out[rows] = pool[rng.integers(pool.size, size=rows.size)]
    return out


def decompose(
    frame: pd.DataFrame, estimator, *, source_a: str | list[str] = SOURCE_A,
    bootstrap: bool = True, seed: int = 0,
) -> dict:
    a = frame[source_a].to_numpy(float)
    b = frame[SOURCE_B].to_numpy(float)
    t = np.log10(frame["t_g"].to_numpy(float))
    groups = frame["group"].to_numpy()

    leading = a if a.ndim == 1 else a[:, 0]

    point = estimator(a, b, t)
    if not np.isfinite(point.get("total", np.nan)):
        return {"atoms": point, **design_effect(groups, leading)}

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique_groups}

    draws, null = [], []
    if bootstrap:
        for _ in range(N_BOOT):
            picked = rng.choice(unique_groups, size=unique_groups.size, replace=True)
            rows = np.concatenate([index[g] for g in picked])
            draws.append(estimator(a[rows], b[rows], t[rows]))
    for _ in range(N_PERMUTATIONS):
        null.append(estimator(a, b, permute_within_clusters(t, groups, rng)))

    out = {"atoms": {}, "bootstrapped": bootstrap, **design_effect(groups, leading)}
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


def vector_source(frame: pd.DataFrame, vectors: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The plateau-shift vector reduced to its leading components, joined onto the bank rows.

    The components are fitted on the regime being decomposed. That is not a leak — no target is
    involved and nothing is scored out of sample here — but it does mean the axes are the axes
    of this regime's own variation, which is what a decomposition of this regime should use.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    columns = [c for c in vectors.columns if c.endswith("__shift")]
    joined = frame.merge(vectors[["run", *columns]], on="run", how="inner")
    block = joined[columns].to_numpy(float)
    block = np.nan_to_num(block, nan=float(np.nanmedian(block)))
    components = PCA(min(VECTOR_COMPONENTS, *block.shape), random_state=0).fit_transform(
        StandardScaler().fit_transform(block)
    )
    names = [f"pc{i}" for i in range(components.shape[1])]
    reduced = pd.DataFrame(components, columns=names, index=joined.index)
    return pd.concat([joined, reduced], axis=1), names


def main() -> None:
    ap = cli.parser(__doc__)
    args = ap.parse_args()

    bank, _ = load_bank(args.root)
    grokked = bank[bank.grokked & bank.t_g.notna()]
    if "replicate" in grokked:
        grokked = grokked[~grokked.replicate]
    grokked = grokked.dropna(subset=[*SOURCES_A.values(), SOURCE_B]).copy()
    grokked["group"] = grokked[CONDITION_KEYS].astype(str).agg("|".join, axis=1)

    subsets = {"pooled": grokked}
    for name, predicate in REGIMES.items():
        subsets[name] = grokked[predicate(grokked)]

    vector_path = args.out / "vector_terminal.csv"
    vectors = pd.read_csv(vector_path) if vector_path.exists() else None

    results = {
        "target": TARGET,
        "source_a": SOURCE_A,
        "sources_a": SOURCES_A,
        "source_b": SOURCE_B,
        "vector_components": VECTOR_COMPONENTS,
        "n_bootstrap": N_BOOT,
        "n_permutations": N_PERMUTATIONS,
        "regimes": {},
    }
    for name, frame in subsets.items():
        if len(frame) < 12:
            print(f"  {name:10s} skipped ({len(frame)} runs)")
            continue
        results["regimes"][name] = {
            f"{estimator_name}__{pairing}": decompose(
                frame, estimator, source_a=column,
                bootstrap=estimator_name in BOOTSTRAPPED,
            )
            for estimator_name, estimator in ESTIMATORS.items()
            for pairing, column in SOURCES_A.items()
        }
        if vectors is not None:
            # only the Gaussian estimator: Williams-Beer bins each variable, and a
            # three-dimensional source would ask for 4^3 cells from twenty-odd runs
            reduced, names = vector_source(frame, vectors)
            results["regimes"][name][f"gaussian_mmi__{VECTOR_SOURCE}"] = decompose(
                reduced, gaussian_pid, source_a=names
            )
        block = results["regimes"][name]["gaussian_mmi__ratio"]
        print(
            f"\n{name} — {block['n']} runs across {block['n_configurations']} configurations, "
            f"ICC {block['icc']:.2f}, effective n {block['n_effective']:.1f}"
        )
        for key, block in results["regimes"][name].items():
            atoms = block.get("atoms", {})
            if "redundant" not in atoms or not isinstance(atoms["redundant"], dict):
                continue
            parts = " ".join(
                f"{atom[:4]} {atoms[atom]['estimate']:+.3f}"
                + (
                    f"[{atoms[atom]['ci'][0]:+.3f},{atoms[atom]['ci'][1]:+.3f}]"
                    if block.get("bootstrapped")
                    else f" p={atoms[atom]['null_p']:.3f}"
                )
                for atom in ("redundant", "unique_a", "unique_b", "synergistic")
            )
            print(f"  {key:26s} {parts}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pid.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {args.out}/pid.json")


if __name__ == "__main__":
    main()
