"""Recompute every number the thesis quotes and print it as a ledger; write the tidy tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis import cli
from analysis.bank import (
    CIRCULARITY_K,
    RATIO_OBSERVABLES,
    circularity_column,
    condition_label,
    condition_table,
    load_bank,
    null_band,
    null_runs,
    verdicts,
)


def minimum_detectable_effect(
    bank: pd.DataFrame,
    *,
    n_seeds: int = 5,
    power: float = 0.8,
    n_draws: int = 20_000,
    seed: int = 0,
) -> dict:
    """The smallest ratio ``n_seeds`` could have cleared (§4.4).

    Exact rather than searched: the bootstrap bound is homogeneous, so a condition clears a
    null draw of lower bound ``L`` precisely when its multiplier exceeds ``ceiling / L``.
    """
    nulls = null_runs(bank)
    rng = np.random.default_rng(seed)
    out = {"n_seeds": n_seeds, "power": power, "n_draws": n_draws,
           "criterion": "bootstrap lower bound above the null maximum"}
    for column in RATIO_OBSERVABLES:
        name = f"{column}__ratio"
        if name not in bank:
            continue
        values = np.asarray(nulls[name], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size < 4:
            continue
        ceiling = float(values.max())
        draws = rng.choice(values, size=(n_draws, n_seeds))
        picks = rng.integers(0, n_seeds, size=(n_draws, 1_000, n_seeds))
        medians = np.median(np.take_along_axis(draws[:, None, :], picks, axis=2), axis=2)
        lower = np.quantile(medians, 0.025, axis=1)
        needed = ceiling / np.where(lower > 0, lower, np.nan)
        out[column] = {
            "n_null_runs": int(values.size),
            "null_min": float(values.min()),
            "null_max": ceiling,
            "minimum_detectable_ratio": float(np.nanquantile(needed, power)),
        }
    return out


SEED_COUNTS = tuple(range(2, 19))  # up to the seventeen seeds of the non-cyclic condition


def mde_by_seed_count(bank: pd.DataFrame, counts=SEED_COUNTS) -> pd.DataFrame:
    """The detection floor as a function of replication, for §7.4."""
    rows = []
    for n in counts:
        effect = minimum_detectable_effect(bank, n_seeds=n)
        for column in RATIO_OBSERVABLES:
            if column in effect:
                cell = effect[column]
                rows.append(
                    {
                        "n_seeds": n,
                        "observable": column,
                        # the band is repeated on every row: the floor is set by its width
                        # and how many runs define it, not by a condition's seed count
                        "n_null_runs": cell["n_null_runs"],
                        "null_min": cell["null_min"],
                        "null_max": cell["null_max"],
                        "power": effect["power"],
                        "minimum_detectable_ratio": cell["minimum_detectable_ratio"],
                    }
                )
    return pd.DataFrame(rows)


def intervention_timing(bank: pd.DataFrame) -> dict:
    """Does the signature still lag when an intervention moves the transition? (§5.7)"""
    arms = bank[(bank.weight_decay == 0) & (~bank.replicate) & (~bank.diverged)]
    out = {}
    for key, sub in arms.groupby(["model", "loss", "optimizer", "lr"], dropna=False):
        timed = sub[sub.t_g.notna() & sub.t_top.notna()]
        if timed.empty:
            continue
        lags = timed.lead_lag_steps.to_numpy(float)
        out["_".join(str(k) for k in key)] = {
            "n_seeds": int(len(timed)),
            "n_lagging": int((lags < 0).sum()),
            "t_g": sorted(int(v) for v in timed.t_g),
            "lag_steps": sorted(int(v) for v in lags),
            "lag_over_t_g": [float(v) for v in sorted(-lags / timed.t_g.to_numpy(float))],
            "ratio_median": float(timed["h1_max_persistence_normalised__ratio"].median()),
            "circularity_median": float(timed.circularity.median()),
        }
    return out


def dose_response(bank: pd.DataFrame) -> dict:
    """Does the topological transition track ``t_g`` as weight decay drags it? (§5.7.1)"""
    dose = bank[
        (bank.model == "transformer")
        & (bank.operation == "add")
        & (bank.modulus == 97)
        & (bank.train_fraction == 0.3)
        & (bank.loss == "softmax_ce")
        & (bank.optimizer == "adamw")
        & (~bank.label_permutation)
        & (~bank.replicate)
        & (bank.weight_decay > 0)
    ]
    paired = dose[dose.t_g.notna() & dose.t_top.notna()]
    if len(paired) < 6:
        return {"n": int(len(paired))}
    x = np.log10(paired.t_g.to_numpy(float))
    y = np.log10(paired.t_top.to_numpy(float).clip(min=1.0))
    fit = stats.linregress(x, y)
    return {
        "n": int(len(paired)),
        "doses": sorted(float(v) for v in paired.weight_decay.unique()),
        "slope": float(fit.slope),
        "slope_stderr": float(fit.stderr),
        "intercept": float(fit.intercept),
        "r2": float(fit.rvalue**2),
        "p": float(fit.pvalue),
        "spearman_rho": float(stats.spearmanr(paired.t_g, paired.t_top).statistic),
        "note": "slope 1 through the origin would mean the two transitions move together",
    }


def circularity_association(bank: pd.DataFrame) -> dict:
    """Does the signature track circularity? (§4.5) S_n is excluded: no basis measures it there."""
    m = bank[bank.grokked & (bank.operation != "compose") & (~bank.replicate)].copy()
    out = {"n_runs": int(len(m))}
    for target in (
        "h1_max_persistence_normalised__ratio",
        "h1_total_persistence_normalised__ratio",
        "h1_max_persistence__ratio",
    ):
        v = m[["circularity", target]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p_rho = stats.spearmanr(v["circularity"], v[target])
        logged = np.log(v[target].clip(lower=1e-6))
        r, p_r = stats.pearsonr(v["circularity"], logged)
        out[target] = {
            "n": int(len(v)),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_log_r": float(r),
            "pearson_log_p": float(p_r),
        }
    return out


def circularity_k_sensitivity(root: Path, bank: pd.DataFrame) -> dict:
    m = bank[bank.grokked & (bank.operation != "compose") & (~bank.replicate)]
    out = {}
    for k in (1, 2, 3, 5, 10, 20):
        values = []
        for row in m.itertuples():
            obs = pd.read_csv(root / row.run / "analysis" / "observables.csv")
            column = circularity_column(row.operation, obs.columns, k)
            values.append(float(obs[column].iloc[-1]) if column else np.nan)
        frame = m.assign(circ=values)[
            ["circ", "h1_max_persistence_normalised__ratio"]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p = stats.spearmanr(frame.circ, frame.iloc[:, 1])
        r, _ = stats.pearsonr(frame.circ, np.log(frame.iloc[:, 1].clip(lower=1e-6)))
        out[f"k{k}"] = {"n": int(len(frame)), "spearman_rho": float(rho),
                        "spearman_p": float(p), "pearson_log_r": float(r)}
    return out


def instability(root: Path) -> dict:
    """Runs that reach train accuracy 0.99 and later collapse below 0.5 (§4.1)."""
    converged = collapsed = total = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        path = d / "metrics.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.open() if line.strip()]
        frame = pd.DataFrame(rows)
        if "train_acc" not in frame:
            continue
        total += 1
        frame = frame.sort_values("step")
        hit = frame[frame.train_acc >= 0.99]
        if hit.empty:
            continue
        converged += 1
        after = frame[frame.step > hit.step.iloc[0]]
        collapsed += bool((after.train_acc < 0.5).any())
    return {"runs_with_metrics": total, "reached_099": converged, "collapsed_after": collapsed}


def commutativity_prediction(n_symbols: int = 5, train_fraction: float = 0.6) -> dict:
    """Fraction of S_n pairs that commute, and the plateau it predicts (§3.5)."""
    from itertools import permutations

    elements = list(permutations(range(n_symbols)))

    def compose(p, q):
        return tuple(p[q[k]] for k in range(n_symbols))

    commuting = sum(1 for p in elements for q in elements if compose(p, q) == compose(q, p))
    order = len(elements)
    # a pair (a, a) is its own transpose, so it is held out with itself and cannot leak
    leakable = commuting - order
    fraction = leakable / (order * order)
    return {
        "group": f"S_{n_symbols}",
        "order": order,
        "ordered_pairs": order * order,
        "commuting_pairs": commuting,
        "leakable_pairs": leakable,
        "leakable_fraction": fraction,
        "predicted_plateau": train_fraction * fraction,
        "train_fraction": train_fraction,
    }


def measured_plateau(root: Path, runs: list[str], lo: float = 0.3, hi: float = 0.8) -> dict:
    values = []
    for name in runs:
        obs_path = root / name / "analysis" / "observables.csv"
        summ_path = root / name / "analysis" / "summary.json"
        if not (obs_path.exists() and summ_path.exists()):
            continue
        tg = json.loads(summ_path.read_text()).get("grokking_step")
        if not tg:
            continue
        obs = pd.read_csv(obs_path)
        window = obs[(obs.step > lo * tg) & (obs.step < hi * tg)]
        value = float(window.test_acc.median()) if len(window) else float("nan")
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "runs": runs,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bank, fourier_k = load_bank(root)
    conditions = verdicts(bank, condition_table(bank))
    bank.to_csv(out / "bank.csv", index=False)
    conditions.to_csv(out / "conditions.csv", index=False)
    mde_by_seed_count(bank).to_csv(out / "mde_by_seed_count.csv", index=False)
    bank[bank.grokked & (bank.operation != "compose") & (~bank.replicate)][
        ["run", "model", "operation", "modulus", "weight_decay", "circularity"]
        + [c for c in bank.columns if c.endswith("__ratio")]
    ].to_csv(out / "circularity.csv", index=False)

    bands = {f"{c}__ratio": null_band(bank, f"{c}__ratio") for c in RATIO_OBSERVABLES
             if f"{c}__ratio" in bank}
    # the five-seed set, not the two-seed fraction sweep that located it
    s5 = bank[
        (bank.operation == "compose") & (bank.modulus == 120) & (bank.train_fraction == 0.6)
    ]
    s5_runs = sorted(s5.run)
    main = bank[~bank.replicate]
    canonical = bank[
        (bank.operation == "add")
        & (bank.modulus == 97)
        & (bank.train_fraction == 0.3)
        & (bank.weight_decay == 1.0)
        & (bank.model == "transformer")
        & (~bank.label_permutation)
        & (bank.loss == "softmax_ce")
        & (bank.optimizer == "adamw")
    ]
    add_runs = sorted(canonical.run)
    decayed = main[main.weight_decay > 0]
    claims = {
        "generated_from": os.path.relpath(root.resolve(), Path(__file__).resolve().parents[1]),
        "n_runs_loaded": int(len(bank)),
        "n_main_programme": int((~bank.replicate).sum()),
        "n_dense": int(bank.replicate.sum()),
        "n_grokked": int(bank.grokked.sum()),
        "plateau_relaxed": {
            "n_runs": int(bank.plateau_relaxed.sum()),
            "n_main_programme": int(main.plateau_relaxed.sum()),
            "runs": sorted(bank.loc[bank.plateau_relaxed, "run"]),
        },
        "fourier_k_selected": fourier_k,
        "scale_collapse": {
            # runs without weight decay barely contract
            "min_with_decay": float(decayed.scale_collapse.min()),
            "max_with_decay": float(decayed.scale_collapse.max()),
            "min_all": float(bank.scale_collapse.min()),
            "max_all": float(bank.scale_collapse.max()),
            "initial_median": float(bank.scale_initial.median()),
        },
        "null_bands": bands,
        "minimum_detectable_effect": minimum_detectable_effect(bank),
        "intervention_timing": intervention_timing(bank),
        "dose_response": dose_response(bank),
        "verdicts": {
            column: conditions[f"{column}__verdict"].value_counts().to_dict()
            for column in RATIO_OBSERVABLES
            if f"{column}__verdict" in conditions
        },
        "circularity_k": CIRCULARITY_K,
        "circularity_association": circularity_association(bank),
        "circularity_k_sensitivity": circularity_k_sensitivity(root, bank),
        "instability": instability(root),
        "commutativity_s5": commutativity_prediction(),
        "s5_measured_plateau": measured_plateau(root, s5_runs),
        "modular_addition_measured_plateau": measured_plateau(root, add_runs),
    }
    (out / "claims.json").write_text(json.dumps(claims, indent=2))
    return claims


def _fmt_ci(row, column: str) -> str:
    lo, med, hi = row.get(f"{column}__lo"), row.get(f"{column}__med"), row.get(f"{column}__hi")
    if med is None or not np.isfinite(med):
        return "—"
    return f"{med:.2f} [{lo:.2f}, {hi:.2f}]"


def report(claims: dict, conditions: pd.DataFrame) -> None:
    print(
        f"\nrun bank: {claims['n_runs_loaded']} analysed "
        f"({claims['n_main_programme']} main programme + {claims['n_dense']} dense re-runs), "
        f"{claims['n_grokked']} grokked"
    )
    pr = claims["plateau_relaxed"]
    print(
        f"plateau relaxed to t_g (too few snapshots past 1.2 t_g): {pr['n_runs']} runs, "
        f"{pr['n_main_programme']} in the main programme"
    )
    print(f"Fourier k selected on the time series: {claims['fourier_k_selected']}")
    sc = claims["scale_collapse"]
    print(
        f"connectivity scale: starts at {sc['initial_median']:.1f}; with weight decay it "
        f"collapses {sc['min_with_decay']:.0f}x to {sc['max_with_decay']:.0f}x "
        f"({sc['min_all']:.1f}x-{sc['max_all']:.0f}x over the whole bank)"
    )
    for name, nb in claims["null_bands"].items():
        print(
            f"null band, {name:46s} n={nb['n_runs']:3d}  "
            f"observed [{nb['observed_min']:.2f}, {nb['observed_max']:.2f}]"
        )
    print("conditions against those bands:", claims["verdicts"])
    print("\nintervention timing (§5.7): negative lag means topology follows")
    for arm, v in claims["intervention_timing"].items():
        print(
            f"  {arm:44s} {v['n_lagging']}/{v['n_seeds']} lag  "
            f"t_g {v['t_g']}  lag {v['lag_steps']}  "
            f"ratio {v['ratio_median']:.2f}  circ {v['circularity_median']:.3f}"
        )
    dr = claims["dose_response"]
    if "slope" in dr:
        print(
            f"\ndose-response (§5.7.1): log t_top on log t_g over {dr['n']} runs at "
            f"weight decays {dr['doses']} — slope {dr['slope']:+.2f} "
            f"(se {dr['slope_stderr']:.2f}), R^2 {dr['r2']:.2f}, p {dr['p']:.3g}, "
            f"Spearman {dr['spearman_rho']:+.2f}"
        )

    mde = claims["minimum_detectable_effect"]
    print(
        f"\nsmallest ratio a {mde['n_seeds']}-seed condition could have cleared "
        f"(80% power, {mde['criterion']}):"
    )
    for column in RATIO_OBSERVABLES:
        if column in mde:
            print(f"  {column:46s} {mde[column]['minimum_detectable_ratio']:.2f}x")
    ca = claims["circularity_association"]
    print(f"\ncircularity association, {ca['n_runs']} grokking runs with a defined measure:")
    for key, value in ca.items():
        if key == "n_runs":
            continue
        print(
            f"  {key:44s} n={value['n']:3d}  rho={value['spearman_rho']:+.3f} "
            f"(p={value['spearman_p']:.2g})  pearson(log)={value['pearson_log_r']:+.3f}"
        )
    sensitivity = claims["circularity_k_sensitivity"]
    print(
        "  sensitivity to k:",
        " ".join(f"k{k[1:]}={v['spearman_rho']:.3f}" for k, v in sensitivity.items()),
    )
    ins = claims["instability"]
    print(
        f"\ninstability: {ins['collapsed_after']} of {ins['reached_099']} converged runs "
        f"later collapse below 0.5 train accuracy"
    )
    cs, s5 = claims["commutativity_s5"], claims["s5_measured_plateau"]
    print(
        f"\nS_5 commutativity: {cs['commuting_pairs']}/{cs['ordered_pairs']} pairs commute, "
        f"{cs['leakable_pairs']} off the diagonal ({cs['leakable_fraction']:.4f}); predicted "
        f"plateau at f={cs['train_fraction']} is {cs['predicted_plateau']:.4f}"
    )
    if s5.get("n"):
        print(
            f"                   measured over {s5['n']} seeds: {s5['median']:.4f} "
            f"({s5['min']:.4f}-{s5['max']:.4f})"
        )
    add = claims["modular_addition_measured_plateau"]
    if add.get("n"):
        print(
            f"modular addition plateau (f=0.3): {add['median']:.4f} "
            f"({add['min']:.4f}-{add['max']:.4f})"
        )

    print("\nconditions, by normalised H1 max ratio:")
    table = conditions.sort_values("h1_max_persistence_normalised__med", ascending=False)
    columns = ("condition", "grok", "t_g", "raw", "norm max", "norm tot", "circ", "scale")
    widths = (52, 6, 9, 18, 18, 18, 6, 7)
    print(" ".join(f"{c:{'<' if i == 0 else '>'}{w}s}"
                   for i, (c, w) in enumerate(zip(columns, widths, strict=True))))
    for _, row in table.iterrows():
        label = condition_label(row)
        tg = "—" if not np.isfinite(row.t_g_median) else f"{row.t_g_median:,.0f}"
        print(
            f"{label[:52]:52s} {int(row.n_grokked)}/{int(row.n_runs):<4d} {tg:>9s} "
            f"{_fmt_ci(row, 'h1_max_persistence'):>18s} "
            f"{_fmt_ci(row, 'h1_max_persistence_normalised'):>18s} "
            f"{_fmt_ci(row, 'h1_total_persistence_normalised'):>18s} "
            f"{row.circularity:6.3f} {row.scale_collapse:6.0f}x"
        )


def main() -> None:
    parser = cli.parser(__doc__)
    args = parser.parse_args()
    claims = build(args.root, args.out)
    report(claims, pd.read_csv(args.out / "conditions.csv"))
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
