"""Which single ingredient of the recipe decides how circular the solution is? (§4.5, §7.6)

§4.5 concludes that *the recipe as a whole* determines circularity, which is an admission
rather than a finding: the two headline regimes differ in six things at once and the
weight-decay dose sweep shows the dose alone is not it. R14 holds the modulus, the weight
decay and the budget fixed and moves one factor at a time from each of two anchors, so a
difference between a cell and its anchor is attributable to that factor and nothing else.

A cell is identified from its configuration rather than its name: the name records the design
and the configuration is what ran, and where they disagree the configuration is right.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis import cli
from analysis.bank import HEADLINE_OBSERVABLE, load_bank

TAG = "_recipe-"

# the six the two regimes differ in, read off their manifests rather than assumed
FACTORS = ("batch_size", "lr", "n_layers", "act", "d_mlp", "eps")
ANCHORS = {
    "reference": {
        "batch_size": 512, "lr": 3e-3, "n_layers": 2,
        "act": "gelu", "d_mlp": 256, "eps": 1e-6,
    },
    "canonical": {
        "batch_size": None, "lr": 1e-3, "n_layers": 1,
        "act": "relu", "d_mlp": 512, "eps": 1e-8,
    },
}


def factors(config: dict) -> dict:
    train, model = config["train"], config["model"]
    return {
        "batch_size": train.get("batch_size"),
        "lr": train["optimizer"]["lr"],
        "n_layers": model["n_layers"],
        "act": model["act"],
        "d_mlp": model["d_mlp"],
        "eps": train["optimizer"]["eps"],
    }


def classify(config: dict) -> tuple[str, str | None]:
    """``(anchor, the one factor moved away from it)``; ``None`` for an anchor itself."""
    seen = factors(config)
    for anchor, settings in ANCHORS.items():
        differing = [f for f in FACTORS if seen[f] != settings[f]]
        if not differing:
            return anchor, None
        if len(differing) == 1:
            return anchor, differing[0]
    raise ValueError(f"no anchor within one factor of {seen}")


def main() -> None:
    args = cli.parser(__doc__).parse_args()
    bank, _ = load_bank(args.root)
    sweep = bank[bank.run.str.contains(TAG)].copy()
    if sweep.empty:
        raise SystemExit(f"no runs matching {TAG!r} under {args.root} — the sweep has not landed")

    ratio = f"{HEADLINE_OBSERVABLE}__ratio"
    rows = []
    for run in sweep.itertuples():
        anchor, moved = classify(_config(args.root, run.run))
        rows.append(
            {
                "run": run.run,
                "anchor": anchor,
                "factor": moved or "—",
                "circularity": run.circularity,
                "ratio": getattr(run, ratio),
                "t_g": run.t_g,
            }
        )
    table = pd.DataFrame(rows)

    cells = table.groupby(["anchor", "factor"]).agg(
        n=("run", "size"),
        circularity=("circularity", "median"),
        ratio=("ratio", "median"),
        t_g=("t_g", "median"),
    ).reset_index()

    # what one factor is worth: the cell against its own anchor
    base = {a: cells[(cells.anchor == a) & (cells.factor == "—")] for a in ANCHORS}
    cells["d_circularity"] = [
        row.circularity - base[row.anchor].circularity.iloc[0]
        if len(base[row.anchor]) else float("nan")
        for row in cells.itertuples()
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "recipe_runs.csv", index=False)
    cells.to_csv(args.out / "recipe_cells.csv", index=False)

    ranked = cells[cells.factor != "—"].reindex(
        cells[cells.factor != "—"].d_circularity.abs().sort_values(ascending=False).index
    )
    summary = {
        "anchors": {a: {"circularity": float(b.circularity.iloc[0]), "n": int(b.n.iloc[0])}
                    for a, b in base.items() if len(b)},
        "largest_single_factor": ranked.iloc[0].factor if len(ranked) else None,
        "by_factor": {
            f: {a: float(v) for a, v in
                zip(sub.anchor, sub.d_circularity, strict=True)}
            for f, sub in ranked.groupby("factor")
        },
    }
    (args.out / "recipe.json").write_text(json.dumps(summary, indent=2, default=str))

    print(cells.to_string(index=False))
    print(f"\nlargest single-factor move: {summary['largest_single_factor']}")
    print(f"written to {args.out}/recipe_runs.csv, recipe_cells.csv and recipe.json")


def _config(root: Path, run_name: str) -> dict:
    return json.loads((root / run_name / "manifest.json").read_text())["config"]


if __name__ == "__main__":
    main()
