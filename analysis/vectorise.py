"""Is the redundancy verdict about topology, or about a scalar summary of it? (§5.4, §5.5)

Every topological feature the head-to-head scores is one number per diagram. A diagram is a
multiset, and reducing it to a scalar is a choice that the verdict then inherits: if vectorised
topology predicts no better than the scalars do, the negative belongs to the topology, and if it
does, the negative belonged to the summary.

Landscapes and images are computed here from the diagrams already cached under each run, so this
adds no homology computation and touches nothing the bank has committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import cli
from analysis.bank import window_medians
from analysis.predictive import regression_variants
from grokking_tda.analysis.aggregate import early_window_table
from grokking_tda.analysis.context import diagram_cache_digest, stored_analysis_cfg
from grokking_tda.analysis.identity import is_replicate
from grokking_tda.artifacts.reader import Run
from grokking_tda.evaluation.headtohead import (
    CHEAP,
    COMPONENTS,
    FEATURE_SETS,
    FOURIER,
    head_to_head,
)
from grokking_tda.evaluation.predictive import (
    PREREGISTERED_WINDOWS,
    before_the_event,
    early_window_feature_grid,
)
from grokking_tda.tda.summaries import connectivity_scale
from grokking_tda.tda.vectorise import diagram_extent, landscape, persistence_image

LAYERS, RESOLUTION, GRID = 5, 20, 8
HOMOLOGY_DIM = 1
PREFIXES = ("landscape_", "pimage_")
PERMUTED = "perm_"  # the same block with its rows shuffled: what the grid scores on nothing


def cached_diagrams(run: Run) -> dict[int, dict[int, np.ndarray]]:
    """Every cached diagram of this run, by step, from *its own* construction.

    A run's ``analysis/diagrams/`` holds several hundred files from several constructions, one
    digest each. Globbing without the digest would average point clouds that are not comparable.
    """
    digest = diagram_cache_digest(stored_analysis_cfg(run), int(run.config["seed"]))
    diagrams: dict[int, dict[int, np.ndarray]] = {}
    for path in sorted(run.dir.glob(f"analysis/diagrams/step_*_{digest}.npz")):
        with np.load(path) as data:
            diagrams[int(path.name.split("_")[1])] = {
                int(name[3:]): data[name] for name in data.files
            }
    return diagrams


def unit_series(run: Run) -> tuple[list[int], dict[str, list[np.ndarray | None]]]:
    """This run's H1 diagrams by step, raw and divided by their own connectivity scale."""
    cached = cached_diagrams(run)
    steps = sorted(cached)
    raw = [cached[s].get(HOMOLOGY_DIM) for s in steps]
    scales = [connectivity_scale(cached[s]) for s in steps]
    scaled = [
        d / s if d is not None and d.size and s > 0 else None
        for d, s in zip(raw, scales, strict=True)
    ]
    return steps, {"raw": raw, "normalised": scaled}


def vector_frame(run: Run, extents: dict[str, tuple[float, float]]) -> tuple[pd.DataFrame, int]:
    """One row per checkpoint, one column per landscape sample and image cell.

    The sampling grid is shared by every run in the bank, not fitted to each. A per-run grid
    makes the checkpoints of one run comparable and its columns mean something different in the
    next one, which is fatal for a comparison whose unit is the run.
    """
    steps, series = unit_series(run)
    if not steps:
        return pd.DataFrame(), 0
    rows = []
    for index, step in enumerate(steps):
        row = {"step": step}
        for units, diagrams in series.items():
            diagram, extent = diagrams[index], extents[units]
            for i, layer in enumerate(
                landscape(diagram, layers=LAYERS, resolution=RESOLUTION, extent=extent)
            ):
                row |= {f"landscape_{units}_l{i}_p{j:02d}": v for j, v in enumerate(layer)}
            image = persistence_image(diagram, grid=GRID, extent=extent)
            row |= {
                f"pimage_{units}_r{i}_c{j}": image[i, j] for i in range(GRID) for j in range(GRID)
            }
        rows.append(row)
    return pd.DataFrame(rows), len(steps)


def terminal_row(frame: pd.DataFrame, tg: float | None) -> dict[str, float]:
    """The plateau vector and its shift from the memorisation baseline, under §4.2's window rule.

    ``h1_max_persistence_normalised__ratio`` is a plateau divided by a baseline. On a vector the
    additive form is the stable one — a landscape sample can sit at zero on the baseline, and a
    ratio there is a division by nothing.
    """
    row: dict[str, float] = {}
    for column in frame.columns:
        if not column.startswith(PREFIXES):
            continue
        baseline, plateau = window_medians(frame, column, tg)
        row[f"{column}__plateau"] = plateau
        row[f"{column}__shift"] = plateau - baseline
    return row


def bank_runs(root: Path) -> list[Run]:
    runs = []
    for manifest in sorted(Path(root).rglob("manifest.json")):
        run_dir = manifest.parent
        if not (run_dir / "analysis" / "summary.json").exists():
            continue
        try:
            run = Run(run_dir)
        except Exception:
            continue
        if not is_replicate(run.run_name, run.config):
            runs.append(run)
    return runs


def bank_extents(runs: list[Run]) -> dict[str, tuple[float, float]]:
    """One sampling grid per unit system, spanning every diagram in the bank."""
    collected: dict[str, list[np.ndarray | None]] = {"raw": [], "normalised": []}
    for run in runs:
        _, series = unit_series(run)
        for units, diagrams in series.items():
            collected[units].extend(diagrams)
    return {units: diagram_extent(d) for units, d in collected.items()}


def variance_kept(frame: pd.DataFrame, vectors: tuple[str, ...]) -> float:
    """What the compression costs: the share of the block's variance its components keep.

    Without this the negative below is unreadable — a vector block that predicts no better than
    six scalars could be a fact about topology or an artefact of squeezing it into ten columns.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    columns = [f"{v}__{stat}" for v in vectors for stat in ("mean", "trend")]
    columns = [c for c in columns if c in frame]
    block = StandardScaler().fit_transform(np.nan_to_num(frame[columns].to_numpy(float)))
    fitted = PCA(min(COMPONENTS, *block.shape), random_state=0).fit(block)
    return float(fitted.explained_variance_ratio_.sum())


def permuted_block(frame: pd.DataFrame, vectors: tuple[str, ...], seed: int) -> pd.DataFrame:
    """The vector block with its rows shuffled, so it carries no information about the run.

    A block of six hundred columns compressed to ten components can score above chance on
    structure the compression itself imposes. This is the control that separates that from
    topology: the same width, the same pipeline, the same folds, and nothing to know.
    """
    columns = [f"{v}__{stat}" for v in vectors for stat in ("mean", "trend")]
    columns = [c for c in columns if c in frame]
    order = np.random.default_rng(seed).permutation(len(frame))
    shuffled = frame[columns].to_numpy()[order]
    return pd.DataFrame(shuffled, index=frame.index, columns=[f"{PERMUTED}{c}" for c in columns])


def scored(
    table: pd.DataFrame, window: str, vectors: tuple[str, ...], *, seed: int = 0
) -> pd.DataFrame:
    """The §5.4 grid with the vector block beside the scalar one, on the same folds.

    Only the guarded grid is scored. On the leaked one a quarter of the runs are read after the
    step they are asked to predict, and a feature set given more columns would inherit more of
    that inflation and read as a success.
    """
    guarded = before_the_event(table)
    guarded = pd.concat([guarded, permuted_block(guarded, vectors, seed)], axis=1)
    sets = FEATURE_SETS | {
        "topology_vector": vectors,
        "baselines+topology_vector": CHEAP + FOURIER + vectors,
        "topology_vector_permuted": tuple(f"{PERMUTED}{v}" for v in vectors),
    }
    compressed = PREFIXES + (PERMUTED,)
    classification = guarded.assign(target=guarded["grokking_step"].notna().astype(float))
    frames = [
        head_to_head(
            classification,
            task="classification",
            window=window,
            feature_sets=sets,
            compressed=compressed,
        ).assign(variant="full"),
        regression_variants(guarded, window, feature_sets=sets, compressed=compressed),
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = cli.parser(__doc__)
    args = ap.parse_args()

    runs = bank_runs(args.root)
    if not runs:
        raise SystemExit(f"no analysed runs under {args.root}")
    extents = bank_extents(runs)
    grids = "  ".join(f"{u} [{lo:.3f}, {hi:.3f}]" for u, (lo, hi) in extents.items())
    print(f"sampling grid, shared by every run: {grids}")

    rows: list[dict] = []
    terminal: list[dict] = []
    for run in runs:
        frame, n_checkpoints = vector_frame(run, extents)
        if frame.empty:
            continue
        summary = json.loads((run.dir / "analysis" / "summary.json").read_text())
        grid = early_window_feature_grid(
            frame, PREREGISTERED_WINDOWS, summary.get("train_convergence_step")
        )
        for window, features in grid.items():
            if features:
                rows.append({"run": run.run_name, "window": window} | features)
        terminal.append({"run": run.run_name} | terminal_row(frame, summary.get("grokking_step")))
        print(f"  {run.run_name}  {n_checkpoints} checkpoints", flush=True)

    if not rows:
        raise SystemExit(f"no cached diagrams under {args.root}")

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "vector_features.csv", index=False)
    pd.DataFrame(terminal).to_csv(args.out / "vector_terminal.csv", index=False)

    vectors = tuple(sorted({c.rsplit("__", 1)[0] for c in table.columns if c.startswith(PREFIXES)}))
    scores, kept = [], {}
    for window, features in table.groupby("window"):
        scalars = early_window_table(args.root, str(window))
        if scalars.empty:
            continue
        joined = scalars.merge(features.drop(columns="window"), on="run", how="inner")
        kept[str(window)] = variance_kept(joined, vectors)
        grid = scored(joined, str(window), vectors)
        if not grid.empty:
            scores.append(grid)
    head_to_head_table = pd.concat(scores, ignore_index=True)
    head_to_head_table.to_csv(args.out / "vector_head_to_head.csv", index=False)

    for task in ("classification", "regression"):
        sub = head_to_head_table[head_to_head_table.task == task]
        if sub.empty:
            continue
        print(f"\n{task}, {'AUC' if task == 'classification' else 'R^2'}, guarded grid:")
        print(
            sub.pivot_table(index=["window", "variant"], columns="feature_set", values="score_mean")
            .round(3)
            .to_string()
        )
    (args.out / "vectorised.json").write_text(
        json.dumps(
            {
                "layers": LAYERS,
                "resolution": RESOLUTION,
                "image_grid": GRID,
                "homology_dim": HOMOLOGY_DIM,
                "extents": {u: list(e) for u, e in extents.items()},
                "components": COMPONENTS,
                "variance_kept": kept,
                "n_runs": int(table.run.nunique()),
                "n_features": int(table.shape[1] - 2),
            },
            indent=2,
        )
    )
    print(
        f"\n{table.run.nunique()} runs, {table.shape[1] - 2} features per window\n"
        f"written to {args.out}/vector_features.csv, vector_terminal.csv, "
        "vector_head_to_head.csv and vectorised.json"
    )


if __name__ == "__main__":
    main()
