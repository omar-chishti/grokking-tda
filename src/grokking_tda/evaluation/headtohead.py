"""What does topology add over the cheap baselines, on identical configuration-split folds?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grokking_tda.baselines.fourier import FOURIER_K_SWEEP

TOPOLOGY = (
    "h1_max_persistence",
    "h1_total_persistence",
    "h1_max_persistence_normalised",
    "h1_total_persistence_normalised",
    "h1_persistence_entropy",
    "h0_total_persistence",
)
FOURIER = ("fourier_concentration", "fourier_concentration_group") + tuple(
    f"fourier_concentration{g}_k{k}" for g in ("", "_group") for k in FOURIER_K_SWEEP
)
CHEAP = ("weight_norm", "lid")

# A vectorised diagram carries hundreds of columns for eighty runs, which a ridge reports as a
# worse score whatever the topology says, so the block is compressed to this many components —
# fitted inside each training fold, never on the whole table.
COMPONENTS = 10

# baselines is nested in baselines+topology, so their difference is the topological block
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "weight_norm": CHEAP[:1],
    "lid": CHEAP[1:],
    "fourier": FOURIER,
    "topology": TOPOLOGY,
    "baselines": CHEAP + FOURIER,
    "baselines+topology": CHEAP + FOURIER + TOPOLOGY,
}


def feature_columns(observables: tuple[str, ...], available: list[str]) -> list[str]:
    wanted = {f"{name}__{stat}" for name in observables for stat in ("mean", "trend")}
    return [column for column in available if column in wanted]


def _prepare(*extra) -> Pipeline:
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), *extra]
    )


def _estimator(task: str, compress: list[str] | None = None) -> tuple[Pipeline, dict]:
    if task == "classification":
        model = LogisticRegression(max_iter=2000)
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        model = Ridge()
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    if compress:
        # only the vector block is compressed; the baselines pass through at full width, so
        # they stay nested in baselines+vector and the difference is still the topological block
        compressed = _prepare(("pca", PCA(COMPONENTS, random_state=0)))
        prepare = ColumnTransformer(
            [("vector", compressed, compress)], remainder=_prepare()
        )
        pipeline = Pipeline([("prepare", prepare), ("model", model)])
    else:
        pipeline = _prepare(("model", model))
    return pipeline, grid


def _n_splits(task: str, target: np.ndarray, groups: np.ndarray, requested: int) -> int:
    """Fold count every fold can satisfy: a single-class fold makes AUC a silent NaN."""
    n_groups = len(np.unique(groups))
    limit = min(requested, n_groups)
    if task == "classification":
        per_class = [len(np.unique(groups[target == c])) for c in np.unique(target)]
        limit = min(limit, min(per_class))
    return max(2, limit)


def _tunable(task: str, target: np.ndarray, groups: np.ndarray, n_splits: int) -> bool:
    """Whether an inner search would score anything; a grid of NaN picks its first candidate."""
    if task != "classification":
        return True
    splitter = _splitter(task, n_splits)
    dummy = np.zeros((len(target), 1))
    return all(
        len(np.unique(target[train])) > 1
        for train, _ in splitter.split(dummy, target, groups)
    )


def _splitter(task: str, n_splits: int):
    if task == "classification":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    return GroupKFold(n_splits=n_splits)


def nested_scores(
    features: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    task: str,
    n_outer: int = 5,
    n_inner: int = 3,
    winsor: tuple[float, float] | None = None,
    compress: list[str] | None = None,
) -> list[float]:
    outer = _splitter(task, _n_splits(task, target, groups, n_outer))
    scores: list[float] = []
    for train_idx, test_idx in outer.split(features, target, groups):
        train_groups = groups[train_idx]
        if winsor is not None:
            # bounds from the training fold only, then applied to both sides of it, as the
            # imputer and the scaler in the same pipeline already are
            lo, hi = np.quantile(target[train_idx], winsor)
            target = target.copy()
            target[train_idx] = np.clip(target[train_idx], lo, hi)
            target[test_idx] = np.clip(target[test_idx], lo, hi)
        if len(np.unique(train_groups)) < 2:
            continue
        if task == "classification" and len(np.unique(target[train_idx])) < 2:
            continue
        pipeline, grid = _estimator(task, compress)
        inner = _n_splits(task, target[train_idx], train_groups, n_inner)
        train_x, train_y = features.iloc[train_idx], target[train_idx]
        if _tunable(task, train_y, train_groups, inner):
            model = GridSearchCV(
                pipeline,
                grid,
                cv=_splitter(task, inner),
                scoring="roc_auc" if task == "classification" else "r2",
            )
            model.fit(train_x, train_y, groups=train_groups)
        else:
            model = pipeline.set_params(**{k: v[len(v) // 2] for k, v in grid.items()})
            model.fit(train_x, train_y)
        if task == "classification":
            if len(np.unique(target[test_idx])) < 2:
                continue
            predicted = model.predict_proba(features.iloc[test_idx])[:, 1]
            scores.append(float(roc_auc_score(target[test_idx], predicted)))
        else:
            predicted = model.predict(features.iloc[test_idx])
            scores.append(float(r2_score(target[test_idx], predicted)))
    return scores


def head_to_head(
    table: pd.DataFrame,
    task: str,
    window: str,
    winsor: tuple[float, float] | None = None,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    compressed: tuple[str, ...] = (),
) -> pd.DataFrame:
    available = [c for c in table.columns if c.endswith(("__mean", "__trend"))]
    groups = table["group"].to_numpy()
    target = table["target"].to_numpy(dtype=float)
    rows = []
    for name, observables in (feature_sets or FEATURE_SETS).items():
        columns = feature_columns(observables, available)
        if not columns:
            continue
        compress = [c for c in columns if c.startswith(compressed)] if compressed else None
        scores = nested_scores(
            table[columns], target, groups, task, winsor=winsor, compress=compress or None
        )
        row = {
            "window": window,
            "task": task,
            "feature_set": name,
            "n_features": len(columns),
            "n_folds": len(scores),
            "score_mean": float(np.mean(scores)) if scores else float("nan"),
            "score_std": float(np.std(scores)) if scores else float("nan"),
        }
        if compressed:
            row["n_compressed"] = len(compress or ())
        rows.append(row)
    frame = pd.DataFrame(rows)
    # the headline: what the topological block adds
    if {"baselines", "baselines+topology"} <= set(frame["feature_set"]):
        base = frame.loc[frame.feature_set == "baselines", "score_mean"].iloc[0]
        both = frame.loc[frame.feature_set == "baselines+topology", "score_mean"].iloc[0]
        frame["increment_over_baselines"] = np.where(
            frame.feature_set == "baselines+topology", both - base, np.nan
        )
    return frame
