"""Does a topological observable predict grokking beyond what cheaper ones already do?

The redundancy question is not a horse race. Topology beating Fourier on its own would
say little, since both may be reading the same circle; what matters is whether adding
topology to the cheap baselines *improves* the prediction. Every feature set is therefore
scored on identical folds, and the reported quantity is the increment from adding the
topological block to the baseline block.

Two rules keep the comparison honest. Seeds of one configuration are near-duplicates, so
folds are split by *configuration*, never by run — a random split would put a run's own
siblings in the training set and inflate every score. And features come only from
pre-registered early windows (``evaluation/predictive.py``), so the window length cannot
encode the label being predicted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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

# Nested by design: each set adds one block to the one before it, so the difference in
# score is attributable to that block alone.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "weight_norm": CHEAP[:1],
    "lid": CHEAP[1:],
    "fourier": FOURIER,
    "topology": TOPOLOGY,
    "baselines": CHEAP + FOURIER,
    "baselines+topology": CHEAP + FOURIER + TOPOLOGY,
}


def feature_columns(observables: tuple[str, ...], available: list[str]) -> list[str]:
    """The ``__mean``/``__trend`` columns belonging to a set, in a stable order."""
    wanted = {f"{name}__{stat}" for name in observables for stat in ("mean", "trend")}
    return [column for column in available if column in wanted]


def _estimator(task: str) -> tuple[Pipeline, dict]:
    """A regularised linear model; the grid is searched in the inner fold only."""
    if task == "classification":
        model = LogisticRegression(max_iter=2000)
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        model = Ridge()
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    return pipeline, grid


def _n_splits(task: str, target: np.ndarray, groups: np.ndarray, requested: int) -> int:
    """Fold count that every fold can actually satisfy.

    A fold holding one class makes ROC AUC undefined, and sklearn answers that with a
    silent NaN that then decides the hyper-parameter. Since folds are split by
    configuration, the binding constraint is how many *configurations* the rarer class
    has, not how many runs.
    """
    n_groups = len(np.unique(groups))
    limit = min(requested, n_groups)
    if task == "classification":
        per_class = [len(np.unique(groups[target == c])) for c in np.unique(target)]
        limit = min(limit, min(per_class))
    return max(2, limit)


def _tunable(task: str, target: np.ndarray, groups: np.ndarray, n_splits: int) -> bool:
    """Whether an inner search on this split would score anything.

    ``_n_splits`` bounds the fold count by how many configurations the rarer class has,
    but two configurations spread over two folds can still land in the same one, and the
    training side then holds a single class. sklearn scores that fit ``NaN``, and a grid
    of ``NaN`` picks its first candidate — a hyper-parameter chosen by ordering rather
    than by data. When no fold count avoids it, tune nothing and say so.
    """
    if task != "classification":
        return True
    splitter = _splitter(task, n_splits)
    dummy = np.zeros((len(target), 1))
    return all(
        len(np.unique(target[train])) > 1
        for train, _ in splitter.split(dummy, target, groups)
    )


def _splitter(task: str, n_splits: int):
    """Stratify classification folds by class as well as by group.

    Plain grouped folds regularly produce a fold with a single class here — many
    configurations grok and many do not, and the split is by configuration.
    """
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
) -> list[float]:
    """Outer-fold scores with the hyper-parameter chosen inside each training fold."""
    outer = _splitter(task, _n_splits(task, target, groups, n_outer))
    scores: list[float] = []
    for train_idx, test_idx in outer.split(features, target, groups):
        train_groups = groups[train_idx]
        if len(np.unique(train_groups)) < 2:
            continue
        if task == "classification" and len(np.unique(target[train_idx])) < 2:
            continue
        pipeline, grid = _estimator(task)
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


def head_to_head(table: pd.DataFrame, task: str, window: str) -> pd.DataFrame:
    """Score every feature set on identical folds for one task and one early window.

    ``table`` carries one row per run: a ``group`` column (the configuration, so seeds
    stay together), a ``target`` column, and the early-window feature columns.
    """
    available = [c for c in table.columns if c.endswith(("__mean", "__trend"))]
    groups = table["group"].to_numpy()
    target = table["target"].to_numpy(dtype=float)
    rows = []
    for name, observables in FEATURE_SETS.items():
        columns = feature_columns(observables, available)
        if not columns:
            continue
        scores = nested_scores(table[columns], target, groups, task)
        rows.append(
            {
                "window": window,
                "task": task,
                "feature_set": name,
                "n_features": len(columns),
                "n_folds": len(scores),
                "score_mean": float(np.mean(scores)) if scores else float("nan"),
                "score_std": float(np.std(scores)) if scores else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    # The headline number: what the topological block adds to the cheap baselines.
    if {"baselines", "baselines+topology"} <= set(frame["feature_set"]):
        base = frame.loc[frame.feature_set == "baselines", "score_mean"].iloc[0]
        both = frame.loc[frame.feature_set == "baselines+topology", "score_mean"].iloc[0]
        frame["increment_over_baselines"] = np.where(
            frame.feature_set == "baselines+topology", both - base, np.nan
        )
    return frame
