"""Guards for the choices upstream of every topological number.

Point-cloud construction, the snapshot schedule and the registry decide what is
measured, at which steps, and by whose factory. None of the three shows up in a
diagram when it goes wrong; the numbers simply become about something else.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from grokking_tda.config.schema import PointCloudCfg
from grokking_tda.models.transformer import GrokkingTransformer
from grokking_tda.registry import Registry
from grokking_tda.tda.pointcloud import build_point_cloud
from grokking_tda.tda.significance import bootstrap_summary_ci
from grokking_tda.training.schedules import snapshot_steps


def _circle(n: int, radius: float = 1.0, dim: int = 4) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cloud = np.zeros((n, dim))
    cloud[:, 0], cloud[:, 1] = radius * np.cos(theta), radius * np.sin(theta)
    return cloud


def test_maxmin_landmarks_cover_a_circle_where_a_random_draw_need_not() -> None:
    """The reason maxmin is the default for H1/H2: it leaves no long empty arc, so the
    loop survives the subsample."""
    cloud = _circle(200)
    cfg = PointCloudCfg(normalize="none", max_points=16, subsample="maxmin")
    landmarks = build_point_cloud(cloud, cfg, seed=0)

    angles = np.sort(np.arctan2(landmarks[:, 1], landmarks[:, 0]))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
    assert len(landmarks) == 16
    assert gaps.max() < 2.5 * (2 * np.pi / 16)


def test_maxmin_landmarks_keep_an_outlier_a_random_draw_would_usually_miss() -> None:
    rng = np.random.default_rng(0)
    cloud = np.vstack([rng.normal(size=(30, 2)) * 0.1, [[100.0, 0.0]]])
    cfg = PointCloudCfg(normalize="none", max_points=5, subsample="maxmin")
    landmarks = build_point_cloud(cloud, cfg, seed=0)
    assert len(landmarks) == 5
    assert np.linalg.norm(landmarks, axis=1).max() > 50


def test_hook_names_are_unique_across_blocks() -> None:
    """A last-component name collides at two blocks, and the representation read out
    would then depend on which one registered last."""
    model = GrokkingTransformer(vocab_size=12, num_classes=11, seq_len=3, n_layers=2)
    _, cache = model.run_with_cache(torch.zeros((2, 3), dtype=torch.long))
    assert "blocks.0.hook_attn_out" in cache
    assert "blocks.1.hook_attn_out" in cache
    assert "hook_resid_final" in cache  # top-level hooks keep their bare name


def test_normalisation_and_drop_first_do_what_the_config_names_claim() -> None:
    cloud = np.arange(24, dtype=float).reshape(6, 4)
    assert build_point_cloud(cloud, PointCloudCfg(drop_first=True)).shape[0] == 5

    centred = build_point_cloud(cloud, PointCloudCfg(normalize="center"))
    assert np.allclose(centred.mean(axis=0), 0.0)

    unit = build_point_cloud(cloud, PointCloudCfg(normalize="unit_norm"))
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)


def test_an_unknown_construction_choice_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="normalize"):
        build_point_cloud(np.zeros((4, 2)), PointCloudCfg(normalize="whiten"))
    with pytest.raises(ValueError, match="subsample"):
        build_point_cloud(np.zeros((8, 2)), PointCloudCfg(max_points=4, subsample="every-other"))


@pytest.mark.slow
def test_bootstrap_interval_brackets_the_radius_of_a_circle() -> None:
    """H1 max persistence of a circle of radius r is order r; the interval must contain
    the full-sample value rather than sit to one side of it."""
    out = bootstrap_summary_ci(_circle(60, radius=2.0, dim=2), n_boot=25, seed=0)
    interval = out["max"]
    assert interval["lo"] <= interval["median"] <= interval["hi"]
    assert 1.0 < interval["median"] < 4.0
    assert len(out["samples"]) == 25


def test_log_schedule_is_dense_early_and_keeps_both_endpoints() -> None:
    steps = snapshot_steps(10_000, 20, schedule="log")
    assert steps[0] == 0 and steps[-1] == 10_000
    assert steps == sorted(set(steps))
    assert sum(s < 1_000 for s in steps) > sum(s >= 1_000 for s in steps)


def test_a_dense_window_spends_half_the_budget_inside_it() -> None:
    """The window exists so the signed lag is resolved finer than the lag itself."""
    plain = snapshot_steps(10_000, 40, schedule="log")
    dense = snapshot_steps(10_000, 40, schedule="log", dense_from=4_000, dense_to=6_000)
    inside = sum(4_000 <= s <= 6_000 for s in dense)
    assert inside > 4 * sum(4_000 <= s <= 6_000 for s in plain)
    assert dense[0] == 0 and dense[-1] == 10_000


def test_registry_refuses_a_duplicate_name_and_names_the_alternatives() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("a")(lambda: "built")

    assert registry.build("a") == "built"
    assert "a" in registry and list(registry.keys()) == ["a"]
    with pytest.raises(KeyError, match="already has"):
        registry.register("a")(lambda: "shadow")
    with pytest.raises(KeyError, match=r"available: \['a'\]"):
        registry.build("b")
