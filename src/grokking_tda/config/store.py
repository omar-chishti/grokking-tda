"""Register the schema and config-group presets with Hydra's ConfigStore.

Group presets live here as *typed Python instances* (not YAML) so they are
validated at definition time and stay close to the schema. YAML is reserved for
the thin top-level ``config.yaml`` and the ``experiment/`` presets, which is
where composition/override actually happens.

Composition model (Hydra defaults list in ``configs/config.yaml``)::

    experiment_base   # full typed defaults (ExperimentCfg)
      + model: transformer
      + data:  mod_add_p97
      + train: full_batch_adamw
      + analysis: default

An ``experiment/`` preset then overrides any of those groups in one file.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from grokking_tda.config.schema import (
    AnalysisCfg,
    DataCfg,
    ExperimentCfg,
    ModelCfg,
    OptimCfg,
    TrainCfg,
)


def register_configs() -> None:
    """Idempotently register the schema and all group presets."""
    cs = ConfigStore.instance()

    # Top-level schema (full defaults; runnable as-is).
    cs.store(name="experiment_base", node=ExperimentCfg)

    # --- model group ---
    cs.store(group="model", name="transformer", node=ModelCfg(name="transformer"))
    cs.store(group="model", name="mlp", node=ModelCfg(name="mlp"))
    # Tang et al. architectures (faithful reproduction, PDF §3): 2 pre-LN encoder
    # blocks with GELU / a 3-hidden-layer width-512 GELU MLP.
    cs.store(
        group="model",
        name="transformer_tang",
        node=ModelCfg(
            name="transformer",
            n_layers=2,
            d_model=128,
            n_heads=4,
            d_mlp=256,
            act="gelu",
            use_layernorm=True,
        ),
    )
    cs.store(
        group="model",
        name="mlp_tang",
        node=ModelCfg(name="mlp", embedding_dim=128, hidden_dim=512, depth=3, act="gelu"),
    )

    # --- data group ---
    cs.store(group="data", name="mod_add_p97", node=DataCfg("modular_arithmetic", "add", 97, 0.3))
    # Prieto et al. run the interventions at fraction 0.4.
    cs.store(
        group="data", name="mod_add_p97_f04", node=DataCfg("modular_arithmetic", "add", 97, 0.4)
    )
    cs.store(group="data", name="mod_add_p113", node=DataCfg("modular_arithmetic", "add", 113, 0.3))
    cs.store(group="data", name="mod_add_p149", node=DataCfg("modular_arithmetic", "add", 149, 0.3))
    cs.store(group="data", name="mod_sub_p97", node=DataCfg("modular_arithmetic", "sub", 97, 0.3))
    cs.store(group="data", name="mod_mul_p97", node=DataCfg("modular_arithmetic", "mul", 97, 0.5))
    cs.store(group="data", name="mod_div_p97", node=DataCfg("modular_arithmetic", "div", 97, 0.5))
    cs.store(group="data", name="mod_poly_p97", node=DataCfg("modular_arithmetic", "poly", 97, 0.5))
    smoke = DataCfg("modular_arithmetic", "add", 11, 0.5)
    cs.store(group="data", name="mod_add_p11_smoke", node=smoke)

    # --- train group ---
    cs.store(group="train", name="full_batch_adamw", node=TrainCfg())
    # Prieto et al. interventions. Their published settings are lr 1e-2 with no weight
    # decay, train fraction 0.4, and beta2 raised (0.999 for StableMax, 0.99 for OrthoGrad);
    # they apply the two separately, so each gets its own preset and the combination is a
    # third condition rather than the default.
    cs.store(
        group="train",
        name="stablemax",
        node=TrainCfg(
            loss="stablemax_ce",
            optimizer=OptimCfg(name="adamw", lr=1e-2, weight_decay=0.0, betas=[0.9, 0.999]),
        ),
    )
    cs.store(
        group="train",
        name="orthograd",
        node=TrainCfg(
            optimizer=OptimCfg(
                name="orthograd_adamw", lr=1e-2, weight_decay=0.0, betas=[0.9, 0.99]
            ),
        ),
    )
    cs.store(
        group="train",
        name="stablemax_orthograd",
        node=TrainCfg(
            loss="stablemax_ce",
            optimizer=OptimCfg(
                name="orthograd_adamw", lr=1e-2, weight_decay=0.0, betas=[0.9, 0.999]
            ),
        ),
    )
    cs.store(
        group="train",
        name="smoke",
        node=TrainCfg(steps=200, n_snapshots=8, metric_every=20),
    )
    # Tang et al. training recipe: minibatch AdamW, lr 3e-3, wd 0.1, eps 1e-6,
    # 60k steps, snapshots every 500 steps (120 linear snapshots).
    cs.store(
        group="train",
        name="tang",
        node=TrainCfg(
            steps=60_000,
            batch_size=512,
            n_snapshots=120,
            snapshot_schedule="linear",
            metric_every=500,
            optimizer=OptimCfg(
                name="adamw", lr=3e-3, weight_decay=0.1, eps=1e-6, betas=[0.9, 0.98]
            ),
        ),
    )

    # --- analysis group ---
    cs.store(group="analysis", name="default", node=AnalysisCfg())
