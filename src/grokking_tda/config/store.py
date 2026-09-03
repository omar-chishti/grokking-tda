"""Register the schema and config-group presets with Hydra's ConfigStore, as typed instances."""

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
    cs = ConfigStore.instance()

    cs.store(name="experiment_base", node=ExperimentCfg)

    cs.store(group="model", name="transformer", node=ModelCfg(name="transformer"))
    cs.store(group="model", name="mlp", node=ModelCfg(name="mlp"))
    # Tang et al. §3: 2 pre-LN encoder blocks with GELU / a 3-hidden-layer width-512 MLP.
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

    cs.store(group="data", name="mod_add_p97", node=DataCfg("modular_arithmetic", "add", 97, 0.3))
    # Prieto et al. run the interventions at fraction 0.4
    cs.store(
        group="data", name="mod_add_p97_f04", node=DataCfg("modular_arithmetic", "add", 97, 0.4)
    )
    cs.store(group="data", name="mod_add_p113", node=DataCfg("modular_arithmetic", "add", 113, 0.3))
    cs.store(group="data", name="mod_add_p149", node=DataCfg("modular_arithmetic", "add", 149, 0.3))
    cs.store(group="data", name="mod_sub_p97", node=DataCfg("modular_arithmetic", "sub", 97, 0.3))
    cs.store(group="data", name="mod_mul_p97", node=DataCfg("modular_arithmetic", "mul", 97, 0.5))
    cs.store(group="data", name="mod_div_p97", node=DataCfg("modular_arithmetic", "div", 97, 0.5))
    cs.store(group="data", name="mod_poly_p97", node=DataCfg("modular_arithmetic", "poly", 97, 0.5))
    # non-abelian, so no circle respects the group operation
    cs.store(
        group="data",
        name="s5_composition",
        # the group order, so run names read compose120 rather than the modular default
        node=DataCfg(
            task="permutation_group", operation="compose", modulus=120, train_fraction=0.5
        ),
    )
    # the operator as a token: one model, both operations, R24 (a side quest, not the programme)
    cs.store(
        group="data",
        name="mod_addsub_p113",
        node=DataCfg(task="modular_multiop", operation="add+sub", modulus=113,
                     train_fraction=0.3),
    )
    cs.store(
        group="data",
        name="mod_addsub_p97",
        node=DataCfg(task="modular_multiop", operation="add+sub", modulus=97,
                     train_fraction=0.3),
    )
    cs.store(
        group="data",
        name="mod_addsub_p11_smoke",
        node=DataCfg(task="modular_multiop", operation="add+sub", modulus=11,
                     train_fraction=0.5),
    )
    smoke = DataCfg("modular_arithmetic", "add", 11, 0.5)
    cs.store(group="data", name="mod_add_p11_smoke", node=smoke)

    cs.store(group="train", name="full_batch_adamw", node=TrainCfg())
    # Prieto et al.: lr 1e-2, no weight decay, beta2 raised; applied separately, so the
    # combination is a third condition
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
    # Tang et al.: minibatch AdamW, lr 3e-3, wd 0.1, eps 1e-6, 60k steps, 120 snapshots.
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

    cs.store(group="analysis", name="default", node=AnalysisCfg())
