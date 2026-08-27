"""The typed configuration schema.

These dataclasses are the single source of truth for what a run *is*. Hydra
validates composed YAML/CLI overrides against them, and the fully-resolved
instance is serialised into every run's manifest. Keeping the schema explicit
(rather than free-form dicts) means a typo like ``weight_deacy=0.5`` fails fast
instead of silently doing nothing.

Design note: ``ModelCfg`` is a deliberate *superset* of the fields used by the
transformer and the MLP. A single flat schema keeps Hydra composition simple;
each model builder reads only the fields it needs (and we document which).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataCfg:
    """The dataset / task definition."""

    task: str = "modular_arithmetic"  # registry key in grokking_tda.data
    operation: str = "add"  # add | sub | mul | div | poly  (binary op on residues mod p)
    modulus: int = 97  # the prime p; |dataset| = p*p (p*(p-1) for div, which excludes b=0)
    train_fraction: float = 0.3  # fraction of the input pairs used for training
    label_permutation: bool = False  # permute labels (destroys the rule; Tang's control / nulls)
    n_symbols: int = 5  # symmetric-group tasks only: S_n, order n!


@dataclass
class ModelCfg:
    """Model hyper-parameters (superset across architectures; ``name`` selects one)."""

    name: str = "transformer"  # registry key: transformer | mlp
    # --- transformer fields ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    d_mlp: int = 512
    act: str = "relu"  # relu | gelu
    use_layernorm: bool = True
    # --- mlp fields ---
    embedding_dim: int = 128
    hidden_dim: int = 256
    depth: int = 2


@dataclass
class OptimCfg:
    """Optimizer configuration. ``name`` selects from the optimizer registry."""

    name: str = "adamw"  # adamw | sgd | orthograd_adamw | orthograd_sgd
    lr: float = 1e-3
    weight_decay: float = 1.0  # strong WD is the canonical grokking regime
    betas: list[float] = field(default_factory=lambda: [0.9, 0.98])
    eps: float = 1e-8
    momentum: float = 0.0  # used by SGD-family only


@dataclass
class TrainCfg:
    """Everything about *how* training runs and *when* we record state."""

    steps: int = 40_000  # optimizer steps (grokking is a step-scale phenomenon)
    batch_size: int | None = None  # None => full-batch GD (canonical for mod-arith)
    loss: str = "softmax_ce"  # softmax_ce | stablemax_ce  (loss registry key)
    loss_dtype: str = "float64"  # high-precision CE avoids softmax-collapse artefacts
    optimizer: OptimCfg = field(default_factory=OptimCfg)
    device: str = "auto"  # auto | cuda | mps | cpu
    deterministic: bool = True
    # --- recording schedules (decoupled on purpose) ---
    metric_every: int = 100  # cadence (steps) for cheap scalar metrics
    n_snapshots: int = 60  # number of heavy snapshots (weights + representations)
    snapshot_schedule: str = "log"  # log | linear  spacing of snapshots over steps
    # Optional dense window, for resolving the transition finely enough to time it.
    dense_from: int = 0
    dense_to: int = 0
    capture_representations: bool = True  # cache embeddings/activations into snapshots
    keep_optimizer_state: bool = False  # snapshots are for analysis, not resuming
    # Dense record of the optimisation path, projected (0 => off). Snapshots are far
    # too sparse to treat the trajectory as a point cloud; see training/trajectory.py.
    trajectory_dim: int = 0
    trajectory_every: int = 10


@dataclass
class PointCloudCfg:
    """How a snapshot's representation matrix becomes a point cloud for PH."""

    normalize: str = "center"  # none | center | unit_norm | standardize
    metric: str = "euclidean"  # euclidean | cosine
    max_points: int = 0  # 0 => use all rows; otherwise landmark subsample to this count
    subsample: str = "random"  # random | maxmin  (maxmin: greedy farthest-point landmarks)
    drop_first: bool = False  # exclude row 0 (residue 0 sits outside the multiplicative group
    # for mul/div; its embedding is a structural outlier that can manufacture spurious bars)


@dataclass
class HomologyCfg:
    """Vietoris-Rips persistent-homology parameters (ripser)."""

    maxdim: int = 1  # compute H0..H_maxdim
    coeff: int = 2  # field coefficient (Z/2)
    thresh: float = -1.0  # -1 => no threshold (ripser default infinity)


@dataclass
class AnalysisCfg:
    """The offline analysis recipe applied to a run's snapshots."""

    representation: str = "embedding"  # embedding | hidden | logits
    # Rows used for hidden/logit clouds. Tang builds them from the *test* split; "all"
    # is the full p*p table. The embedding cloud (p residue rows) ignores this.
    representation_split: str = "all"  # all | train | test
    cache_diagrams: bool = True  # persist per-snapshot diagrams under analysis/diagrams/
    pointcloud: PointCloudCfg = field(default_factory=PointCloudCfg)
    homology: HomologyCfg = field(default_factory=HomologyCfg)
    observables: list[str] = field(
        default_factory=lambda: [
            "h1_max_persistence",
            "h1_total_persistence",
            "h1_max_persistence_normalised",
            "h1_total_persistence_normalised",
            "pointcloud_scale",
            "h1_persistence_entropy",
            "h0_total_persistence",
            "weight_norm",
            "fourier_concentration",
            "fourier_concentration_group",
            # The k sweep: a redundancy claim against Fourier must hold against the
            # strongest member of the family, not against an arbitrary default.
            *[f"fourier_concentration_k{k}" for k in (1, 2, 3, 5, 10, 20)],
            *[f"fourier_concentration_group_k{k}" for k in (1, 2, 3, 5, 10, 20)],
            "lid",
            "test_acc",
            "test_acc_novel",
        ]
    )


@dataclass
class ExperimentCfg:
    """The top-level config: one fully-specified, reproducible experiment."""

    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    seed: int = 0
    output_root: str = "results/raw"
    # Refuse to reuse a non-empty run directory unless this is set (the artifact writer
    # would otherwise truncate logs over stale snapshots — a mixed-run artifact).
    overwrite: bool = False
    # Auto-generated, human-readable run name (OmegaConf interpolation from root).
    # Carries every knob the planned sweeps vary, so distinct configs never collide.
    run_name: str = (
        "${model.name}_${data.operation}${data.modulus}"
        "_f${data.train_fraction}_wd${train.optimizer.weight_decay}_${train.loss}_s${seed}"
    )
    notes: str = ""
