"""The typed configuration schema: the single source of truth for what a run *is*."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataCfg:
    task: str = "modular_arithmetic"  # registry key in grokking_tda.data
    operation: str = "add"  # add | sub | mul | div | poly  (binary op on residues mod p)
    modulus: int = 97  # the prime p; |dataset| = p*p (p*(p-1) for div, which excludes b=0)
    train_fraction: float = 0.3
    label_permutation: bool = False  # destroys the rule; Tang's control and the nulls
    n_symbols: int = 5  # symmetric-group tasks only: S_n, order n!


@dataclass
class ModelCfg:
    name: str = "transformer"  # registry key: transformer | mlp
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    d_mlp: int = 512
    act: str = "relu"  # relu | gelu
    use_layernorm: bool = True
    embedding_dim: int = 128
    hidden_dim: int = 256
    depth: int = 2


@dataclass
class OptimCfg:
    name: str = "adamw"  # adamw | sgd | orthograd_adamw | orthograd_sgd
    lr: float = 1e-3
    weight_decay: float = 1.0  # strong WD is the canonical grokking regime
    betas: list[float] = field(default_factory=lambda: [0.9, 0.98])
    eps: float = 1e-8
    momentum: float = 0.0  # used by SGD-family only


@dataclass
class TrainCfg:
    steps: int = 40_000
    batch_size: int | None = None  # None => full-batch GD (canonical for mod-arith)
    loss: str = "softmax_ce"  # softmax_ce | stablemax_ce  (loss registry key)
    loss_dtype: str = "float64"  # high-precision CE avoids softmax-collapse artefacts
    optimizer: OptimCfg = field(default_factory=OptimCfg)
    device: str = "auto"  # auto | cuda | mps | cpu
    deterministic: bool = True
    metric_every: int = 100
    n_snapshots: int = 60  # each writes weights and representations to disk
    snapshot_schedule: str = "log"  # log | linear  spacing of snapshots over steps
    # dense window, to resolve the transition finely enough to time it
    dense_from: int = 0
    dense_to: int = 0
    capture_representations: bool = True
    keep_optimizer_state: bool = False  # snapshots are for analysis, not resuming
    # projected optimisation path (0 => off); snapshots are too sparse for a point cloud
    trajectory_dim: int = 0
    trajectory_every: int = 10


@dataclass
class PointCloudCfg:
    normalize: str = "center"  # none | center | unit_norm | standardize
    metric: str = "euclidean"  # euclidean | cosine
    max_points: int = 0  # 0 => every row
    subsample: str = "random"  # random | maxmin  (maxmin: greedy farthest-point landmarks)
    # residue 0 is outside the multiplicative group for mul and div, and a structural outlier
    # that can manufacture spurious bars
    drop_first: bool = False


@dataclass
class HomologyCfg:
    maxdim: int = 1  # computes H0..H_maxdim
    coeff: int = 2  # field coefficient (Z/2)
    thresh: float = -1.0  # -1 => no threshold (ripser default infinity)


@dataclass
class AnalysisCfg:
    representation: str = "embedding"  # embedding | hidden | logits
    # rows of the hidden/logit clouds, which Tang draws from the test split; embeddings ignore it
    representation_split: str = "all"  # all | train | test
    cache_diagrams: bool = True
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
            # swept: the redundancy claim must hold against the family's strongest member
            *[f"fourier_concentration_k{k}" for k in (1, 2, 3, 5, 10, 20)],
            *[f"fourier_concentration_group_k{k}" for k in (1, 2, 3, 5, 10, 20)],
            "lid",
            "test_acc",
            "test_acc_novel",
        ]
    )


@dataclass
class ExperimentCfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    seed: int = 0
    output_root: str = "results/raw"
    # the writer truncates its logs on start, so reusing a directory would mix two runs
    overwrite: bool = False
    # A readable label, not a key: it omits the optimiser, the learning rate, the architecture,
    # the batch size and the label permutation, so two distinct configurations can generate one
    # name. A sweep that moves a field the template omits sets `run_name` itself, as the recipe,
    # learning-rate and interaction programmes in `experiments/` do.
    run_name: str = (
        "${model.name}_${data.operation}${data.modulus}"
        "_f${data.train_fraction}_wd${train.optimizer.weight_decay}_${train.loss}_s${seed}"
    )
    notes: str = ""
