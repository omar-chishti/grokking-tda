# grokking-tda

A research framework for studying **grokking** through **topological data analysis**.

It trains small models on modular arithmetic, records their training trajectory as
immutable artifacts, and measures topological signatures (persistent homology) over
training — always alongside non-topological baselines (Fourier concentration, weight
norm, local intrinsic dimension) so the central question, *what does topology add?*,
is built into the design.

See **`docs/ARCHITECTURE.md`** for the design and **`docs/decisions/`** for the rationale.

## Quickstart

```bash
# 1. Environment (uv resolves the locked deps; macOS gets MPS/CPU torch, Linux gets CUDA)
uv sync

# 2. A tiny end-to-end run on your laptop (CPU/MPS)
uv run gtda-train +experiment=smoke

# 3. A canonical grokking reproduction (transformer, mod 97) — longer
uv run gtda-train +experiment=tf_mod97_grok

# 4. Analyse + plot a run (cheap, CPU; reads the artifact store)
uv run gtda-analyse results/raw/transformer_add97_s0
uv run gtda-plot    results/raw/transformer_add97_s0
```

## Experiments and overrides

```bash
uv run gtda-train +experiment=mlp_mod97_grok            # MLP architecture track
uv run gtda-train +experiment=tf_mod97_stablemax        # StableMax + OrthoGrad intervention
uv run gtda-train data=mod_add_p113 train.optimizer.weight_decay=0.5 seed=2
uv run gtda-train -m +experiment=tf_mod97_grok seed=0,1,2   # local multirun sweep
```

Configs compose from typed presets (`src/grokking_tda/configs/`): one per group
(`model/`, `data/`, `train/`, `analysis/`) plus `experiment/` presets. Every run
writes its fully-resolved config to `manifest.json`.

## Output layout

```
results/raw/<run_name>/
  manifest.json                      # resolved config + git/env provenance + task meta
  metrics.jsonl                      # step-indexed train/test loss+acc, weight norm
  events.jsonl                       # lifecycle/timing events (start/end, steps/s, failures)
  snapshots/step_*/                  # weights.pt + representations.npz (log-spaced)
  analysis/observables.csv           # written by gtda-analyse
  analysis/summary.json              # grokking step, topological transition, lead/lag
  figures/                           # written by gtda-plot (vector-first PDFs):
    training_curves.pdf              #   accuracy/loss vs step
    observables_over_time.pdf        #   each observable (PH + baselines) vs step
    persistence_diagram_final.pdf    #   final-snapshot birth-death diagram
    crocker_h1.pdf                   #   CROCKER: topology *of* the trajectory
```

## Cluster

```bash
# Hydra + submitit (SLURM):
uv run gtda-train -m +experiment=tf_mod97_grok seed=0,1,2,3,4 hydra/launcher=imperial_slurm

# or the scheduler-agnostic templates (SLURM or PBS):
sbatch --export=EXPERIMENT=tf_mod97_grok src/grokking_tda/orchestration/templates/slurm_array.sbatch
qsub   -v EXPERIMENT=tf_mod97_grok       src/grokking_tda/orchestration/templates/pbs_array.pbs

# Stage 2 (CPU): analyse every run
bash src/grokking_tda/orchestration/templates/analyse_stage.sh results/raw
```

Set the real partition/account in `configs/hydra/launcher/imperial_slurm.yaml` once
the Imperial scheduler is confirmed (`Open_Questions.md`).

## Docker

```bash
docker build -t grokking-tda .                    # builds from uv.lock (CUDA base)
docker run --gpus all grokking-tda +experiment=tf_mod97_grok
```

## Development

```bash
uv run pytest          # tests
uv run ruff check src tests   # lint
uv run mypy            # types
```
