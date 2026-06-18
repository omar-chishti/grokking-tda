# Architecture

This document explains how the `grokking_tda` codebase is put together and *why*.
It is the map to read before changing anything.

## 1. The one idea: three decoupled layers

```
   ┌──────────────────────────┐     ┌───────────────────────────┐     ┌──────────────────────────────┐
   │  LAYER 1 — TRAIN (GPU)    │     │  LAYER 2 — ARTIFACT STORE │     │  LAYER 3 — ANALYSE (CPU)     │
   │                          │     │        (on disk)          │     │                              │
   │  data → model → engine   │ ──▶ │  manifest.json            │ ──▶ │  observables (PH, Fourier,   │
   │  + callbacks             │     │  metrics.jsonl            │     │   weight-norm, LID)          │
   │                          │     │  snapshots/step_*/        │     │  evaluation (lead/lag)       │
   │  run once, expensively   │     │   weights.pt + reps.npz   │     │  vector figures              │
   └──────────────────────────┘     └───────────────────────────┘     └──────────────────────────────┘
```

Training is expensive and seed-sensitive, so it runs **once** and writes a
**self-describing artifact**. Analysis is cheap and iterative, so it runs **many
times** off that artifact, never touching a GPU and never retraining. The two
layers communicate *only* through the artifact store contract (`artifacts/`). This
is the decision everything else follows from — it is what makes the pipeline cheap
to iterate, easy to parallelise on a cluster, and reproducible.

## 2. Module map

```
src/grokking_tda/
  registry.py        Generic typed Registry (name -> factory) used everywhere.
  config/            Typed dataclass schema (schema.py) + Hydra ConfigStore presets (store.py).
  configs/           The Hydra YAML tree: config.yaml + experiment/ + hydra/launcher/.
  data/              modular.py: add/sub/mul tasks, full tensors, deterministic split.
  models/            hooks.py (HookPoint capture) + transformer.py + mlp.py; all expose
                     forward(tokens)->logits and embedding_matrix().
  training/          engine.py (step loop) + callbacks.py + losses.py (float64 / StableMax)
                     + optimizers.py (AdamW/SGD/OrthoGrad) + schedules.py (log-spaced snapshots).
  artifacts/         schema.py (Manifest) + writer.py (+ prepare_run_dir overwrite guard)
                     + reader.py (Run/Snapshot). THE contract.
  analysis/          observable.py (Observable interface + ObservationContext + runner;
                     per-snapshot diagrams disk-cached under analysis/diagrams/)
                     + representations.py (embedding/hidden/logits extraction, train/test split)
                     + aggregate.py (many runs -> one tidy robustness table).
  tda/               pointcloud.py (normalise, drop_first, random/maxmin landmarks)
                     + homology.py (ripser) + summaries.py (max/total/entropy) + observables.py
                     + distances.py (bottleneck/sliced-Wasserstein + trajectory velocity)
                     + significance.py (bootstrap CIs + random-init null models)
                     + trajectory.py (CROCKER — topology of the trajectory).
  baselines/         fourier.py (residue-axis + discrete-log "group" variant)
                     + weight_norm.py + lid.py  (registered as observables).
  evaluation/        transitions.py (t_g, t_top per observable, lead/lag)
                     + predictive.py (pre-registered early windows — never the run's own t_g).
  plotting/          style.py (vector-first, thesis bronze/ink palette) + curves.py
                     + persistence.py + trajectory.py (CROCKER heatmap of the trajectory).
  orchestration/     SLURM/PBS templates + analyse stage script.
  cli/               train.py (Hydra app) + analyse.py + plot.py + aggregate.py (argparse).
  utils/             seeding.py + env.py (provenance) + logging.py
                     + precision.py (device-aware float64) + io.py (atomic JSON).
```

## 3. The four patterns that keep it modular

1. **Registry** (`registry.py`). Models, datasets, optimizers, losses and observables
   each register by name; configs select by name. A new component is *additive* — add
   a factory + register it; no call site changes.
2. **Callbacks** (`training/callbacks.py`). The training loop is ~15 lines; metric
   logging, snapshotting, console output and (future) online probes are callbacks fired
   on `on_step_end`. New training-time behaviour never edits the loop.
3. **Hook points** (`models/hooks.py`). Activations are captured by name via
   `run_with_cache`, so the analysis layer can pull any internal representation without
   the model knowing about TDA.
4. **Observable** (`analysis/observable.py`). *Everything* measured from a snapshot —
   H1 persistence, Fourier concentration, weight norm, LID — is the same kind of object
   `(ctx) -> float`. This is what makes the thesis's central comparison ("what does
   topology add over cheaper diagnostics?") a one-line config change.

## 4. Data flow of a run

1. `cli/train.py` composes an `ExperimentCfg` (Hydra) → builds data + model → creates an
   `ArtifactWriter` → writes `manifest.json` (resolved config + git/env provenance + task
   meta) → runs `Trainer.fit()`.
2. `Trainer` steps the optimizer; `MetricLogger` appends scalars to `metrics.jsonl`;
   `SnapshotSaver` writes log-spaced `snapshots/step_*/` (weights + the cheap embedding
   point cloud, plus optional hidden/logits).
3. `cli/analyse.py` opens the run via `Run`, builds an `AnalysisCfg`, and `run_observables`
   computes the configured observables over every snapshot → `analysis/observables.csv`.
   Per-snapshot persistence diagrams are cached to `analysis/diagrams/` (keyed by the
   construction config) so re-analysis, plotting, and trajectory metrics never recompute
   them. `evaluation` locates `t_g` (grokking) and `t_top` + signed lag for **every**
   observable, plus early-window features at pre-registered windows →
   `analysis/summary.json`; the distance between consecutive H1 diagrams →
   `analysis/trajectory_distance.csv` (topological velocity).
4. `cli/plot.py` renders vector PDFs from the metrics + observables + cached diagrams
   (`--steps` selects stage diagrams). `cli/aggregate.py` joins many runs' configs and
   summaries into one tidy table for the robustness/predictive figures.

Because snapshots store **weights** (authoritative) plus the small embedding matrix,
hidden states and logits are *recomputed deterministically* from weights when needed
(`analysis/representations.py`) — full information, small footprint.

## 5. Reproducibility

- Every run embeds its fully-resolved config, git commit + dirty flag, library versions
  and hardware in `manifest.json`.
- Seeds control weights, the data split, and the batch order; `utils/seeding.py` also
  requests deterministic kernels.
- `uv.lock` pins the exact dependency graph; the Dockerfile builds from it.
- Numerics: float64 cross-entropy by default (avoids softmax-collapse loss spikes).
  On MPS (which lacks float64) high-precision ops degrade to float32 with a one-time
  warning (`utils/precision.py`); CPU/CUDA keep full float64.
- Durability: JSON artifacts are written atomically (temp file + rename) and the snapshot
  `index.json` is written last, so an interrupted run never leaves a partial file that
  breaks a reader. Snapshot weights load with `weights_only=True`.
- Observability: every run writes `events.jsonl` (train start/end, wall-clock, steps/s,
  final metrics, and a `failed` event with the error if it crashes); the console shows
  live throughput and ETA. Analysis tolerates a bad snapshot (records `NaN`, logs, and
  continues) rather than aborting.

## 6. Local vs cluster

- **Laptop (Intel Mac, MPS/CPU):** `device=auto` resolves to MPS; use `+experiment=smoke`.
  torch is pinned to 2.2.2 (last x86-mac wheel) via per-platform markers in `pyproject.toml`.
- **Cluster (Linux/CUDA):** the same lockfile resolves CUDA wheels. Sweeps run either as a
  Hydra+submitit multirun (`hydra/launcher=imperial_slurm`) or via the scheduler-agnostic
  SLURM/PBS templates in `orchestration/templates/`. Run training (GPU) and analysis (CPU)
  as two stages — never recompute training to re-analyse.

## 7. How to extend

- **New task** (e.g. modular polynomial): add a builder in `data/`, register it, add a
  `data/` config preset.
- **New model**: subclass with `HookPoint`s, expose `embedding_matrix()` + `hidden_hook`,
  register in `models/__init__.py`.
- **New observable** (e.g. persistence landscape, a new baseline): write `(ctx) -> float`
  and decorate with `@register_observable("name")`; add the name to `AnalysisCfg.observables`.
- **New intervention**: add a loss in `losses.py` or wrap an optimizer in `optimizers.py`.

See `docs/decisions/` for the rationale behind the major choices.
