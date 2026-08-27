# grokking-tda

[![ci](https://github.com/omar-chishti/grokking-tda/actions/workflows/ci.yml/badge.svg)](https://github.com/omar-chishti/grokking-tda/actions/workflows/ci.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)

A research framework for asking whether **topological data analysis** explains **grokking** —
and for answering honestly when it does not.

It trains small models on modular arithmetic, records each training trajectory as an immutable
artifact, and measures persistent homology over training *always alongside* the cheap
non-topological diagnostics it would have to beat: Fourier concentration, weight norm, local
intrinsic dimension. The comparison is not a section of the analysis. It is the design.

![A transformer memorises modular addition at step 200 and generalises at step 28,600 — 143 times later.](docs/assets/grokking.png)

## What it found

Across **145 training runs** (99 of which grok), on modular addition, subtraction, multiplication,
division, a memorisable polynomial control, permuted-label nulls, and composition in the
non-abelian group *S*₅:

- **The published signature is confounded with scale.** Persistence carries the units of the point
  cloud it is measured on, and under weight decay that cloud's own scale moves by up to **239×**
  across training. A rise in raw H₁ persistence can be the cloud growing rather than its shape
  changing. Dividing by the cloud's connectivity scale (ADR&nbsp;0005) narrows the null band from
  **0.54–1.54** to **0.75–1.32** — nearly half, in log units.
- **Normalising strengthens the result rather than dissolving it.** Against a null band built from
  sixteen runs that fit their training data and never generalise, **6 of 30** experimental
  conditions clear the band on normalised H₁ maximum persistence, against **1 of 30** on the raw
  quantity.
- **But most of what topology sees, a Fourier baseline already sees.** The normalised persistence
  ratio tracks a spectral circularity measure at **ρ = 0.69** (*n* = 75, *p* ≈ 6 × 10⁻¹²), stable
  across every choice of *k* from 1 to 20. The loop persistent homology detects is, largely, the
  circle the Fourier analysis reports.
- **Topology does not predict.** In a nested, configuration-grouped head-to-head on pre-registered
  early windows, cheap baselines reach **0.77–0.81** AUC at predicting whether a run will grok;
  topological features reach **0.68–0.83**; adding topology to the baselines does not improve them.
- **Topology lags.** It is a description of the transition, not an early warning of it —
  reproducing the timing result in the literature rather than overturning it.

The framework is built so that each of those statements is a command, not a claim. See
[`analysis/README.md`](analysis/README.md).

## The one idea

Three layers that talk only through an on-disk contract:

```
  ┌────────────────────────┐    ┌───────────────────────┐    ┌──────────────────────────┐
  │  TRAIN  (GPU)          │    │  ARTIFACT STORE       │    │  ANALYSE  (CPU)          │
  │  data → model → engine │ ─▶ │  manifest.json        │ ─▶ │  observables: PH, Fourier│
  │  + callbacks           │    │  metrics.jsonl        │    │  weight norm, LID        │
  │                        │    │  snapshots/step_*/    │    │  lead/lag, significance  │
  │  run once, expensively │    │   weights + reps      │    │  vector figures          │
  └────────────────────────┘    └───────────────────────┘    └──────────────────────────┘
```

Training is expensive and seed-sensitive, so it runs **once** and writes a self-describing
artifact. Analysis is cheap and iterated hundreds of times, so it runs off that artifact — never
touching a GPU, never retraining. Every design decision in the repository follows from this one.

Details in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the reasoning behind the major choices
in [`docs/decisions/`](docs/decisions/).

## Quickstart

```bash
uv sync                                     # resolves the lockfile: CUDA on Linux, MPS/CPU on macOS
uv run gtda-train +experiment=smoke         # ~10 s end-to-end, proves the chain works

RUN=results/raw/transformer_add11_f0.5_wd1.0_softmax_ce_s0
uv run gtda-analyse $RUN                    # observables + transitions, off the stored artifact
uv run gtda-plot    $RUN                    # vector PDFs
```

Then a real one. A canonical grokking reproduction is minutes on a GPU and about an hour on a
laptop CPU:

```bash
uv run gtda-train +experiment=tf_mod97_grok train.device=cpu
uv run gtda-analyse results/raw/transformer_add97_f0.3_wd1.0_softmax_ce_s0
```

> On Apple silicon, pass `train.device=cpu` for anything you intend to report: MPS has no float64,
> so the high-precision cross-entropy this project relies on silently degrades to float32.

Configs compose from typed presets under `src/grokking_tda/configs/` — one group each for
`model/`, `data/`, `train/`, `analysis/`, plus `experiment/` presets that combine them. Any field
is overridable from the command line, and every run writes its fully-resolved config, the git
commit, the library versions and the hardware into `manifest.json`.

```bash
uv run gtda-train +experiment=mlp_mod97_grok                      # MLP track
uv run gtda-train +experiment=tf_mod97_stablemax                  # StableMax intervention
uv run gtda-train +experiment=tf_s5_grok                          # composition in S_5
uv run gtda-train data=mod_add_p113 train.optimizer.weight_decay=0.5 seed=2
uv run gtda-train -m +experiment=tf_mod97_grok seed=0,1,2         # multirun sweep
```

## What a run leaves behind

```
results/raw/<run_name>/
  manifest.json                    resolved config + git/env provenance + task metadata
  metrics.jsonl                    step-indexed train/test loss and accuracy, weight norm
  events.jsonl                     lifecycle and timing (start, end, steps/s, failures)
  snapshots/step_*/                weights.pt + representations.npz, log-spaced
  analysis/observables.csv         every observable over every snapshot
  analysis/summary.json            t_g, t_top per observable, signed lag, early-window features
  analysis/trajectory_distance.csv topological velocity between consecutive diagrams
  figures/                         vector PDFs
```

## What is in this repository

| | |
|---|---|
| `src/grokking_tda/` | the framework — models, engine, artifact contract, TDA, baselines, evaluation |
| `analysis/` | the reduction layer: a bank of runs → the tables and figures a write-up quotes |
| `tests/` | 97 tests over the framework |
| `experiments/*.runs` | the run programme as checked-in data — the pre-registration |
| `scripts/remote/` | orchestration for a pool of shared GPU workstations ([docs/CLUSTER.md](docs/CLUSTER.md)) |
| `docs/` | architecture and the architecture decision records |

The split between the first two is the rule that keeps the numbers honest: **`src/` holds the
method** — observables, detectors, estimators — installed, imported and unit tested; **`analysis/`
holds the reduction**, which imports the method and never reimplements it. Method code has tests;
reduction code has a committed output. `analysis/` runs from the repository root as
`uv run python -m analysis.<name>`.

## Reproducing the numbers

With an artifact store under `results/raw/`:

```bash
uv run gtda-aggregate results/raw --out results/processed/robustness.csv
uv run python -m analysis.thesis_numbers     # every headline number, as a printed ledger
uv run python -m analysis.figures.build      # one tidy CSV per figure, plus a claims manifest
uv run python -m analysis.figures.render     # vector PDFs from those CSVs
```

`thesis_numbers` writes `claims.json`: every quoted number beside the inputs it was computed from.
The remaining drivers — `normalisation`, `circularity`, `predictive`, `pid`, `redundancy`,
`torus`, `phdim`, `velocity`, `detector`, `instability` — each answer one question and are
documented in [`analysis/README.md`](analysis/README.md).

## Development

```bash
uv run pytest                              # 97 tests
uv run ruff check src tests analysis scripts
uv run mypy
docker build -t grokking-tda .             # CUDA image built from uv.lock
```

## Provenance

The framework was written through May and June 2026 and placed under version control on
18 June, so the history opens with a single baseline commit rather than with the framework's
construction. Everything after that date is committed as it landed.

## Licence and citation

Code is MIT ([`LICENSE`](LICENSE)). The accompanying thesis text is not in this repository and is
not MIT-licensed. If you use this framework, cite it via [`CITATION.cff`](CITATION.cff).
