# Running training — local and on the Imperial GPU cluster

How to launch `grokking_tda` runs, where to run them, and the gotchas that matter for
this project's numerics and provenance. Pairs with the first-run runbook in
`Documentation/Code_Review_Findings.md` and the experiment programme in
`Documentation/Experiment_Design.md`.

---

## TL;DR — where to run what

| Phase | Runs | Recommendation |
|---|---|---|
| **This weekend — G0/G1 gates (R1, R3)** | ~10 short runs | **Run locally on CPU.** Faster end-to-end than fighting a queue, and it preserves float64. |
| **July robustness sweeps (R4–R8)** | ~100 runs | **Cluster.** Embarrassingly parallel as array jobs; the wall-clock win is real here. |
| **Trajectory / interventions (Aug)** | tens | Either; cluster if sweeping. |

**Why local is fine — and often better — for the gate runs.** The models are tiny (1–2 layer
transformer, p ≤ 149, full-batch), so one run is **minutes on a GPU but only ~15–30 min on a CPU**.
For 5–10 runs, local CPU avoids queue latency, module-load friction, and data round-trips entirely.
The cluster earns its keep when you launch ~100 runs that can fan out across array tasks.

**The numerics caveat that decides the device.** The default loss is **float64 cross-entropy**
(avoids the softmax-collapse loss spikes that distort grokking). Support by device:

- **CPU** ✅ float64 — the safe default for the dev box.
- **CUDA** (cluster) ✅ float64 — use `train.device=cuda`.
- **MPS** (Apple GPU) ❌ no float64 — it **silently degrades to float32** with a one-time warning.
  Do **not** run the real grokking experiments on MPS; use `train.device=cpu` on the Mac.

---

## 1. Local runs (recommended for the weekend)

From `Code/`:

```sh
# 0. smoke (~10 s) — confirm the chain still works
uv run gtda-train +experiment=smoke train.device=cpu

# 1. R1 — canonical reproduction, 5 seeds (the G0 gate)
uv run gtda-train -m +experiment=tf_mod97_grok seed=0,1,2,3,4 \
    train.device=cpu train.capture_representations=false

# 2. analyse + plot every run (cheap, CPU, re-runnable)
bash src/grokking_tda/orchestration/templates/analyse_stage.sh results/raw
```

- `-m` is Hydra **multirun**: it runs the seeds in sequence locally (each writes its own run dir,
  since `run_name` encodes the seed). Drop `-m` and pass a single `seed=` to run just one.
- `capture_representations=false` keeps disk small (~tens of MB/run vs ~0.5 GB). You lose nothing
  analytically — `analysis/representations.py` recomputes hidden states / logits from the saved
  weights on demand. Leave it `false` unless you specifically want them cached.
- **Confirm G0** in `figures/training_curves.pdf`: train accuracy → 1.0 early, test accuracy snaps
  up later. If it has not grokked by 40k steps, rerun with `train.steps=60000`.
- The headline numbers land in `analysis/summary.json` (`grokking_step`,
  `topological_transition_step`, `lead_lag_steps`) and `figures/observables_over_time.pdf`.

Re-running a seed into an existing run dir is **refused** (mixed-run protection); pass
`overwrite=true` to replace it deliberately.

---

## 2. Cluster runs (for the sweeps)

> ⚠️ **Confirm the cluster specifics first.** The exact partition/queue names, account/project
> string, GPU type, and module names are **not yet pinned down** (see `Open_Questions.md` →
> *Confirmed Imperial GPU resources*). Everything below has clearly-marked `TODO` placeholders —
> get the real values from the RCS/CSG docs or your supervisor before submitting. There are two
> plausible Imperial targets and the repo ships templates for both schedulers:
>
> - **Imperial RCS central HPC** (e.g. the `cx3` cluster) — **PBS Pro** (`qsub`).
> - **DoC GPU cluster** (CSG-managed) — typically **SLURM** (`sbatch`).

### 2.1 One-time environment setup on the cluster

The project is `uv`-managed and the lockfile pins a **CUDA 12.1** torch build on Linux
(`pyproject.toml` → `[tool.uv.sources]`), so the environment is reproducible without conda.

```sh
# on a login node
git clone <your-private-remote>/grokking-tda.git   # or rsync the Code/ tree up
cd grokking-tda            # (the Code/-rooted repo)
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv isn't already available
module load cuda/12.1      # TODO: confirm the exact module name on this cluster
uv sync                    # resolves the Linux CUDA wheels from the pinned index
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Run `uv sync` on a node that can see a GPU (or where the CUDA toolkit module is loaded) so the
CUDA wheels resolve. Do the build **once**; jobs reuse `.venv`.

### 2.2 Submit a sweep — two-stage (GPU train → CPU analyse)

Keep training (GPU, expensive, once) separate from analysis (CPU, cheap, many times). Never burn
GPU time re-analysing.

**Option A — shipped array templates (simplest).** Submit from `Code/`:

```sh
# PBS (RCS central HPC) — 5 seeds as array indices 0–4
qsub -v EXPERIMENT=tf_mod97_grok src/grokking_tda/orchestration/templates/pbs_array.pbs

# SLURM (DoC cluster)
sbatch --export=EXPERIMENT=tf_mod97_grok src/grokking_tda/orchestration/templates/slurm_array.sbatch
```

Both templates set the seed from the array index and call
`uv run gtda-train +experiment=$EXPERIMENT seed=$SEED train.device=cuda`. **Before first use**,
edit the header for the real cluster: the `#PBS -l select=...:gpu_type=TODO` /
`#SBATCH --partition=TODO`, walltime, and any `module load` line. Then analyse on a CPU node:

```sh
bash src/grokking_tda/orchestration/templates/analyse_stage.sh results/raw
```

The analyse script pins OpenMP threads and sets `KMP_DUPLICATE_LIB_OK=TRUE` — needed because
ripser (libomp) and torch (libiomp) load two OpenMP runtimes that can otherwise deadlock a CPU
process. Keep parallelism at the **job-array** level, one thread per task.

**Option B — Hydra submitit launcher (SLURM only).** For sweeping config axes, not just seeds:

```sh
uv run gtda-train -m +experiment=tf_mod97_grok \
    seed=0,1,2,3,4 data.train_fraction=0.2,0.3,0.4 \
    hydra/launcher=imperial_slurm
```

Edit `src/grokking_tda/configs/hydra/launcher/imperial_slurm.yaml` first: set `partition`,
`array_parallelism`, and any module loads in `setup:`. This needs `hydra-submitit-launcher`
installed in the env.

### 2.3 Getting results back

Run dirs under `results/raw/` are self-describing (manifest + metrics + snapshots + figures).
Pull them to the dev box to analyse/plot interactively, or analyse on the cluster and pull only
the small `analysis/` + `figures/` outputs:

```sh
rsync -av <cluster>:~/grokking-tda/results/raw/  ./results/raw/
```

Then aggregate the sweep into one tidy table for the robustness/predictive figures:

```sh
uv run gtda-aggregate results/raw --out results/robustness.csv
```

---

## 3. Checklist before a cluster sweep

- [ ] Real partition/queue, account/project, GPU type, walltime filled into the template header.
- [ ] `module load` line for CUDA confirmed and matching the lockfile (CUDA 12.1).
- [ ] `uv sync` succeeds on a GPU-visible node; `torch.cuda.is_available()` is `True`.
- [ ] `train.device=cuda` (the templates already set this) — float64 CE is supported on CUDA.
- [ ] Disk budget sane: with `capture_representations=false`, runs are small; with it `true`,
      full-table snapshots are ~hundreds of MB–GB each — multiply by snapshots × seeds.
- [ ] Two-stage: GPU job trains, a separate CPU job/loop analyses. Don't analyse on the GPU node.
- [ ] Each run records a real `git_commit` (the repo is under git as of the 2026-06-18 baseline) —
      commit any code change before a sweep so results stay traceable.
