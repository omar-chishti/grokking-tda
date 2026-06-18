"""Cluster orchestration.

Two routes, because the Imperial scheduler is not yet confirmed:

1. **Hydra + submitit (SLURM)** — the integrated path. Multirun sweeps launch as a
   SLURM array directly from the CLI:
   ``gtda-train -m +experiment=tf_mod97_grok seed=0,1,2 hydra/launcher=imperial_slurm``.

2. **Plain job-array templates** (``templates/``) — scheduler-agnostic fallback for
   SLURM *or* PBS Pro (Imperial's central HPC uses PBS). Edit the header, set the
   sweep variables, submit with ``sbatch`` / ``qsub``.

Keep expensive training on the cluster and run the cheap analysis/plotting stage
separately (CPU), reading the artifact store — never recompute training to re-analyse.
"""
