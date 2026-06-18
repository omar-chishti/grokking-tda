#!/bin/bash
# Stage 2 (cheap, CPU): analyse + plot every run under a results directory.
# Decoupled from training on purpose — re-run freely without retraining.
#   bash src/grokking_tda/orchestration/templates/analyse_stage.sh results/raw
set -euo pipefail

# Analysis imports both ripser/scikit-learn (libomp) and torch/MKL (libiomp); the dual
# OpenMP runtime can deadlock or crash a CPU process. Pinning threads and allowing the
# duplicate runtime is the standard, safe mitigation for batch analysis (parallelism comes
# from the job array, not threads), and it makes CPU numerics reproducible.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

ROOT="${1:-results/raw}"
for manifest in "${ROOT}"/*/manifest.json; do
    run_dir="$(dirname "${manifest}")"
    echo "=== analysing ${run_dir} ==="
    uv run gtda-analyse "${run_dir}"
    uv run gtda-plot "${run_dir}"
done
