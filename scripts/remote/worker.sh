#!/usr/bin/env bash
# One host's share of a run manifest, executed `jobs` at a time.
# Invoked on the remote machine by `gtda-remote launch`; not run by hand.
set -euo pipefail

slice=$1 jobs=$2 out_root=$3
mkdir -p "${out_root}/logs"

export PATH="${HOME}/.local/bin:${PATH}"
# Every run is a separate process on one GPU; let each use a single CPU thread so
# concurrent runs do not fight over the machine's cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run_one() {
    local overrides=$1 out_root=$2
    local tag
    tag=$(printf '%s' "${overrides}" | tr -cs 'A-Za-z0-9' '_')
    # Word splitting is intentional: each line is a list of Hydra overrides.
    # shellcheck disable=SC2086
    uv run gtda-train ${overrides} train.device=cuda "output_root=${out_root}/raw" \
        > "${out_root}/logs/${tag}.log" 2>&1 \
        || echo "FAILED ${overrides}" >> "${out_root}/logs/failures.txt"
}
export -f run_one

grep -vE '^[[:space:]]*(#|$)' "${slice}" \
    | xargs -P "${jobs}" -I LINE bash -c 'run_one "LINE" "$0"' "${out_root}"
