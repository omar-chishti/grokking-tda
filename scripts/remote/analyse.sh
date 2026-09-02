#!/usr/bin/env bash
# Analyse and plot every run under a results root, `jobs` at a time.
# Runs where the artifacts already live, so only derived data crosses the network.
set -euo pipefail

root=$1 jobs=${2:-3}
export PATH="${HOME}/.local/bin:${PATH}"
# ripser/sklearn (libomp) and torch (libiomp) both load OpenMP; pin threads and
# permit the duplicate runtime, or a CPU analysis process can deadlock.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

# Analysis is re-run as new runs land, so completed runs are skipped; FORCE=1
# redoes everything, which is what a change to an observable requires.
# A run still training has no summary, so the skip guard would let it through, and the
# partial summary it produced would then *be* the guard for every later pass. Analysis is
# re-run as runs land, so this is the normal case, not an edge one.
finished() {
    local run=$1 want last
    want=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['config']['train']['steps'])" \
        "${run}/manifest.json" 2>/dev/null) || return 1
    last=$(tail -1 "${run}/metrics.jsonl" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['step'])" 2>/dev/null) || return 1
    [[ -n "${want}" && -n "${last}" && "${last}" -ge "${want}" ]]
}

analyse_one() {
    local run=$1
    [[ -z "${FORCE:-}" && -f "${run}/analysis/summary.json" ]] && return 0
    if ! finished "${run}"; then
        echo "skipping ${run}: still training" >&2
        return 0
    fi
    uv run gtda-analyse "${run}" > "${run}/analysis.log" 2>&1 \
        && uv run gtda-plot "${run}" >> "${run}/analysis.log" 2>&1 \
        || echo "FAILED ${run}" >> "$(dirname "${run}")/../logs/analysis_failures.txt"
}
export -f analyse_one finished
export FORCE

find "${root}" -mindepth 1 -maxdepth 1 -type d -print0 \
    | xargs -0 -P "${jobs}" -I RUN bash -c 'analyse_one "RUN"'
