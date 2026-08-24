#!/usr/bin/env bash
# Shared configuration for the remote-execution scripts. Override any value from
# the environment: REMOTE_USER=abc123 ./scripts/remote/gtda-remote probe

REMOTE_USER="${REMOTE_USER:-oc525}"
REMOTE_DOMAIN="${REMOTE_DOMAIN:-doc.ic.ac.uk}"
JUMP_HOST="${JUMP_HOST:-imperial}"           # a Host entry in ~/.ssh/config

# Home is NFS-mounted on every lab machine, so the repo is synced once and every
# host sees it. Results are written to node-local disk (no NFS quota, no
# contention) and pulled back with `fetch`.
REPO_DIR="${REPO_DIR:-grokking-tda}"                       # relative to remote $HOME
RUN_ROOT="${RUN_ROOT:-/tmp/${REMOTE_USER}/grokking-tda}"   # node-local results + logs

# Lab 210 GPU workstations (gpu01-gpu36). Trim to a known-good subset if CSG has
# reserved machines; `probe` filters out whatever is busy or unreachable.
GPU_POOL="${GPU_POOL:-$(printf 'gpu%02d ' $(seq 1 36))}"

JOBS_PER_HOST="${JOBS_PER_HOST:-3}"    # concurrent runs per GPU; the model is tiny
MIN_FREE_MB="${MIN_FREE_MB:-3000}"     # skip a GPU with less free memory than this
SSH_TIMEOUT="${SSH_TIMEOUT:-8}"
TMUX_SESSION="${TMUX_SESSION:-gtda}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout="${SSH_TIMEOUT}" -o StrictHostKeyChecking=no)
[[ -n "${JUMP_HOST}" ]] && SSH_OPTS+=(-J "${JUMP_HOST}")

on_host() {  # on_host <host> <command...>
    local host=$1; shift
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${host}.${REMOTE_DOMAIN}" "$@"
}
