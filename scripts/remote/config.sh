#!/usr/bin/env bash
# Shared configuration for the remote-execution scripts. Override any value from
# the environment: REMOTE_USER=abc123 ./scripts/remote/gtda-remote probe

REMOTE_USER="${REMOTE_USER:-oc525}"
REMOTE_DOMAIN="${REMOTE_DOMAIN:-doc.ic.ac.uk}"
JUMP_HOST="${JUMP_HOST:-imperial}"           # a Host entry in ~/.ssh/config

# Everything lives on /vol/bitbucket, not in $HOME: home is quota-limited to a few
# gigabytes (a CUDA venv alone exceeds it), while bitbucket is multi-terabyte and
# NFS-mounted on every DoC machine. One shared checkout, one shared environment,
# one shared results tree — so no per-node bootstrap and no per-node fetch.
REMOTE_BASE="${REMOTE_BASE:-/vol/bitbucket/${REMOTE_USER}/grokking-tda}"
UV_CACHE="${UV_CACHE:-/vol/bitbucket/${REMOTE_USER}/.uv-cache}"

# Lab 210 GPU workstations. `probe` filters this down to what is actually idle.
GPU_POOL="${GPU_POOL:-$(printf 'gpu%02d ' $(seq 1 36))}"

JOBS_PER_HOST="${JOBS_PER_HOST:-3}"    # concurrent runs per GPU; the model is tiny
MIN_FREE_MB="${MIN_FREE_MB:-6000}"     # skip a GPU with less free memory than this
MAX_LOAD="${MAX_LOAD:-2.0}"            # skip a machine whose CPU is already busy
SSH_TIMEOUT="${SSH_TIMEOUT:-8}"
TMUX_SESSION="${TMUX_SESSION:-gtda}"

# Note: logged-in-user count is a poor idleness signal here — these machines carry
# dozens of detached SSH sessions — so free GPU memory and load average are used
# instead. Both together are what "nobody is working on this box" looks like.

# Reuse one TCP connection per host: launching across a dozen machines otherwise
# opens enough sessions through the shell server to trip its connection limits
# ("timed out during banner exchange").
MUX_OPTS=(
    -o ControlMaster=auto
    -o ControlPath="${HOME}/.ssh/cm-%r@%h:%p"
    -o ControlPersist=300
)

SSH_OPTS=(
    "${MUX_OPTS[@]}"
    -o BatchMode=yes
    -o ConnectTimeout="${SSH_TIMEOUT}"
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR
)
[[ -n "${JUMP_HOST}" ]] && SSH_OPTS+=(-J "${JUMP_HOST}")

# The shell server; reached by its ~/.ssh/config alias, without the -J hop.
SHELL_OPTS=(
    "${MUX_OPTS[@]}"
    -o BatchMode=yes
    -o ConnectTimeout="${SSH_TIMEOUT}"
    -o StrictHostKeyChecking=no
    -o LogLevel=ERROR
)

on_host() {  # on_host <gpu-host> <command...>
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${1}.${REMOTE_DOMAIN}" "${@:2}"
}

on_shell() { # on_shell <command...>
    ssh "${SHELL_OPTS[@]}" "${JUMP_HOST}" "$@"
}
