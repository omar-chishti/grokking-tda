# Running on the DoC GPU workstations

The Department of Computing has no batch scheduler for its GPU machines. Lab 210 holds
`gpu01`–`gpu36` — ordinary Linux desktops with Nvidia cards (Titan XP, GTX 1080, Quadro P4000,
RTX 2080 Ti) — reached over SSH through a departmental shell server. There is no queue, no
allocation and no fair-share: you pick a machine, check nobody is using it, and run.

`scripts/remote/gtda-remote` is the small scheduler that fills that gap. It probes the pool for
idle GPUs, splits a run manifest across them, runs each host's share inside `tmux`, and pulls the
artifacts back.

## Why not the SLURM templates

`src/grokking_tda/orchestration/templates/` contains SLURM and PBS job-array scripts and
`configs/hydra/launcher/imperial_slurm.yaml` configures a submitit launcher. These target Imperial's
**central** HPC (PBS Pro), not the DoC lab machines, and were never validated against a real
scheduler. They are kept for the central-HPC path; everything below is the route that matches how
DoC access actually works.

## One-time setup

Access is public-key only — the shell servers offer `publickey` and `gssapi-with-mic`, and refuse
passwords. Confirm the key works before anything else:

```bash
ssh imperial 'hostname; quota -s | tail -2'
```

`imperial` is a `Host` entry in `~/.ssh/config` pointing at `shell1.doc.ic.ac.uk`. If this returns
`Permission denied (publickey)`, the public key is no longer in `~/.ssh/authorized_keys` on the DoC
side; see *Restoring access* below. Nothing else in this document works until it succeeds.

Then push the repo and build the environment:

```bash
./scripts/remote/gtda-remote sync
```

DoC home directories are NFS-mounted on every lab machine, so the repo and its `.venv` are built
**once** and every GPU host sees them. `sync` installs `uv` into `~/.local/bin` if it is missing and
runs `uv sync --frozen`, which resolves the CUDA 12.1 torch wheels pinned for Linux in
`pyproject.toml`.

## Running a sweep

```bash
./scripts/remote/gtda-remote probe                    # which GPUs are free right now
./scripts/remote/gtda-remote launch experiments/R1.runs
./scripts/remote/gtda-remote status                   # tmux state, run counts, latest metrics
./scripts/remote/gtda-remote fetch                    # pull artifacts into results/raw/
```

A manifest is one line of Hydra overrides per run, `#` for comments — see `experiments/*.runs`.
`launch` distributes lines round-robin across available hosts, so no single machine inherits the
expensive tail of an ordered manifest, and runs `JOBS_PER_HOST` of them concurrently. The model is
small enough that three concurrent runs share one GPU comfortably.

Results are written to node-local `/tmp`, never to NFS home: home quotas are a few gigabytes and
the full programme produces well over ten. `fetch` rsyncs each host's `raw/` into the local
`results/raw/`, after which analysis runs on the laptop exactly as it does for a local run.

## Etiquette

These are shared desktops that students sit at.

- `probe` already skips any machine with a logged-in user or less than `MIN_FREE_MB` of free GPU
  memory. Do not widen those filters to grab a busy machine.
- Keep `JOBS_PER_HOST` modest; the point is to use idle capacity, not to saturate a workstation.
- `stop` and `clean` when a sweep finishes. Leaving `tmux` sessions and `/tmp` trees behind on
  thirty machines is how access gets revoked.
- CSG sometimes reserves specific `gpu` machines for individual projects. If a host refuses
  connections, drop it from `GPU_POOL` rather than retrying.

## Configuration

Every value in `scripts/remote/config.sh` can be overridden from the environment:

```bash
GPU_POOL="gpu05 gpu06 gpu07" JOBS_PER_HOST=2 ./scripts/remote/gtda-remote launch experiments/R4.runs
```

| Variable | Default | Meaning |
|---|---|---|
| `REMOTE_USER` | `oc525` | DoC username |
| `JUMP_HOST` | `imperial` | `~/.ssh/config` entry for the shell server |
| `GPU_POOL` | `gpu01`–`gpu36` | candidate hosts |
| `JOBS_PER_HOST` | `3` | concurrent runs per GPU |
| `MIN_FREE_MB` | `3000` | skip a GPU with less free memory |
| `RUN_ROOT` | `/tmp/$USER/grokking-tda` | node-local scratch |

## Expected throughput

The model is tiny — a 1-layer transformer, ~222k parameters, full-batch over 2,822 examples — so
GPU time per step is dominated by kernel-launch overhead rather than arithmetic. On the laptop CPU
a 40,000-step run takes about 56 minutes and the machine saturates at roughly 17 optimiser steps
per second in aggregate. On a single lab GPU the same run is expected to take a few minutes, and
the whole ~85-run programme to finish in well under an hour of wall time across a handful of hosts.

Verify this on the first sweep rather than trusting it: `status` reports steps per second from each
run's `events.jsonl`.

## Restoring access

If `ssh imperial` returns `Permission denied (publickey)`:

1. Log in to the DoC self-service pages with your college credentials and re-upload
   `~/.ssh/id_rsa.pub`, or
2. Obtain a Kerberos ticket and use GSSAPI, which the servers still offer:
   ```bash
   /usr/bin/kinit oc525@IC.AC.UK
   ssh -K imperial
   ```
   then append the public key to `~/.ssh/authorized_keys` on the remote side, or
3. Email CSG (`doc-help@imperial.ac.uk`) with your username and public key.

The laptop remains a complete fallback throughout — `gtda-train` is identical, only slower — so a
lost afternoon of access delays the sweeps rather than blocking the thesis.
