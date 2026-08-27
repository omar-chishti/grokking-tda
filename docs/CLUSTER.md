# Running on the DoC GPU workstations

> Site-specific. These scripts drive a pool of shared lab workstations at Imperial's
> Department of Computing over SSH. Everything is parameterised through
> `scripts/remote/config.sh`, so the shape transfers to any similar pool, but the
> defaults (domain, jump host, `/vol/bitbucket`) do not. For a scheduler instead of a
> pool, use the SLURM/PBS templates in `src/grokking_tda/orchestration/templates/`.

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

Access is public-key only — the shell servers offer `publickey` and `gssapi-with-mic`, and password
authentication was disabled department-wide in December 2024.

The key is passphrase-protected, so **load it into `ssh-agent` before anything else**. Every script
here runs non-interactively, and a merely *locked* key fails exactly like a rejected one:
`Permission denied (publickey)`. Diagnose with `ssh-add -l` before assuming anything is wrong
server-side.

```bash
ssh-add --apple-use-keychain ~/.ssh/id_rsa   # once; persists via Keychain
ssh-add -l                                   # should list the key
ssh imperial 'hostname; quota -s | tail -2'  # should now succeed
```

Add this to the top of `~/.ssh/config` so the agent loads the key automatically after a reboot:

```
Host *
    AddKeysToAgent yes
    UseKeychain yes
    IdentityFile ~/.ssh/id_rsa
```

`imperial` is a `Host` entry pointing at `shell1.doc.ic.ac.uk`. Then push the repo and build the
environment:

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
| `REMOTE_USER` | *(required)* | your login on the GPU pool |
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

## If access genuinely breaks

Work through these in order; the first is by far the most likely.

1. **The key is not loaded.** `ssh-add -l` reports `The agent has no identities`. Re-run
   `ssh-add --apple-use-keychain ~/.ssh/id_rsa`. This accounts for most apparent failures, because
   `BatchMode=yes` and any non-interactive caller turn a locked key into
   `Permission denied (publickey)`.
2. **Home directory permissions.** A group-writable home makes `sshd` ignore `authorized_keys`
   silently. From a working session: `chmod go-w ~ && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys`.
3. **The key really is absent server-side.** The servers still accept Kerberos, so you can get in
   without it: `/usr/bin/kinit $REMOTE_USER@IC.AC.UK` (the macOS binary — Anaconda's `kinit` shadows it on
   `PATH` and needs a config file that is not present), then
   `ssh -o PreferredAuthentications=gssapi-with-mic $REMOTE_USER@shell1.doc.ic.ac.uk`, and re-add the key
   with `ssh-copy-id`.
4. **CSG's own procedure**, which requires being physically at a DoC lab machine in Huxley:
   `~dcw/bin/setup-ssh --changereal` generates an ed25519 key, appends it to `authorized_keys`, and
   writes a `~/.ssh/HomeConfig` to copy to the laptop. Otherwise email `doc-help@imperial.ac.uk`.

The laptop remains a complete fallback throughout: `gtda-train` is identical there, only slower.
