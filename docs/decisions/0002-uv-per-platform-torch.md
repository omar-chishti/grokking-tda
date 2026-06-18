# ADR 0002 — uv for environments, with per-platform PyTorch pinning

**Status:** accepted · **Date:** 2026-06-01

## Context
We need one reproducible dependency definition that works on an Intel-Mac dev box
(MPS/CPU) and a Linux+CUDA cluster, and that leads cleanly to a Docker image.
PyTorch dropped x86-macOS wheels after 2.2.2, and CUDA vs CPU/MPS wheels differ.

## Decision
Use **uv** with a single `uv.lock`. Pin torch per platform via environment markers
(`torch==2.2.2; sys_platform=='darwin'`, `torch>=2.4; sys_platform!='darwin'`) and
point Linux at the `pytorch-cu121` index via `[tool.uv.sources]`. macOS falls
through to PyPI (whose wheel carries MPS). Considered Poetry — rejected because
varying CPU/CUDA wheels per platform from one lock is awkward and resolves are slow
in ephemeral cluster/Docker envs.

## Consequences
- `uv sync` works identically on laptop and cluster from the same lockfile.
- `uv sync --frozen` gives deterministic, cache-friendly Docker builds.
- Standard PEP-621 `pyproject.toml`, so no lock-in if we later change tools.
