# ADR 0003 — Hydra structured configs + registries

**Status:** accepted · **Date:** 2026-06-01

## Context
The project is sweep-heavy (seeds × primes × train fractions × architectures ×
interventions). We need composable, validated configuration and first-class
multirun/cluster launching, while staying readable for a thesis reader.

## Decision
Typed dataclasses (`config/schema.py`) registered with Hydra's ConfigStore give
schema validation; Hydra composes group presets and `experiment/` overrides and
provides multirun + the submitit SLURM launcher. Component selection is via small
name-keyed **registries**, so configs stay declarative and new components are
additive. Considered plain dataclasses+OmegaConf (less automation) and raw JSON
(no composition/validation) — rejected for a sweep-heavy pipeline.

## Consequences
- `gtda-train +experiment=… a.b=… -m seed=0,1,2` covers single runs and sweeps.
- Typos in overrides fail fast against the schema.
- Cost: Hydra adds some "framework magic"; mitigated by keeping configs thin and
  documenting the composition model in `ARCHITECTURE.md`.
