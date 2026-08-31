# ADR 0004 — A single Observable abstraction for topology and baselines

**Status:** accepted · **Date:** 2026-06-01

## Context
The thesis's central question is not "does topology move at grokking?" but "what
does topology add over cheaper diagnostics (Fourier, weight norm, LID), and does it
lead in time?" That demands comparing topological and non-topological quantities on
equal footing.

## Decision
Define an observable as `(ObservationContext) -> float` and register all of them —
H1/H0 persistence summaries *and* Fourier/weight-norm/LID baselines — in one
registry. A per-snapshot `ObservationContext` lazily builds and caches the shared
expensive artifacts (point cloud, persistence diagrams) so many observables pay
that cost once.

## Consequences
- Adding or swapping a diagnostic is a one-line `AnalysisCfg.observables` change.
- Topology is never reported in isolation; the baseline comparison is structural.
- `evaluation/` consumes the resulting table uniformly to compute lead/lag and
  predictive features.
