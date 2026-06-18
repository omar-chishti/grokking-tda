# ADR 0001 — Decouple training, artifacts, and analysis

**Status:** accepted · **Date:** 2026-06-01

## Context
Grokking runs are expensive (10⁴–10⁵ steps) and seed-sensitive, while the TDA
analysis is cheap but iterated heavily (many observables, point-cloud and PH
choices, baselines). Coupling them would mean re-training to try a new analysis.

## Decision
Three layers that communicate only through an on-disk artifact store:
training writes immutable, self-describing run artifacts (manifest + step-indexed
metrics + snapshots); analysis consumes them offline. Snapshots store authoritative
weights plus a cheap embedding point cloud; hidden states/logits are recomputed
from weights on demand.

## Consequences
- Train once on GPU; iterate analysis hundreds of times on CPU.
- Natural two-stage cluster workflow (train array → analyse stage).
- Full reproducibility: a run is interpretable from its manifest alone.
- Cost: a defined artifact schema must be maintained (`artifacts/`), and analysis
  may recompute representations (cheap for these model sizes).
