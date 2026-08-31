# ADR 0005 — Report persistence relative to the cloud's own scale

**Status:** accepted · **Date:** 2026-08-24

## Context
Persistence is measured in the units of the point cloud, and under weight decay the
embedding's overall scale drifts by orders of magnitude across training. A rise in raw
H1 persistence is therefore ambiguous: the cloud may have changed shape, or it may
have grown. The reference literature reports the raw quantity, so the ambiguity
is inherited rather than introduced — and it sits directly under the thesis's central
claim, that topology carries something the cheap diagnostics do not.

## Decision
Register every persistence summary twice: raw, as the reference paper reports it, and
divided by the cloud's **connectivity scale** — the largest finite H0 death, the
filtration value at which the cloud becomes one component. That scale costs nothing
(the H0 diagram is already computed) and is exactly linear in the cloud's size, which
is the one property a denominator needs here. `pointcloud_scale` is registered as an
observable in its own right, so the size of the confound is visible rather than
assumed away.

## Consequences
- Every downstream comparison can be run under both, and the difference is reportable
  rather than hidden: the null band on the raw H1 maximum is nearly twice as wide in
  log units as the normalised one.
- The normalised summary, not the raw one, is what the analysis layer treats as
  headline (`analysis.bank.HEADLINE_OBSERVABLE`); the raw series is kept alongside it
  under an explicit `__raw` suffix so nothing is silently substituted.
- Three alternative denominators (mean pairwise distance, diameter, root-mean-square
  radius) are swept in `analysis/normalisation.py`, because a normaliser that changes
  the verdict is a result about the normaliser, not about topology.
