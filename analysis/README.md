# `analysis/`

Every number the thesis quotes in Chapters 4–7 is computed here, so each has a path from the
artefact store to the page.

`src/grokking_tda/` holds the method — observables, detectors, estimators — installed, imported and
unit tested; its own `analysis/` subpackage, which shares this name, computes observables from a
single snapshot. This directory holds the reduction: how a bank of runs becomes the tables the
chapters quote. It imports the method rather than restating it, so the two cannot drift. Method
code has tests; reduction code has a committed output, under `results/processed/thesis/`, whose
`README.md` maps each file to the module that wrote it.

```
bank.py              the run bank as one frame; the window rule; nulls; verdicts
cli.py               the --root/--out command line every driver shares
thesis_numbers.py    every quoted number, as a printed ledger and a claims file
figures/build.py     one tidy CSV per figure, plus a manifest of claims
figures/render.py    the manuscript figures, from those CSVs
figures/talk.py      the presentation figures: separate compositions at slide type size
figures/dial.py      TikZ ring coordinates, read from raw embeddings rather than the tables
figures/conceptual.py  a point set, its Rips complex and that barcode, as TikZ coordinates

normalisation.py     is the contraction a similarity, and does the verdict survive a
                     different denominator?                            (thesis 3.3.1, 4.4)
torus.py             degree-two homology of the joint-input cloud, by stage, depth and
                     projected dimension                               (thesis 4.6, 6.5)
representation.py    the signature on the embedding, the hidden state and the logits, at
                     matched cloud cardinality                         (thesis 7.1, RQ4)
circularity.py       two non-spectral circularity measures, the residual after
                     circularity, and a column-shuffle null            (thesis 4.5)
predictive.py        the head-to-head, extreme condition held out, target winsorised
                                                                       (thesis 5.4)
vectorise.py         the same head-to-head and decomposition against landscapes and
                     images rather than scalar summaries              (thesis 5.4, 5.5)
pid.py               decomposition per regime: two redundancy functions, two source
                     pairings, cluster bootstrap and null              (thesis 5.5)
redundancy.py        resampled-null p-values across the grid, BH and BY   (thesis 3.7, 4.4)
detector.py          detector recovery on a step of known location, and whether the
                     timing result survives the second detector        (thesis 3.6, 5.6)
velocity.py          changepoint on the topological velocity, with its controls (thesis 6.3)
phdim.py             window x projection sweep; the estimator on known dimensions (thesis 6.4)
instability.py       do the transient collapses reach the analysed checkpoints? (thesis 7.4)
windows.py           how much does the window rule decide the answer?  (thesis 4.2, A.2)
collapse.py          the ratio against dimensional collapse, and where the non-cyclic
                     task exceeds it                                   (thesis 4.5, 5.3)
shape.py             is there one dominant cycle? two scale-free statistics (thesis 3.3, 4.3)
```

## Running

Every module above except `bank.py` and `cli.py` is a CLI in the same shape,
`uv run python -m analysis.<name>`, writing to `results/processed/thesis/`.

```bash
cd Code
uv run python -m analysis.thesis_numbers
uv run python -m analysis.figures.build --only 4.2 4.6   # or all of them, with no --only
uv run python -m analysis.figures.render
```

Use the project interpreter — `uv run`, or `.venv/bin/python`. A system Python may be old enough to
reject `zip(..., strict=True)`, and may carry a different matplotlib, which renders a figure that
differs from the rest of the set without failing.

The figure commands write to `results/figures/generated/`, overridden by `--out` or
`GTDA_FIGURE_DIR`; where a manuscript tree sits beside the repository they write into it instead, so
the build stays one command.

## Assumptions

**Timing is passed through.** `t_top`, the signed lag and the PH-dimension transition are read from
each run's `analysis/summary.json`; the detector lives in `grokking_tda.evaluation.transitions`, and
a second implementation here would drift from it. What this directory derives is a function of the
observable series and `t_g`, and `t_g` is a threshold crossing on test accuracy, so nothing derived
depends on the transition detector.

The series passed through is the one the chapters quote, the scale-normalised H1 maximum
(`bank.HEADLINE_OBSERVABLE`). The summary's top-level `lead_lag_steps` belongs to the raw series.
Both are in the frame under names that say which: `t_top` and `t_top__raw`.

**Repeats are separated.** `is_replicate` keys on the tags in `identity.REPLICATE_TAGS`: the
twenty R12 dense runs and the twenty trajectory re-runs repeat existing conditions on a different
snapshot schedule, which moves every window median, and the forty-eight recipe cells of R14 and R18
sit outside the condition table by construction. All are excluded from the condition table, the null
band, the circularity association and the head-to-head; the trajectory ones are kept for Chapter 6.
**A re-run programme adds its own tag there before it launches**, or it silently pools into the
condition it repeats.

## Window rule (§4.2)

Comparing a persistence series across a transition needs a fixed rule for what counts as before and
after:

- **baseline** — the median over snapshots in `[0.5·t_g, 0.9·t_g]`
- **plateau** — the median from `1.2·t_g` to the end of the run
- **effect** — plateau ÷ baseline
- with no `t_g` there is no anchor, so baseline is `[0.3·end, 0.6·end]` and plateau `[0.8·end, end]`

A freshly initialised embedding carries high raw H1 persistence which decays over the first few
thousand steps, and a baseline reaching into that transient makes every raw ratio look like a fall;
hence the lower bound at `0.5·t_g`. Persistence keeps moving for a while after the accuracy
threshold is crossed; hence `1.2·t_g`. Both bounds are stated in the thesis and neither was tuned
after seeing results.

## Null bands

The nulls are the sixteen runs that fit their training data and never generalise: permuted labels,
and the polynomial `a³ + ab`, which the network memorises completely. Per observable, the null band
is the range its ratio takes across those sixteen. A condition is `above` if its whole bootstrap
interval over seeds clears the band, `below` if the interval falls short, `inside` otherwise.

Bands are per observable: the raw H1 maximum spans 0.54–1.54 and the normalised one 0.75–1.32,
nearly half as wide in log units. That difference is the argument of thesis §4.4.

## Fourier concentration

`fourier_concentration` is swept over `k ∈ {1, 2, 3, 5, 10, 20}` in the residue-axis and
discrete-log bases, and the family is kept whole: concentration rises monotonically in `k`, so the
member has to be chosen downstream, and the two downstream roles want different answers.

- **The competitor**, for the redundancy comparison. `select_fourier_k` takes the member whose time
  series best tracks test accuracy across grokking runs, and returns `k = 20`.
- **The circularity measure**, for §4.5. A circle is power in a few modes, so a large `k` measures
  something else; this uses `CIRCULARITY_K = 5`. `thesis_numbers.circularity_k_sensitivity` reports
  the association at every member, so the claim does not rest on the choice (ρ spans 0.67–0.75).

Under multiplication and division the grokked circle is ordered by discrete logarithm, so
`circularity_column` selects the group-reordered variant there; the residue-axis transform is blind
to that ordering and would rig the comparison in topology's favour.

## Caveats

**The skip guard is coarse.** `scripts/remote/analyse.sh` skips any run that already has an
`analysis/summary.json`, without checking whether that summary predates a code change. After
changing an observable or a detector, pass `FORCE=1` or delete the stale summaries. The PH-dimension
estimator is uncached and costs about three minutes per trajectory run.

**MMI gives the weaker source zero unique information by construction.** A `unique_a` of exactly
zero says topology is dominated by the Fourier baseline on that target, and says nothing about
whether it is informative. `analysis/pid.py` records both marginals and sets
`mmi_zero_by_construction` to keep the distinction visible.

**The Betti counter is calibrated on clean shapes.** A well-sampled synthetic circle's loop spans
0.84 of the cloud's diameter; a grokked 128-dimensional embedding's spans about 0.08. Counts against
real data are therefore conservative, and the meaningful comparison is null-relative
(`life_k_vs_init`, or `tda.significance.random_init_null`). It reads dominance as a *gap* in the
barcode, so a cloud whose above-floor bars are all signal has no gap to find: two comparable loops
and nothing beneath them count as zero, and one noise bar beneath them counts as two. The failure is
directional — the cleaner the cloud, the worse the answer — so it weakens a negative and cannot
manufacture a positive. F2 in `Documentation/Code_Audit_2026-08-31.md`.

## Figure builders

```python
@builder("4.7", "torus", "The joint representation is a torus upstream and a circle at the readout.")
def fig_torus(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    ...
    raise Missing("no maxdim=2 analysis")   # skips with a note; the rest of the set still builds
```

The claim string is what the figure has to establish, and it is written into the manifest, so
a reader can hold the figure to it.

Not every builder has a renderer. `4.7` builds `figures/fig-4-7-torus.csv` and nothing draws it:
the degree-two result is quoted in the prose of §4.6 and §6.5 rather than plotted, and the table is
kept because it is the evidence behind those sentences.
