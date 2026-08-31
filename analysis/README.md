# `analysis/` — the thesis provenance layer

Every number quoted in Chapters 4–7 of the thesis is computed here, so that each has a path from
the artefact store to the page. If a figure or a table in the manuscript disagrees with the output
of this directory, the manuscript is wrong.

Not to be confused with `src/grokking_tda/analysis/`, which is the **library** layer that computes
observables from a snapshot. This directory sits on top of the artefact store and answers questions
about the *bank* — many runs at once.

```
analysis/
  bank.py              the run bank as one frame; the window rule; nulls; verdicts
  cli.py               the --root/--out command line every driver below shares
  thesis_numbers.py    CLI: recompute every quoted number, print a ledger, write CSV + JSON
  figures/build.py     CLI: one tidy CSV per thesis figure, plus a manifest of claims
  figures/render.py    CLI: the manuscript figures, from those CSVs
  figures/talk.py      CLI: the presentation figures, from the same CSVs -- separate
                       compositions, one claim each, at slide type size

  normalisation.py     is the contraction a similarity, and does the verdict survive a
                       different denominator?                            (thesis 3.3.1, 4.4)
  torus.py             degree-two homology of the joint-input cloud, by stage, depth and
                       projected dimension                               (thesis 4.6, 6.5)
  representation.py    the same signature on the embedding, the hidden state and the
                       logits, at matched cloud cardinality              (thesis 7.1, RQ4)
  circularity.py       two non-spectral circularity measures, the residual after circularity,
                       a column-shuffle null, and the recipe's share of
                       the variance in it                                (thesis 4.5)
  predictive.py        the head-to-head with the extreme condition held out and the target
                       winsorised                                        (thesis 5.4)
  pid.py               information decomposition per regime, two redundancy functions, two
                       source pairings, cluster bootstrap and null       (thesis 5.5)
  redundancy.py        resampled-null p-values across the grid, drawn one null configuration
                       at a time, BH and BY side by side                 (thesis 3.7, 4.4)
  detector.py          what the detector does to a step of known location, what its
                       unrepaired form did, and whether the timing result survives the
                       second detector                                   (thesis 3.6, 5.6)
  velocity.py          changepoint on the topological velocity, with its controls
                                                                         (thesis 6.3)
  phdim.py             window x projection sweep, and the estimator on known dimensions
                                                                         (thesis 6.4)
  instability.py       do the transient collapses reach the analysed checkpoints?
                                                                         (thesis 7.4)
  windows.py           how much does the window rule decide the answer?  (thesis 4.2, A.2)
  collapse.py          the ratio as a function of dimensional collapse, and where the
                       non-cyclic task exceeds it                        (thesis 4.5, 5.3)
  shape.py             is there one dominant cycle? two scale-free statistics
                                                                         (thesis 3.3, 4.3)
```

Each module past `figures/build.py` is a CLI in the same shape,
`uv run python -m analysis.<name>`, writing to `results/processed/thesis/`.

## Running it

```bash
cd Code
uv run python -m analysis.thesis_numbers        # -> results/processed/thesis/
uv run python -m analysis.figures.build         # -> results/processed/thesis/figures/
uv run python -m analysis.figures.build --only 4.2 4.6
uv run python -m analysis.figures.render        # -> results/figures/generated/*.pdf
uv run python -m analysis.figures.talk          # -> ../LaTeX/Presentation/figures/generated/
uv run python -m analysis.figures.conceptual    # -> .../tikz-methodology-data.tex
uv run python -m analysis.figures.dial          # -> .../tikz-dial-data.tex
```

Both figure commands write to `results/figures/generated/` unless `--out` (or
`GTDA_FIGURE_DIR`) says otherwise; when a manuscript tree sits beside the repository they
default to its `figures/generated/` instead, so the build stays one command.

**Use the project interpreter.** `uv run` or `.venv/bin/python`, never a bare `python`: the system
one may be old enough to reject `zip(..., strict=True)` and, worse, may carry a different matplotlib
and render a figure that differs from the rest of the set without failing.

Every figure is drawn to one set of rules. A shared vocabulary of devices — seed comb, null band,
sparkline column, key block, named value, seed tally, condition column block — so that a reader who
learns one plate can read the next; and a precision standard beneath it: axes joined at one origin,
leaders that reach their target and do not cross, panel-letter offsets that belong to the column
rather than the panel. `style.py` implements all of them, and `panel_letter`'s docstring carries the
column rule where it is used.

`talk` draws the **presentation** figures. They are not the manuscript figures rescaled: a 158 mm
plate at 8 pt would have to be set 217 mm across for its labels to read at slide body size, which
is wider than a 160 mm page, so each is a separate composition carrying one claim at the exact
width the frame gives it (148 mm, or 152 mm for the two forests). It imports `render` rather than
restating it -- `_load`, `_claims`, `_condition_row`, `_baseline_ratio`, `_seed_tally`,
`_condition_columns` -- so the two sets cannot drift, and it owns nothing but composition. It is
held to the same drawing rules, at slide type size.

`dial` reads three terminal embeddings straight from `results/raw/`, takes the top-two principal
plane of each, and emits the ring coordinates, the winding, the mean radius and the residues at the
quarter turns. It is the only figure module that touches the raw snapshots rather than the tidy
tables, because what it draws is the embedding itself and not a summary of it.

`conceptual` is the odd one out: it computes a point set, its Vietoris--Rips complex and that
complex's barcode, and emits them as TikZ coordinates for the Chapter 2 teaching figure. Drawing
those by hand invites them to disagree --- an octagon of edges beside four bars of invented length
says nothing true about either --- so the composition is authored in the `.tex` file and the
mathematics is computed here. It asserts what the figure claims (that the ring closes at the
smaller radius, that the cycle is alive there and dead at the larger one) rather than hoping.

Requires Python 3.10+ (`zip(strict=)`), which the project environment supplies; the system
interpreter may not.

### Outputs

| File | Contents |
|---|---|
| `results/processed/thesis/bank.csv` | One row per run: config, passed-through timing, window ratios, terminal circularity, scale collapse. |
| `results/processed/thesis/conditions.csv` | One row per experimental condition: seeds grokked, per-seed `t_g`, ratio medians with bootstrap intervals over seeds, and an above/inside/below verdict against each observable's null band. |
| `results/processed/thesis/circularity.csv` | The 71 grokking runs behind the association in thesis §4.5. |
| `results/processed/thesis/claims.json` | The ledger: every headline number with the inputs it was computed from. |
| `results/processed/thesis/figures/fig-*.csv` | Tidy data for one figure each. |
| `results/processed/thesis/figures/manifest.json` | Figure number, name, the claim it must carry, row count. |

## The two rules that keep this honest

**Timing is passed through, never recomputed.** `t_top`, the signed lag and the PH-dimension
transition are read from each run's `analysis/summary.json`. The detector lives in
`grokking_tda.evaluation.transitions`; a second implementation here would silently drift from it.
Everything this directory *derives* is a function of the observable series and `t_g`, and `t_g` is a
threshold crossing on test accuracy, so no derived quantity depends on the transition detector.

Passed through from **the observable the chapters quote**, which is the scale-normalised `H1`
maximum (`bank.HEADLINE_OBSERVABLE`) — not the summary's top-level `lead_lag_steps`, which belongs
to the *raw* series. Both are in the frame, named for what they are: `t_top` / `lead_lag_steps`
against `t_top__raw` / `lead_lag_steps__raw`.

**Dense re-runs are flagged and separated.** The twenty R12 runs repeat existing conditions on a
different snapshot schedule, which changes every window median. They are excluded from the condition
table and from the null band — pooling them with the main programme would double-count — and
included for the trajectory analyses in Chapter 6.

## The window rule (thesis §4.2)

Comparing a persistence series before and after a transition needs a rule for what counts as before
and after, fixed once for every run:

- **baseline** — median of the observable over snapshots in `[0.5·t_g, 0.9·t_g]`
- **plateau** — median from `1.2·t_g` to the end of the run
- **effect** — plateau ÷ baseline
- a run with no `t_g` has no anchor, so baseline is `[0.3·end, 0.6·end]` and plateau `[0.8·end, end]`

The lower bound at `0.5·t_g` exists because a freshly initialised embedding carries very high *raw*
`H1` persistence which decays over the first few thousand steps; a baseline including that transient
makes every raw ratio look like a fall. The `1.2·t_g` plateau bound exists because persistence keeps
moving for a while after the accuracy threshold is crossed. Both are stated in the thesis and neither
was tuned after seeing results.

## Null bands and verdicts

The null models are the runs that fit their training data and never generalise: permuted labels, and
the polynomial `a³ + ab`, which the network memorises completely. Sixteen runs. For each observable,
the **null band** is the range its ratio takes across those sixteen, and a condition is marked

- `above` if its whole bootstrap interval over seeds exceeds the band,
- `below` if the whole interval falls short of it,
- `inside` otherwise.

Bands are computed per observable, which matters: the raw `H1` max band is 0.54–1.54 while the
scale-normalised one is 0.75–1.32, nearly half as wide in log units, and that difference is the
argument of thesis §4.4.

## Two roles for the Fourier family, two values of *k*

`fourier_concentration` is swept over `k ∈ {1, 2, 3, 5, 10, 20}` in both the residue-axis and the
discrete-log basis. The family is recorded and not reduced, because concentration rises monotonically
in `k`, so the strongest member has to be chosen downstream. There are two downstream roles and they
want different answers:

- **The competitor**, for the redundancy comparison. `select_fourier_k` picks the member whose time
  series best tracks test accuracy across grokking runs. It returns **k = 20**.
- **The circularity measure**, for the association in §4.5. A circle is power concentrated in a
  *few* modes, so a large `k` measures something else; this uses **`CIRCULARITY_K = 5`**, and
  `thesis_numbers.circularity_k_sensitivity` reports the association at every member so the claim
  does not rest on the choice (ρ ranges 0.63–0.72).

For multiplication and division the grokked circle is ordered by discrete logarithm, so
`circularity_column` selects the group-reordered variant there — otherwise the comparison would be
rigged in topology's favour.

## Where each number comes from

The layer is split in two, and the split is the rule: **`src/grokking_tda/` holds anything that is
part of the method** — observables, detectors, estimators — installed, imported and unit tested;
**`analysis/` holds the thesis-specific reduction**, how the run bank becomes the tables the
chapters quote. Method code has tests; reduction code has a committed output.

| Quantity as the thesis reports it | Produced by | Written to |
|---|---|---|
| Grokking step `t_g`, per run | `evaluation.transitions.grokking_step` | `summary.json` |
| `t_g` at 0.8 / 0.9 / 0.95, and midpoint-of-rise on both accuracy series | `evaluation.transitions.grokking_step_sensitivity` | `summary.json` |
| Topological transition `t_top`, signed lag | `evaluation.transitions.transition_step`, `lead_lag` | `summary.json` |
| Second detector `t_changepoint`, its signed lag | `evaluation.changepoint.changepoint_step` | `summary.json` |
| Per-observable series | `analysis.observable.run_observables` | `observables.csv` |
| Scale-normalised persistence, connectivity scale | `tda.observables` | `observables.csv` |
| Fourier concentration, both bases, `k ∈ {1,2,3,5,10,20}` | `baselines.fourier` | `observables.csv` |
| PH-dimension of the weight trajectory | `tda.phdim`, wired in `cli.analyse` | `ph_dimension.csv`, `summary.json` |
| Divergence flag | `cli.analyse` | `summary.json` |
| Ratio of each observable across the transition | `analysis.bank.summarise` | `thesis/bank.csv` |
| Condition table with bootstrap intervals over seeds | `analysis.bank.condition_table` | `thesis/conditions.csv` |
| Null band, verdicts against it | `analysis.bank.null_band`, `verdicts` | `thesis/conditions.csv` |
| Circularity association (Spearman, Pearson) | `analysis.thesis_numbers.circularity_association` | `thesis/claims.json` |
| Sensitivity of that association to `k` | `analysis.thesis_numbers.circularity_k_sensitivity` | `thesis/claims.json` |
| Per-condition p-values vs the pooled nulls | `analysis.redundancy.permutation_pvalues` | `thesis/significance.csv` |
| Benjamini–Hochberg correction across the grid | `evaluation.multiplicity.benjamini_hochberg` | `thesis/significance.csv` |
| PID atoms, and the marginals behind them | `evaluation.pid.gaussian_pid`, `analysis.pid` | `thesis/pid.json` |
| Predictive head-to-head, AUC / R² per feature set | `evaluation.headtohead`, `analysis.predictive` | `thesis/head_to_head.csv` |
| Betti profile, torus test | `tda.betti`, `analysis.torus` | `thesis/torus.csv` |
| The signature by representation space | `analysis.representation` | `thesis/representation.csv` |
| Timing under the changepoint detector | `evaluation.changepoint`, `analysis.detector` | `thesis/detector_agreement.csv` |
| Betti profile across depth and stage | `analysis.figures.build` (6.4) | `thesis/figures/fig-6-4-depth.csv` |
| Window-rule sensitivity | `analysis.windows` | `thesis/window_sensitivity.csv` |
| The collapse law, and excess over it by condition | `analysis.collapse` | `thesis/collapse_conditions.csv` |
| Dominance and share of the leading bar | `analysis.shape` | `thesis/diagram_shape_conditions.csv` |

## Things that will bite

**Re-analysis is not free, and the skip guard is coarse.** `scripts/remote/analyse.sh` skips a run
that already has `analysis/summary.json`; it does not check whether that summary predates a code
change. After changing an observable or a detector, pass `FORCE=1` or delete the stale summaries
first. The PH-dimension estimator is not cached and costs about three minutes per trajectory run.

**MMI assigns zero unique information to the weaker source by construction.** A `unique_a` of
exactly zero means topology is dominated by the Fourier baseline on that target, not that it is
uninformative. `analysis/pid.py` records both marginals and sets `mmi_zero_by_construction` so the
distinction stays visible.

**The Betti counter is calibrated on clean shapes, not network clouds.** On a well-sampled
synthetic circle the loop spans 0.84 of the cloud's diameter; on a grokked 128-dimensional
embedding it spans about 0.08. Against real data the counts are conservative, and the meaningful
comparison is null-relative (`life_k_vs_init`, or `tda.significance.random_init_null`).

## Adding a figure builder

```python
@builder("4.7", "torus", "The joint representation is a torus upstream and a circle at the readout.")
def fig_torus(root: Path, bank: pd.DataFrame) -> pd.DataFrame:
    ...
    raise Missing("no maxdim=2 analysis")   # skips with a note instead of failing
```

A builder whose inputs do not exist yet raises `Missing`, so the whole set can be rebuilt at any
time and fills in as analyses land. The claim string is not decoration — it is the one thing the
figure has to make undeniable, and it is written into the manifest so a reader can hold the
figure to it.
