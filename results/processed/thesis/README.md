# Processed tables

One tidy table per analysis, written by `analysis/`. Chapters 4–7 quote these and nothing else, so
a claim in the manuscript can be checked here without the raw artefact store.

| File | Written by |
|---|---|
| `bank.csv`, `conditions.csv`, `claims.json` | `analysis.thesis_numbers` |
| `figures/*.csv`, `figures/manifest.json` | `analysis.figures.build` — one per thesis figure |
| `significance.csv` | `analysis.redundancy` |
| `normalisation.csv`, `normaliser_verdicts.csv` | `analysis.normalisation` |
| `window_sensitivity.csv` | `analysis.windows` |
| `diagram_shape*.csv` | `analysis.shape` |
| `circularity_measures.*`, `column_shuffle_null.csv` | `analysis.circularity` |
| `collapse_*.csv` | `analysis.collapse` |
| `head_to_head.*` | `analysis.predictive` |
| `pid.json` | `analysis.pid` |
| `vector_*.csv`, `vectorised.json` | `analysis.vectorise` |
| `detector_calibration.*`, `detector_agreement.csv` | `analysis.detector` |
| `mde_by_seed_count.csv` | `analysis.thesis_numbers` |
| `velocity_changepoint.*` | `analysis.velocity` |
| `phdim_*.csv/.json` | `analysis.phdim` |
| `torus.csv` | `analysis.torus` |
| `representation.csv`, `representation_conditions.csv` | `analysis.representation` |
| `instability.csv` | `analysis.instability` |

`results/raw/` holds the run artefacts these are derived from. It is gigabytes and is not tracked;
the trajectory files for twenty of the runs behind `figures/fig-6-3-phdim.csv` no longer exist
locally, so that figure's table is the record of the measurement.
