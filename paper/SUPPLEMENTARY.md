# Supplementary material

This document accompanies the manuscript *Non-monotonic source reversion across
language-model pretraining checkpoints: behavioral trajectories and
source-position residual interventions*. It contains the supplementary
figures, the family-by-template semantic details, and the fact-cluster
sensitivity analysis. Checkpoint rows are not independent statistical
replicates. Machine-readable values and provenance are retained in the
repository artifacts named below.

For semantic summaries, `analyze_semantic.py` clusters rows by the frozen
`axis:category:subject:Y` key carried by each runner row and averages repeated
rendered rows before its 4,000-draw bootstrap. The two deterministic key halves
are robustness partitions used together in the gate, not independent held-out
test sets.

The semantic preflight is a tokenizer/schema smoke check, not a model-based
capability filter. It uses 16 items per axis and mode at one frozen revision,
validates one-token `X`, `Y`, and `Z` answers and the single clean/corrupt token
difference across templates A--C, and applies no prior-strength threshold or
outcome-based item exclusion. The exact gate contract is recorded in
`SEMANTIC-PREFLIGHT.md`.

## Supplementary figures

![](phase_a_curve.png)

*Figure S1. Reversion (top), source-erased prior strength, and source
contribution across the OLMo-2 1B trajectory (bottom). The panels share the
training-token axis; the lower panel uses candidate-margin units.*

![](choice_decomposition.png)

*Figure S2. Candidate-choice decomposition across the complete OLMo-2
trajectories. Lines show item-cluster means for the explicit source, parametric
prior, and distractors; shaded regions mark the selected late transitions.*

![](phase_b_deterministic_curve.png)

*Figure S3. Layer-wise source-donor patch effect for conflict and neutral items
at the OLMo-2 1B 2496B-to-2999B transition. The dashed line marks the
first-75%-depth summary boundary.*

![](phase_b_srcpatch_curve.png)

*Figure S4. Layer-wise source-donor patch effect at the OLMo-2 1B
global-maximum pair (1909B-to-1993B) and selected template-robust pair
(2496B-to-2999B), including the neutral diagnostic.*

## Supplementary tables

**Table S1. Family-by-template semantic peak details.** These conditional
descriptive summaries underlie Table 5 and Figure 2. Intervals and means are
copied from `cross_family_semantic_pool.json`; no rows were removed after
observing the effect direction.

### Behavior: source gain

Source gain is the clean-minus-corrupt X-minus-Y margin in single-token
logit-difference units. The `reversion` column is the pairwise Y-over-X error
mean retained by the semantic summary.

| Family | Axis | Template | Source gain mean | 95% CI | Reversion mean |
|---|---|---|---:|---|---:|
| Amber-7B | Type/category | A | 4.462 | [4.462, 4.462] | 0.000 |
| Amber-7B | Type/category | B | 3.618 | [3.618, 3.618] | 0.073 |
| Amber-7B | Type/category | C | 2.198 | [2.198, 2.198] | 0.178 |
| Amber-7B | Relation/slot | A | 4.700 | [4.700, 4.700] | 0.195 |
| Amber-7B | Relation/slot | B | 4.990 | [4.990, 4.990] | 0.043 |
| Amber-7B | Relation/slot | C | 3.061 | [3.061, 3.061] | 0.027 |
| Amber-7B | Naturalistic | A | 4.905 | [4.603, 5.209] | 0.082 |
| Amber-7B | Naturalistic | B | 3.858 | [3.708, 4.005] | 0.038 |
| Amber-7B | Naturalistic | C | 2.925 | [2.795, 3.047] | 0.006 |
| Pythia-6.9B | Type/category | A | 3.120 | [2.848, 3.394] | 0.161 |
| Pythia-6.9B | Type/category | B | 2.484 | [2.275, 2.699] | 0.306 |
| Pythia-6.9B | Type/category | C | 1.494 | [1.314, 1.675] | 0.463 |
| Pythia-6.9B | Relation/slot | A | 2.233 | [1.930, 2.515] | 0.913 |
| Pythia-6.9B | Relation/slot | B | 3.445 | [2.351, 4.527] | 0.196 |
| Pythia-6.9B | Relation/slot | C | 1.590 | [1.358, 1.820] | 0.695 |
| Pythia-6.9B | Naturalistic | A | 4.403 | [3.913, 4.881] | 0.031 |
| Pythia-6.9B | Naturalistic | B | 2.702 | [2.419, 3.001] | 0.300 |
| Pythia-6.9B | Naturalistic | C | 3.660 | [3.403, 3.921] | 0.012 |

The degenerate intervals for some Amber type/relation rows are a property of
the stored summary and should not be read as evidence of zero sampling
uncertainty. The machine-readable artifact is the source of record.

### Source-donor patch effect

The patch effect is the item-wise maximum layer shift from the corrupt baseline,
averaged within each template. The selected layer is zero-indexed in the
family-specific residual output.

| Family | Axis | Template | Patch effect mean | 95% CI | Selected layer |
|---|---|---|---:|---|---:|
| Amber-7B | Type/category | A | 4.892 | [4.483, 5.308] | 5 |
| Amber-7B | Type/category | B | 4.075 | [3.675, 4.500] | 7 |
| Amber-7B | Type/category | C | 2.583 | [2.267, 2.933] | 10 |
| Amber-7B | Relation/slot | A | 4.992 | [4.725, 5.258] | 18 |
| Amber-7B | Relation/slot | B | 5.333 | [5.117, 5.558] | 12 |
| Amber-7B | Relation/slot | C | 3.300 | [3.058, 3.542] | 18 |
| Amber-7B | Naturalistic | A | 5.442 | [5.042, 5.842] | 12 |
| Amber-7B | Naturalistic | B | 4.050 | [3.783, 4.325] | 0 |
| Amber-7B | Naturalistic | C | 3.133 | [2.925, 3.342] | 5 |
| Pythia-6.9B | Type/category | A | 3.273 | [2.961, 3.603] | 3 |
| Pythia-6.9B | Type/category | B | 2.711 | [2.477, 2.948] | 3 |
| Pythia-6.9B | Type/category | C | 1.971 | [1.739, 2.222] | 2 |
| Pythia-6.9B | Relation/slot | A | 2.541 | [2.219, 2.849] | 5 |
| Pythia-6.9B | Relation/slot | B | 3.548 | [3.001, 4.105] | 0 |
| Pythia-6.9B | Relation/slot | C | 1.880 | [1.619, 2.138] | 17 |
| Pythia-6.9B | Naturalistic | A | 4.436 | [4.073, 4.823] | 7 |
| Pythia-6.9B | Naturalistic | B | 2.854 | [2.636, 3.072] | 0 |
| Pythia-6.9B | Naturalistic | C | 3.677 | [3.430, 3.907] | 3 |

The pooled means in Table 5 apply equal template weight within each family and
then equal family weight; they are not obtained by pooling the 18 rows as if
they were independent observations.

**Table S2. Fact-cluster sensitivity.** The existing rows are reweighted so
that each `(category, subject, Y)` tuple contributes one observation. Estimates
are supplementary sensitivity summaries, not replacements for the primary
item-cluster analysis; intervals use 10,000 fact-cluster bootstrap draws. The
patch columns use single-token X-minus-Y logit-difference units.

| Model | Fact clusters | Excursion | 95% CI | Extrema (B) |
|---|---:|---:|---|---|
| OLMo-2 1B | 28 | 0.482 | [0.380, 0.607] | 63 / 1909 / 1993 |
| OLMo-2 7B | 28 | 0.242 | [0.161, 0.321] | 1901 / 3201 / 3896 |

The fact-weighted patch summaries are shown below. The table reports
fact-weighted absolute early-depth patch effects (`q_early`, `q_late`) and
their change (`delta_q = q_late - q_early`); intervals are fact-cluster
bootstrap 95% intervals.

| Group | Matched rows | Facts | q_early [95% CI] | q_late [95% CI] | delta_q [95% CI] |
|---|---:|---:|---:|---:|---:|
| 1B conflict A | 116 | 23 | 2.367 [2.004, 2.726] | 3.423 [3.053, 3.782] | +1.056 [0.826, 1.292] |
| 1B conflict B | 116 | 23 | 3.204 [2.876, 3.557] | 3.049 [2.865, 3.224] | -0.155 [-0.495, 0.179] |
| 1B conflict C | 116 | 23 | 2.213 [1.837, 2.627] | 3.322 [2.886, 3.781] | +1.109 [0.783, 1.438] |
| 1B neutral A | 120 | 70 | 3.070 [2.794, 3.343] | 4.382 [4.111, 4.651] | +1.311 [1.111, 1.517] |
| 1B neutral B | 120 | 70 | 3.455 [3.150, 3.770] | 3.256 [3.043, 3.475] | -0.199 [-0.444, 0.040] |
| 1B neutral C | 120 | 70 | 2.750 [2.445, 3.053] | 3.559 [3.270, 3.857] | +0.809 [0.538, 1.093] |
| 7B conflict A | 57 | 21 | 0.767 [0.518, 0.978] | 1.764 [1.502, 2.017] | +0.997 [0.754, 1.239] |
| 7B conflict B | 57 | 21 | 1.964 [1.714, 2.199] | 2.619 [2.261, 2.957] | +0.656 [0.409, 0.900] |
| 7B neutral A | 60 | 46 | 1.327 [1.084, 1.576] | 2.630 [2.262, 3.022] | +1.303 [1.014, 1.611] |
| 7B neutral B | 60 | 46 | 2.223 [1.980, 2.477] | 2.764 [2.434, 3.092] | +0.541 [0.334, 0.742] |

## Reproducibility note

The complete semantic detail table is also available as
`TABLE-S1-SEMANTIC-DETAILS.md`; the machine-readable fact sensitivity output
and analysis definition are in `CLUSTER-SENSITIVITY.md` and
`cluster_sensitivity.json`. The checksum and container provenance are recorded
in `SUPPLEMENTARY-REPRODUCIBILITY-NOTE.md`. Raw semantic rows are retained in
the experiment archive but are not yet deposited in a persistent public archive.
