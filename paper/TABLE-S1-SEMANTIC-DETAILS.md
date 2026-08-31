# Table S1. Family-by-template semantic peak details

These are the family/template summaries underlying Table 5 and Figure 2. The
semantic suite was gated before pooling, so this table is conditional on the
prespecified quality gate. Intervals and means are copied from
`cross_family_semantic_pool.json`; no rows were removed after observing the
effect direction. Amber checkpoint `ckpt_096` and Pythia step 32,000 (67B label)
are the selected family peaks. Checkpoint rows are not independent statistical
replicates.

## Behavior: source gain

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

## Source-donor patch effect

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
