# Phase A — deterministic within-run and cross-size trajectories

**Date:** 2026-08-24. **Status:** `DETERMINISTIC-REPLICATION` with an
`INDEPENDENT-SIZE-SIGNAL`.

## Reproducibility correction

The original item generator converted candidate IDs through `set`, so the
candidate ordering could depend on `PYTHONHASHSEED` across worker processes.
The generator now preserves insertion order, all authoritative runs use
`PYTHONHASHSEED=0`, and the complete 1B trajectory was rerun. Prompt SHA256 is
identical under hash seeds 1 and 987. Pre-fix files remain discovery provenance
only; see `CLAIM-EVIDENCE-MATRIX.md` and `PUBLIC-RELEASE-CHECKLIST.md`.

## 1B deterministic trajectory

Across 24 checkpoints of the uninterrupted OLMo-2-0425-1B stage-1 run,
source/prior top-choice behavior is strongly non-monotonic. Reversion rises
from .037 at
63B tokens to .499 at 1909B, falls to .065 at 1993B, rises again to .470 at
2496B, and falls to .182 at 2999B.

- Largest low--high--low reversion excursion: **.434**.
- Unique-item bootstrap 95% interval: **[.362, .523]**; none of 10,000 draws
  had excursion ≤ .05.
- Deterministic item-half excursions: **.439** and **.462**.
- Reversion correlates with source-erased prior strength (Spearman ρ=.570,
  p=.0036) and inversely with conflict source contribution (ρ=−.579,
  p=.0030). These associations are descriptive, not causal.

The later 2496B→2999B source-choice transition is the primary event because it
replicates
across all three frozen prompt templates. Reversion drops are A **.336**
95% [.250, .431], B **.138** [.060, .216], and C **.397** [.302, .491]. The
earlier 1909B→1993B event is template-specific and is not used as the primary
claim.

## Independent 7B trajectory

Eight checkpoints from the official OLMo-2-1124-7B stage-1 trajectory provide
cross-scale corroboration in a separate release. Reversion is .295, .251,
.146, .287, .333, .311, .387, and .157 from 101B through 3896B tokens. The
1901B→3201B→3896B
excursion is **.230**, bootstrap 95% **[.158, .298]**.

Progress-aligned 1B and 7B curves have a descriptive Spearman ρ=−.048. Because
the aligned points are sparse and serially dependent, the nominal p-value is
not used as an inferential shared-boundary test; the defensible result is
cross-size instability, not a universal phase boundary.
At the primary 7B transition (3201B→3896B), the template audit uses the
pairwise Y-over-X error indicator rather than the all-candidate top-choice
reversion used for the trajectory. The pairwise error drop is positive for
template A (**.224**, [.129, .319]) and B (**.164**, [.086, .241]) but not the
harder template C (**.069**, [−.026, .164]). The endpoint rates in this
template audit are raw-row means over 120 rows. The paired changes and
intervals use the 116 common item keys after matching, which is the inference
estimand; rounded endpoint subtraction therefore need not equal the paired
change.

## Interpretation boundary

The candidate-choice decomposition shows that the selected events are source/
prior exchanges rather than distractor-only changes. At 1B, p_X changes
0.390→0.762 while p_Y changes 0.470→0.182 and p_other changes
0.140→0.056; at 7B the corresponding changes are 0.366→0.722,
0.387→0.157, and 0.247→0.121. The evidence supports non-monotonic source/
prior choice dynamics within these fixed runs. Transition timing and prompt
expression are trajectory-dependent, and no common token threshold, universal
prompt invariance, or monotonic scaling law is established.

## Authoritative artifacts

- `phase_a_det_shard0.json`, `phase_a_det_shard1.json`,
  `phase_a_summary.json`, `phase_a_curve.png`;
- `phase_a_7b_shard0.json`, `phase_a_7b_shard1.json`,
  `phase_a_7b_summary.json`;
- `deterministic_template_summary.json`, `template_7b_summary.json`;
- `choice_decomposition_1b.json`, `choice_decomposition_7b.json`,
  `choice_decomposition.png`;
- `analyze_phase_a.py`, `analyze_7b_replication.py`,
  `analyze_deterministic_templates.py`, `analyze_7b_templates.py`.
