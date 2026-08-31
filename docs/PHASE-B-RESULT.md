# Phase B — causal source gain with explicit boundary conditions

**Date:** 2026-08-24. **Status:** `CROSS-SIZE-CAUSAL-REPLICATION`;
prospective component-predictor gate failed.

## Design

Clean and corrupt prompts differ in exactly one token: the source answer is X
or neutral candidate Z. At every layer, only the clean source token's residual
is patched into the corrupt run, and the patch-induced shift of the final X-vs-Y
margin is measured. All matched items are retained. The pre-final frozen
summary (not preregistered) uses layers
0–11 of 16 for 1B and 0–23 of 32 for 7B (≤.75 normalized depth).

The runner recomputes the unique differing token after rendering each template
and aborts unless prompts remain token-aligned. This fixed an initial B/C
positioning error documented in the claim/evidence and public-release notes.

## 1B primary transition: 2496B→2999B

The behavior column below is the pairwise Y-over-X error
`1{m_c<0}` computed by the source-patching batch; it is not the
all-candidate top-choice reversion used in the Phase-A trajectory.

| Template | Pairwise Y-over-X error | Δ causal patch effect | Paired bootstrap 95% |
|---|---:|---:|---:|
| A | .474→.138 | **+1.054** | **[.854, 1.255]** |
| B | .293→.164 | −.071 | [−.261, .125] |
| C | .621→.224 | **+1.008** | **[.802, 1.215]** |

The causal patch-effect change is positive on A/C but null on B. B's final
source effect is also essentially stable (3.172→3.123), despite a smaller
behavioral improvement. The result therefore establishes a local causal effect
of replacing the source-position residual, not a complete template-invariant
explanation of every behavioral change.

Neutral-source controls change similarly: A +1.284 [1.095, 1.470], B −.020
[−.221, .180], and C +.677 [.474, .888]. Conflict-minus-neutral
difference-in-differences is null for A/B and positive only for C (+.331,
[.044, .614]). A conflict-specific gate is not supported.

## Independent 7B causal replication: 3201B→3896B

The 7B test was restricted before execution to behavior-replicating templates
A/B, with 60 generated items per condition.
The behavior column is the same pairwise Y-over-X error, rather than the
all-candidate trajectory reversion.

| Template | Pairwise Y-over-X error | Δ causal patch effect | Paired bootstrap 95% |
|---|---:|---:|---:|
| A | .386→.193 | **+.968** | **[.752, 1.191]** |
| B | .316→.088 | **+.672** | **[.469, .877]** |

Neutral causal patch effects also increase: A +1.295 [1.034, 1.569] and B +.499
[.299, .694]. Both conflict-minus-neutral difference-in-differences include
zero. Thus the causal result replicates across model size while again pointing
to a general source-position margin effect, whose behavioral expression is
amplified by a strong parametric competitor.

## Predictor and intervention gate

Across 12 deterministic 1B checkpoint×template groups, final neutral source
gain has a descriptive Spearman association with the pairwise Y-over-X error
(ρ=−.796). Because these groups share checkpoints and templates, the nominal
rank-correlation p-value is not used as inferential evidence. However,
none of 36 pre-final frozen L0–L11 logit-lens, attention, or source-routing tests
survives Bonferroni correction over the 48-test family. Frozen routing models fail B/C generalization and the
checkpoint-rate MAE ≤.05 criterion. A post-hoc L13 attention association also
fails correction across the 48-test family.

The prospective component-selection gate therefore **fails**. No targeted-head
intervention was run, and the paper must not claim an early-warning predictor,
stable causal head, selective rescue, or successful steering.

## Defensible causal claim

Pretraining can non-monotonically change the measured source/prior choice
balance, and source-position residual replacement can causally shift the
measured margin before the final layer. This patch effect changes at the
primary transitions of independent 1B and 7B trajectories, but its expression
depends on prompt format and it is not unique to context–memory conflict.

## Authoritative artifacts

- `phase_b_det_*.json`, `phase_b_deterministic_summary.json`,
  `phase_b_deterministic_curve.png`;
- `phase_b_7b_*.json`, `phase_b_7b_summary.json`;
- `deterministic_template_summary.json`, `routing_predictor_summary.json`;
- `run_phase_b_srcpatch.py`, `run_phase_b_template_batch.sh`,
  `run_phase_b_7b_batch.sh`, `analyze_phase_b_deterministic.py`, and
  `analyze_phase_b_7b.py`.
