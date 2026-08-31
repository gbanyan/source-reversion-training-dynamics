# Results

`canonical/` contains the machine-readable summaries used to write the paper
and generate its tables. `raw/` contains selected JSON shards retained for
auditability and optional recomputation; it does not contain model weights.

Canonical artifacts:

- `phase_a_summary.json` — deterministic OLMo-2 1B trajectory.
- `phase_a_7b_summary.json` — separate-release OLMo-2 7B trajectory.
- `choice_decomposition_1b.json`, `choice_decomposition_7b.json` — explicit
  source, parametric, and distractor top-choice proportions.
- `phase_b_deterministic_summary.json`, `phase_b_7b_summary.json` — causal
  source-position patch summaries and controls.
- `cross_family_semantic_pool.json` — frozen semantic-axis summaries for the
  two independently examined model families.
- `cluster_sensitivity.json` — fact-cluster reweighting sensitivity analysis.
- `routing_predictor_summary.json` — the reported negative predictor gate.

The raw semantic matrix is not part of this initial public snapshot. Its
validated pooled summary is included above and is described in the manuscript
and supplementary material.
