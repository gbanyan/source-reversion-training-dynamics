# Semantic preflight and gate contract

**Audit date:** 2026-08-31

This note records what the semantic preflight does and, equally importantly,
what it does not do. The preflight is a tokenizer/schema smoke check before
the frozen confirmation runs. It is not a model-based item-selection stage.

## Preflight implementation

`run_semantic_preflight.sh` invokes `run_semantic_behavior.py` with
`--template A` and `--n-items 16` for each requested axis and each of the two
item modes (`conflict` and `neutral`) at one frozen revision. The shared item
builder (`semantic_items.py`, through `load_or_build_pairs`) validates the
candidate geometry for templates A--C: `X`, `Y`, and `Z` must be single
first-answer tokens for the pinned tokenizer, and the rendered clean/corrupt
pair must differ at exactly one source-answer token.

Malformed or tokenizer-incompatible items are rejected. No model-output
criterion is used to select the bank: the preflight does not impose a
prior-strength threshold, does not require a minimum capability score, and
does not discard an item or confirmation row because its source gain, patch
effect, or other observed outcome is favorable or unfavorable. The frozen
banks contain the prespecified type, relation, and naturalistic entries
described in `NLP-AXIS-SPEC.md`; the final confirmation inputs retain the full
prespecified matrix.

## Confirmation-gate operationalization

Because there is no separate model-based capability or prior-strength cutoff,
the semantic gate operationalizes validity and signal as follows:

1. all prespecified behavior and source-patch files must be present and pass
   their coverage and schema checks;
2. at least two of the three peak conflict templates must have a positive
   lower 95% confidence bound for source gain;
3. both deterministic item-key halves must have positive source gain for at
   least two peak templates;
4. at least two peak templates must have a positive lower 95% confidence bound
   for the source-position patch effect; and
5. every peak corrupt-donor control must be an exact no-op within `1e-9`.

These checks are implemented by `check_semantic_gate.py` and
`check_confirmation_gate.py`. The deterministic halves are robustness
partitions of the same item keys, not independent held-out test sets. Since
the quality checks and pooled summaries use the same confirmation files, the
pooled semantic estimates remain conditional descriptive summaries.

## Provenance boundary

The preflight script writes a machine-readable `preflight_summary.json` in the
run directory. The local review bundle does not contain those NAS-side raw
preflight logs, so this note documents the executable contract rather than
inventing a missing preflight statistic. The canonical scientific results
remain the frozen confirmation artifacts listed in
`CLAIM-EVIDENCE-MATRIX.md` and `PUBLIC-RELEASE-CHECKLIST.md`.
