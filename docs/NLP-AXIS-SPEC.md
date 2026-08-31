# NLP meaning-extension specification

This document records the semantic extensions run after the identity-axis
result was replicated in two independent model families.
The extensions test whether source use is tied to a particular lexical fact
format or survives a change in the semantic role being answered.

## Common protocol

Every axis uses the same paired construction as the primary experiment:

- **conflict:** the passage states counterfactual answer `X`, while the
  catalogued factual answer `Y` is used as the parametric candidate and its
  source-erased preference is measured;
- **corrupt:** the matched passage states a third candidate `Z`;
- **neutral:** a fictional or otherwise prior-minimized subject without a
  catalogued factual prior, with `X` stated;
- **source-erased:** the passage omits the answer, measuring prior-only choice;
- **agree (pilot schema only):** the generator can state the known answer to
  check a capability ceiling, but no agree-condition measurements are included
  in the canonical confirmation artifacts.

The candidate list, subject, question, and all non-answer tokens are identical
within a conflict/corrupt pair. `X`, `Y`, and `Z` must be single first-answer
tokens for the model tokenizer, and the rendered pair must differ at exactly
one source-answer token. The source-token position is patched upstream in the
causal run. Items are split by semantic subject/fact; repeated rendered rows
are averaged within that key rather than treated as independent train/test
samples.

Each axis has three paraphrase templates, a frozen 50/50 partition of the
semantic item keys, and an item seed recorded in the manifest. The runner's
canonical key is `axis:category:subject:Y`; `analyze_semantic.py` averages rows
sharing that key before its bootstrap. The canonical implementation assigns
halves by a deterministic hash of the same key; they are robustness partitions
used together in the gate, not independent held-out test sets, and the design
does not enforce complete category or relation-family holdouts. A new axis is
not allowed to change the primary identity-axis gate or to introduce a
favorable checkpoint after seeing its results.

## Axis 1 — type/category transfer

**Question:** Does the model use a source-stated category even when its stored
world knowledge favors a different category?

The fact bank consists of recognizable entities with a high-confidence type
(`Y`), for example an entity normally classified as a band, river, novel, or
capital. The passage assigns a counterfactual type `X`; candidates are
same-format type labels plus two distractors. A separate fictional-entity bank
provides neutral items. The tokenizer/schema preflight freezes valid item
construction; it does not apply a model-based prior-strength threshold or
exclude items because of an observed model outcome.

**Primary readout:** pairwise Y-over-X error/source-prior margin and
source-position patch effect at the selected transition. **Transfer readout:**
report the deterministic item-key halves separately; complete type-category
holdout is not part of the canonical confirmation run.

## Axis 2 — relation/slot filling

**Question:** Does source faithfulness generalize across semantic relations and
argument slots, rather than only `located-in` and `written-by`?

Use a balanced bank of relation families such as headquarters, inventor,
birthplace, currency, and language. Each item has an explicit
subject--relation--value triple and a mechanically matched counterfactual value.
The answer position is always the value slot, while the relation wording is
paraphrased across templates. The canonical split is by hashed item key rather
than a complete relation-family or subject holdout, so the result is a semantic
extension rather than a clean test of lexical-transfer independence.

**Primary readout:** source-vs-corrupt margin and source-donor patch effect by relation
family. The canonical confirmation reports wrong-position and random-residual
donors as geometry diagnostics; a relation-preserving global candidate
permutation remains a planned extension rather than a reported result.

## Axis 3 — naturalistic context-conflict QA/IE

**Question:** Does the effect persist in a short, document-like context rather
than a one-sentence synthetic template?

Construct two-to-four sentence profiles or mini news/knowledge snippets. The
evidence sentence states `X`; other sentences contain realistic discourse
connectives, aliases, and distractor entities. The query asks for one named
slot (QA) or extracts a typed subject–relation–object triple (IE). A matched
corrupt context changes only the evidence answer token to `Z`; a neutral context
uses a fictional subject. The model is scored on the candidate margin and on
exact extraction, with the same source-position intervention.

This axis is intentionally secondary and limited to one task format selected
before looking at results. It is not a license to replace the controlled axes
with an unconstrained benchmark.

## Development and confirmation schedule

1. **Preflight:** tokenizer/schema alignment and a 16-item GPU smoke at one
   frozen revision per family. The smoke uses template A while the shared
   builder validates one-token `X`, `Y`, and `Z` answers and the single
   clean/corrupt token difference across templates A--C. It rejects malformed
   or tokenizer-incompatible items, but applies no model-based prior-strength
   or capability threshold and does not filter by observed outcome before
   freezing the manifest.
2. **Confirmation:** after two families pass the primary gate, run each selected
   axis at the same pre/peak/post checkpoints, three templates, conflict and
   neutral modes (behavior `n=120`, source patch `n=60`). Run peak corrupt-donor
   controls and retain all rows, including faithful cases.
3. **Analysis:** report item-cluster bootstrap intervals, category/relation
   deterministic item-key halves, per-template results, and layer-wise patch
   effects. Treat conflict-minus-neutral as diagnostic; the headline is the
   source-vs-corrupt behavior anchored by neutral and erased controls.

An axis passes only if it clears the coverage/schema validity check, non-zero
   deterministic-half behavioral signal, positive source-position patch effect,
   and exact corrupt-donor no-op check. Wrong-position and random-residual
   controls are retained as diagnostics of intervention geometry. A failed axis
   is reported as a boundary of generalization and does not trigger another
   unplanned item bank.

## Artifact contract

Each axis gets a manifest, frozen item JSON, tokenizer/version record, raw
behavior and patch JSON, deterministic summary, and a small checksum file. The
same container and NAS layout are used for both families. Numeric checkpoint
metadata remains separate from semantic-axis metadata so Amber ordinals are
never interpreted as training-token counts.
