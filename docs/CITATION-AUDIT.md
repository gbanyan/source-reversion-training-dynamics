# Citation audit

**Audit date:** 2026-08-31

**Post-revision status:** passed. The 2026 ACL conflict-dynamics comparator,
OLMo 2 report, and LLM360/Amber release were added and verified; Burns et al.
is cited at its ICLR 2023 venue.

The audit was performed after the manuscript draft was complete. Each citation
was checked against an official primary record and mapped to the exact claim in
the text. No citation is used to support a numerical result from this
repository; repository artifacts are cited in prose by filename instead.

| Manuscript location | Claim | Citation | Verification result | Resolution |
|---|---|---|---|---|
| Abstract/Introduction/Related Work 2.1 | Source-grounded generation can be unfaithful to input evidence. | [1] Maynez et al. 2020 | ACL Anthology abstract directly reports input-unfaithful hallucinations. | Retained; wording limited to document-grounded faithfulness. |
| Introduction/Related Work 2.1 | Contextual cues can be overlooked in knowledge conflict. | [2] Zhou et al. 2023 | ACL Anthology abstract directly concerns context-faithful prompting under conflict. | Retained; no claim that their prompt method explains our trajectory. |
| Introduction/Related Work 2.1 | External-evidence use varies with conflict construction. | [3] Xie et al. 2024 | ICLR paper directly studies controlled knowledge conflicts and external evidence. | Retained as behavioral context. |
| Introduction/Methods 3.2 | Public checkpoints enable training-dynamics studies. | [4] Biderman et al. 2023 | PMLR record directly states 16 models and 154 checkpoints per model. | Retained; no claim that Pythia results are ours. |
| Introduction/Methods 3.2 | OLMo releases open weights/code/checkpoints. | [5] Groeneveld et al. 2024 | ACL record directly states open data, code, weights, and intermediate checkpoints. | Retained. |
| Introduction/Related Work 2.2 | Delayed generalization can occur late in controlled training. | [6] Power et al. 2022 | arXiv record directly reports generalization after overfitting; cited as analogy. | Retained with “conceptual analogy”; not used to name a phase in this paper. |
| Introduction/Related Work 2.2 | Progress measures can expose hidden training changes. | [7] Nanda et al. 2023 | ICLR/arXiv paper directly reports mechanistic progress measures for grokking. | Retained with explicit non-equivalence caveat. |
| Introduction/Related Work 2.3 | Probing does not establish behavioral use. | [8] Elazar et al. 2021 | TACL abstract directly motivates amnesic counterfactuals for behavioral explanation. | Retained as methodological distinction. |
| Introduction/Discussion 5.2 | Latent knowledge can differ from generated output. | [9] Burns et al. 2023 (arXiv:2212.03827) | OpenReview/arXiv abstract directly concerns knowledge in activations and output misalignment; the cited version is the ICLR 2023 paper. | Retained; no claim of identical task or model. |
| Introduction/Discussion 5.2 | Internal truthfulness representations can dissociate from outputs. | [10] Orgad et al. 2025 | ICLR record directly reports internal encoding/output discrepancy. | Retained; “truthfulness” is not equated with our counterfactual source task. |
| Introduction/Discussion 5.2 | Latent world-state information can be probed under unfaithful output. | [11] Feng et al. 2025 | ICLR paper directly reports propositional latent-world-state probes. | Retained as context. |
| Introduction/Related Work 2.3 | Causal mediation estimates component contributions through interventions. | [12] Vig et al. 2020 | Official NeurIPS record directly describes causal mediation interventions in GPT-2. | Retained; our intervention is described separately. |
| Introduction/Related Work 2.3 | Causal tracing localizes contributions to factual predictions. | [13] Meng et al. 2022 | Official NeurIPS paper directly describes causal tracing and factual recall. | Retained; no claim that our source path is a factual-recall circuit. |
| Introduction/Related Work 2.3 | Interchange interventions provide a causal-abstraction framework. | [14] Geiger et al. 2021 | Official NeurIPS record directly describes causal abstractions/interchange interventions. | Retained as methodological precedent. |
| Introduction/Related Work 2.3 | Activation-patching estimates depend on design choices. | [15] Zhang and Nanda 2024 | ICLR/arXiv abstract directly reports sensitivity to metrics and corruption. | Retained and reflected in our control design. |
| Introduction/Methods 3.2 | OLMo 2 releases intermediate checkpoints and training artifacts. | [16] Team OLMo et al. 2025 | The technical report describes OLMo 2 and its released checkpoints, data, code, and recipes. | Added as the direct OLMo-2 model-release citation. |
| Introduction/Related Work 2.1/2.4 | Parametric and in-context preferences have been traced during training, including Pythia and OLMo evidence. | [17] Kim et al. 2026 | ACL 2026 paper directly studies training-data effects on parametric/in-context knowledge use and real-model checkpoints. | Added; the manuscript narrows its gap to within-trajectory source/prior exchange plus local intervention. |
| Methods 3.4 | Amber is an independently released 7B model with intermediate checkpoints. | [18] Liu et al. 2024 | COLM 2024 paper directly introduces Amber and its open training artifacts/checkpoints. | Updated from the arXiv-only record to the formal COLM venue; OpenReview record and arXiv identifier retained. |
| Related Work 2.2 | Linear steerability can emerge at intermediate checkpoints and vary across concepts and model families. | [19] She et al. 2025 | ACL Anthology abstract directly reports intermediate-training emergence and cross-family/concept variation. | Added as context for the failed prospective gate; no equivalence with our source task is claimed. |
| Related Work 2.3 | Subspace patches can create an end-to-end effect without uniquely identifying the intended feature. | [20] Makelov et al. 2024 | ICLR paper directly reports interpretability illusions and dormant parallel pathways. | Added as methodological motivation for bounded donor-control interpretation. |
| Methods 3.2 | The exact OLMo-2 1B release used in the trajectory is `allenai/OLMo-2-0425-1B`. | [21] Allen Institute for AI model card | Official Hugging Face card exposes the exact repository identifier and model release. | Added as model metadata; accessed 2026-08-31. |
| Methods 3.2 | The exact OLMo-2 7B release used in the trajectory is `allenai/OLMo-2-1124-7B`. | [22] Allen Institute for AI model card (2024) | Official Hugging Face card exposes the exact repository identifier; its model-card citation metadata dates the release to 2024. | Added as model metadata; accessed 2026-08-31. |

## Bibliography cross-check

- All manuscript keys [1]–[22] are present in the numbered reference list and
  every citation is used in the text; matching BibTeX records are present in
  `references.bib`.
- Every bibliography entry is cited at least once in `MANUSCRIPT.md`.
- Author names, year, venue, page ranges, DOI, and URLs were checked against
  ACL Anthology, PMLR, NeurIPS proceedings, ICLR/OpenReview, or arXiv primary
  records as listed in `CITATION-LEDGER.md`.
- Burns et al. is cited as ICLR 2023 rather than only as its earlier arXiv
  preprint. The OLMo 2, Kim et al., and LLM360 records are included with their
  official identifiers; LLM360 is cited as a COLM 2024 paper with its
  OpenReview record and arXiv identifier.
- No recent Neurocomputing comparator is included in the bibliography because
  those records were used only to assess journal fit, not to support a claim.
