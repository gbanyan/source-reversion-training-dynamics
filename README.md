# Non-monotonic source reversion in language-model pretraining

Reproducibility package for the manuscript:

> **Non-monotonic source reversion across language-model pretraining checkpoints: behavioral trajectories and source-position residual interventions**

The study measures how an explicit source answer and a competing parametric
answer exchange top-choice preference across released language-model
checkpoints. It then tests the causal effect of replacing the residual state at
the answer-bearing source position at selected transitions.

## Repository layout

| Directory | Contents |
| --- | --- |
| [`paper/`](paper/) | Manuscript, supplementary manuscript, references, and rendered PDFs |
| [`figures/`](figures/) | Main and supplementary figures used by the manuscript |
| [`results/canonical/`](results/canonical/) | Machine-readable summaries used for reported values and tables |
| [`results/raw/`](results/raw/) | Selected raw JSON shards supporting the canonical summaries |
| [`code/`](code/) | Analysis scripts and the small item/model helper modules used by the runners |
| [`configs/`](configs/) | Checkpoint manifests and frozen evaluation configuration |
| [`docs/`](docs/) | Claim/evidence map, figure map, metric specification, and audit notes |

`SHA256SUMS` records the release checksums for all packaged files (excluding
the checksum file itself).

Start with [`paper/MANUSCRIPT.pdf`](paper/MANUSCRIPT.pdf) for the article and
[`paper/SUPPLEMENTARY.pdf`](paper/SUPPLEMENTARY.pdf) for the supplementary
material. The source versions are provided alongside the PDFs.

## Regenerating the PDFs

The checked-in PDFs are generated from the Markdown sources with Pandoc and
XeLaTeX. From `paper/`, with the repository's `figures/` directory available:

```bash
pandoc MANUSCRIPT.md -o MANUSCRIPT.pdf \
  --from=markdown+tex_math_single_backslash --pdf-engine=xelatex \
  --resource-path=.:../figures
pandoc SUPPLEMENTARY.md -o SUPPLEMENTARY.pdf \
  --from=markdown+tex_math_single_backslash --pdf-engine=xelatex \
  --resource-path=.:../figures
```

Pandoc and XeLaTeX are system dependencies and are not installed by
`requirements.txt`.

## Evidence package

The canonical results are the JSON files in `results/canonical/`. They include
the OLMo-2 1B and 7B trajectory summaries, candidate-choice decompositions,
source-position patch summaries, frozen semantic-family summaries, control
diagnostics, and cluster-sensitivity outputs. The raw JSON files are retained
for the audited trajectory and causal batches; they are not model weights.

The complete semantic raw matrix is not included in this initial public
snapshot. The public package currently exposes its validated pooled summary and
the corresponding table/detail files. A future release may add the raw matrix
if its redistribution terms permit it.

## Re-running analyses

The scripts in `code/` are snapshots of the validated analysis and model
evaluation code. The pure analysis scripts operate on JSON artifacts and use
NumPy/SciPy/Matplotlib. The model runners additionally require PyTorch,
Transformers, a CUDA-capable device, and access to the upstream Hugging Face
model checkpoints. Model weights are intentionally not redistributed here.

Most scripts preserve the original same-directory artifact conventions. For a
new analysis, use the canonical JSON summaries as the source of record and
pass explicit input/output paths where a script exposes command-line options.
The commands and exact estimands are documented in [`docs/`](docs/) and in the
Methods section of the manuscript.

```bash
python code/analyze_choice_decomposition.py \
  --rows results/raw/phase_a_det_shard0.json results/raw/phase_a_det_shard1.json \
  --output results/recomputed/choice_decomposition_1b.json
```

The repository does not include the internal cluster launchers used to run
models on private machines. Those launchers contained machine-specific mount
paths; portable Python runners are included instead.

## Scope and interpretation

The evidence supports a bounded low--high--low exchange between explicit-source
and parametric top choices in the tested OLMo-2 trajectories, together with a
causal effect of residual replacement at the tested source-token position at
selected transitions. It does not establish a universal training phase law,
a source-content-specific mediator, a complete circuit, or a reliable
prospective steering rule. See [`docs/CLAIM-EVIDENCE-MATRIX.md`](docs/CLAIM-EVIDENCE-MATRIX.md)
for the claim-to-artifact map.

## Data and code availability

See [`DATA-AVAILABILITY.md`](DATA-AVAILABILITY.md) for the statement prepared
for a journal submission. Upstream model and dataset terms remain applicable;
this repository does not redistribute model weights or third-party corpora.

## Citation

Please cite the manuscript once it has a persistent publication identifier.
The BibTeX records used by the manuscript are in
[`paper/references.bib`](paper/references.bib).

## License

No reuse license has been selected for this review snapshot. A code/data
license should be added by the authors before a final public release.
