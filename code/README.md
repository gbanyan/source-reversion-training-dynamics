# Analysis code

The Python files in this directory are copied from the validated analysis
workspace. `analyze_*.py`, `make_*.py`, `check_*.py`, and
`verify_*.py` operate on frozen JSON artifacts. The `run_*.py` files are
portable model-evaluation runners; they require upstream model checkpoints and
GPU-capable PyTorch/Transformers installations.

The private shell launchers are intentionally omitted because they contained
machine-specific NAS mount paths. No validated result depends on those paths
being public. Canonical summaries in `../results/canonical/` remain the source
of record.

Some legacy analysis scripts use the original same-directory file names. When
rerunning one of those scripts, work from a temporary copy with the required
JSON inputs next to the script, or use a script's explicit `--rows`,
`--summary`, or `--output` arguments where available.
