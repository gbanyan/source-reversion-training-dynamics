"""Run the frozen Phase-A behavioral scan over OLMo training checkpoints.

Designed for a GPU host. Results are saved after every checkpoint and completed
revisions are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "pilot"
sys.path.insert(0, str(PILOT))

import gen_items  # noqa: E402
from prior_law import run_one  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "checkpoints.json")
    parser.add_argument("--output", type=Path, default=HERE / "phase_a_rows.json")
    parser.add_argument("--n-per-fame", type=int, default=40)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-tag", help="Metadata tag written to each output row.")
    parser.add_argument("--model-size", type=float, help="Metadata parameter count in billions.")
    parser.add_argument("--family", help="Metadata model-family label.")
    parser.add_argument(
        "--checkpoint-index",
        type=int,
        help="Run only one zero-based manifest entry (for job arrays).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_existing(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    checkpoints = manifest["checkpoints"]
    if args.checkpoint_index is not None:
        checkpoints = [checkpoints[args.checkpoint_index]]

    print(f"repo={manifest['repo']} checkpoints={len(checkpoints)}")
    for checkpoint in checkpoints:
        print(f"  {checkpoint['revision']}")
    if args.dry_run:
        return

    pairs = gen_items.generate_pairs(n_per_fame=args.n_per_fame)
    output = load_existing(args.output)
    completed = {row["revision"] for row in output}

    for checkpoint in checkpoints:
        revision = checkpoint["revision"]
        if revision in completed:
            print(f"skip completed {revision}", flush=True)
            continue
        rows = run_one(
            manifest["repo"],
            pairs,
            revision=revision,
            cache_dir=args.cache_dir,
        )
        tag = args.model_tag or f"{manifest['repo'].split('/')[-1]}-{checkpoint['tokens_b']}B"
        for row in rows:
            row.update(
                model=tag,
                size=args.model_size if args.model_size is not None else 1,
                family=args.family or manifest["repo"].split("/")[0].lower(),
                training_stage=manifest["stage"],
                training_step=checkpoint["step"],
                training_tokens_b=checkpoint["tokens_b"],
                training_axis=checkpoint.get("axis_value", checkpoint["tokens_b"]),
                axis_unit=manifest.get("checkpoint_unit", "training_tokens_b"),
                revision=revision,
            )
        output.extend(rows)
        args.output.write_text(json.dumps(output, indent=1) + "\n")
        print(
            f"{revision}: n={len(rows)} "
            f"reversion={np.mean([r['reversion'] for r in rows]):.3f} "
            f"prior={np.mean([r['prior_strength'] for r in rows]):.3f} "
            f"dm={np.mean([r['dm'] for r in rows]):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
