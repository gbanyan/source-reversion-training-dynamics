"""Measure source gain on neutral subjects without catalogued facts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_phase_b_srcpatch import build_neutral_pairs
from run_patch_srcpos import first_id


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="allenai/OLMo-2-0425-1B")
    parser.add_argument(
        "--manifests", type=Path, nargs="+",
        default=[HERE / "checkpoints.json", HERE / "checkpoints_dense.json"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--n-items", type=int, default=60)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    by_revision = {}
    for manifest_path in args.manifests:
        for checkpoint in json.loads(manifest_path.read_text())["checkpoints"]:
            by_revision[checkpoint["revision"]] = checkpoint
    checkpoints = sorted(by_revision.values(), key=lambda row: row["tokens_b"])
    checkpoints = [
        checkpoint for index, checkpoint in enumerate(checkpoints)
        if index % args.num_shards == args.shard_index
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.repo, cache_dir=args.cache_dir)
    pairs = build_neutral_pairs(tokenizer, n_target=args.n_items)
    existing = json.loads(args.output.read_text()) if args.output.exists() else []
    completed = {row["revision"] for row in existing}
    print(
        f"shard={args.shard_index}/{args.num_shards} "
        f"checkpoints={len(checkpoints)} items={len(pairs)}",
        flush=True,
    )

    for checkpoint in checkpoints:
        revision = checkpoint["revision"]
        if revision in completed:
            print(f"skip {revision}", flush=True)
            continue
        model = AutoModelForCausalLM.from_pretrained(
            args.repo,
            revision=revision,
            cache_dir=args.cache_dir,
            dtype=torch.bfloat16,
        ).to("cuda").eval()
        rows = []
        for index, pair in enumerate(pairs):
            xid = first_id(tokenizer, pair["x_name"])
            yid = first_id(tokenizer, pair["y_name"])
            clean_ids = tokenizer(
                pair["prompt_conf"], return_tensors="pt"
            ).input_ids.to("cuda")
            corrupt_ids = tokenizer(
                pair["prompt_corr"], return_tensors="pt"
            ).input_ids.to("cuda")
            clean_logits = model(clean_ids).logits[0, -1].float()
            corrupt_logits = model(corrupt_ids).logits[0, -1].float()
            clean_margin = float(clean_logits[xid] - clean_logits[yid])
            corrupt_margin = float(corrupt_logits[xid] - corrupt_logits[yid])
            rows.append({
                "revision": revision,
                "training_tokens_b": checkpoint["tokens_b"],
                "item_index": index,
                "cat": pair["cat"],
                "subj": pair["subj"],
                "x": pair["x_name"],
                "z": pair["z_name"],
                "y": pair["y_name"],
                "clean_margin": clean_margin,
                "corrupt_margin": corrupt_margin,
                "source_effect": clean_margin - corrupt_margin,
                "clean_error": int(clean_margin < 0),
            })
        existing.extend(rows)
        args.output.write_text(json.dumps(existing, indent=1) + "\n")
        print(
            f"{revision}: source_effect="
            f"{np.mean([row['source_effect'] for row in rows]):.3f} "
            f"clean_error={np.mean([row['clean_error'] for row in rows]):.3f}",
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
