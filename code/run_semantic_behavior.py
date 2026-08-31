"""Behavioral runner for the post-replication NLP semantic axes."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "pilot"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PILOT))

from prior_law import _cached_snapshot  # noqa: E402
from semantic_items import first_id, load_or_build_pairs, render_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--axis", choices=("type", "relation", "naturalistic"), required=True)
    parser.add_argument("--tokens-b", type=int, required=True)
    parser.add_argument("--axis-unit", default="training_tokens_b")
    parser.add_argument("--template", choices=("A", "B", "C"), required=True)
    parser.add_argument("--item-mode", choices=("conflict", "neutral"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items-json", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--n-items", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--model-tag")
    parser.add_argument("--model-size", type=float)
    parser.add_argument("--family")
    return parser.parse_args()


def load_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    source = args.repo
    kwargs: dict[str, object] = {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
    }
    tokenizer_revision = os.environ.get("TOKENIZER_REVISION")
    if tokenizer_revision:
        kwargs["revision"] = tokenizer_revision
    if os.environ.get("TOKENIZER_LOCAL_ONLY"):
        kwargs["local_files_only"] = True
    if os.environ.get("TOKENIZER_LOCAL_SNAPSHOT"):
        source = _cached_snapshot(args.repo, tokenizer_revision or args.revision, args.cache_dir) or args.repo
        if source != args.repo:
            kwargs = {}
    return AutoTokenizer.from_pretrained(source, **kwargs)


def load_model(args: argparse.Namespace) -> AutoModelForCausalLM:
    source = args.repo
    kwargs: dict[str, object] = {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "dtype": torch.bfloat16,
    }
    if os.environ.get("MODEL_LOCAL_SNAPSHOT"):
        source = _cached_snapshot(args.repo, args.revision, args.cache_dir) or args.repo
        if source != args.repo:
            kwargs = {"dtype": torch.bfloat16}
    return AutoModelForCausalLM.from_pretrained(source, **kwargs).to("cuda").eval()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args)
    pairs = load_or_build_pairs(
        tokenizer, args.axis, args.item_mode, args.n_items, args.seed, args.items_json
    )
    model = load_model(args)
    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs):
        clean, corrupt, erased, fame = render_pair(pair, args.template, args.item_mode)
        clean_ids = tokenizer(clean, return_tensors="pt").input_ids.to("cuda")
        corrupt_ids = tokenizer(corrupt, return_tensors="pt").input_ids.to("cuda")
        erased_ids = tokenizer(erased, return_tensors="pt").input_ids.to("cuda")
        if clean_ids.shape != corrupt_ids.shape:
            raise ValueError(f"clean/corrupt shape mismatch for item {index}")
        differences = (clean_ids[0] != corrupt_ids[0]).nonzero().flatten().tolist()
        if len(differences) != 1:
            raise ValueError(f"expected one source-token difference, got {differences}")
        xid = first_id(tokenizer, pair["x_name"])
        yid = first_id(tokenizer, pair["y_name"])
        clean_logits = model(clean_ids).logits[0, -1].float()
        corrupt_logits = model(corrupt_ids).logits[0, -1].float()
        erased_logits = model(erased_ids).logits[0, -1].float()
        clean_margin = float(clean_logits[xid] - clean_logits[yid])
        corrupt_margin = float(corrupt_logits[xid] - corrupt_logits[yid])
        erased_margin = float(erased_logits[xid] - erased_logits[yid])
        rows.append({
            "repo": args.repo,
            "model": args.model_tag or args.repo.split("/")[-1],
            "size": args.model_size,
            "family": args.family or args.repo.split("/")[0].lower(),
            "axis": args.axis,
            "revision": args.revision,
            "training_tokens_b": args.tokens_b,
            "training_axis": args.tokens_b,
            "axis_unit": args.axis_unit,
            "template": args.template,
            "item_mode": args.item_mode,
            "item_index": index,
            "item_key": pair["item_key"],
            "cat": pair["cat"],
            "fame": fame,
            "subj": pair["subj"],
            "x": pair["x_name"],
            "z": pair["z_name"],
            "y": pair["y_name"],
            "source_position": differences[0],
            "clean_margin": clean_margin,
            "corrupt_margin": corrupt_margin,
            "erased_margin": erased_margin,
            "source_gain": clean_margin - corrupt_margin,
            "erased_shift": clean_margin - erased_margin,
            "reversion": int(clean_margin < 0),
        })
        if (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(pairs)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "repo": args.repo,
        "model": args.model_tag or args.repo.split("/")[-1],
        "size": args.model_size,
        "family": args.family or args.repo.split("/")[0].lower(),
        "axis": args.axis,
        "revision": args.revision,
        "training_tokens_b": args.tokens_b,
        "training_axis": args.tokens_b,
        "axis_unit": args.axis_unit,
        "template": args.template,
        "item_mode": args.item_mode,
        "n_items": len(rows),
        "rows": rows,
    }, indent=1) + "\n")
    print(
        f"saved {args.output}: n={len(rows)} "
        f"error={np.mean([row['reversion'] for row in rows]):.3f} "
        f"source_gain={np.mean([row['source_gain'] for row in rows]):.3f}",
        flush=True,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
