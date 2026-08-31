"""Lightweight deterministic A/B/C template evaluation at one checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "pilot"
sys.path.insert(0, str(PILOT))

from run_phase_b_srcpatch import build_neutral_pairs  # noqa: E402
from prior_law import _cached_snapshot  # noqa: E402
from run_patch_srcpos import build_srcpos_pairs, first_id  # noqa: E402
from template_prompts import render_pair, TEMPLATES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokens-b", type=int, required=True)
    parser.add_argument("--template", choices=tuple(TEMPLATES), required=True)
    parser.add_argument("--item-mode", choices=("conflict", "neutral"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--n-items", type=int, default=120)
    parser.add_argument(
        "--axis-unit",
        default="training_tokens_b",
        help="Label for the numeric checkpoint axis (e.g. checkpoint ordinal).",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    tokenizer_source = args.repo
    tokenizer_kwargs = {"revision": args.revision, "cache_dir": args.cache_dir}
    tokenizer_revision = os.environ.get("TOKENIZER_REVISION")
    if tokenizer_revision:
        tokenizer_kwargs["revision"] = tokenizer_revision
    if os.environ.get("TOKENIZER_LOCAL_ONLY"):
        tokenizer_kwargs["local_files_only"] = True
    if os.environ.get("TOKENIZER_LOCAL_SNAPSHOT"):
        tokenizer_source = _cached_snapshot(
            args.repo, tokenizer_revision or args.revision, args.cache_dir
        ) or args.repo
        if tokenizer_source != args.repo:
            tokenizer_kwargs = {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    pairs = (
        build_srcpos_pairs(tokenizer, n_target=args.n_items)
        if args.item_mode == "conflict"
        else build_neutral_pairs(tokenizer, n_target=args.n_items)
    )
    model_source = args.repo
    model_kwargs = {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "dtype": torch.bfloat16,
    }
    if os.environ.get("MODEL_LOCAL_SNAPSHOT"):
        model_source = _cached_snapshot(args.repo, args.revision, args.cache_dir) or args.repo
        if model_source != args.repo:
            model_kwargs = {"dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs).to("cuda").eval()
    rows = []
    for index, pair in enumerate(pairs):
        clean, _corrupt, erased, fame = render_pair(
            pair, args.template, args.item_mode
        )
        xid = first_id(tokenizer, pair["x_name"])
        yid = first_id(tokenizer, pair["y_name"])
        clean_ids = tokenizer(clean, return_tensors="pt").input_ids.to("cuda")
        erased_ids = tokenizer(erased, return_tensors="pt").input_ids.to("cuda")
        clean_logits = model(clean_ids).logits[0, -1].float()
        erased_logits = model(erased_ids).logits[0, -1].float()
        clean_margin = float(clean_logits[xid] - clean_logits[yid])
        erased_margin = float(erased_logits[xid] - erased_logits[yid])
        rows.append({
            "revision": args.revision,
            "training_tokens_b": args.tokens_b,
            "training_axis": args.tokens_b,
            "axis_unit": args.axis_unit,
            "template": args.template,
            "item_mode": args.item_mode,
            "item_index": index,
            "cat": pair["cat"],
            "fame": fame,
            "subj": pair["subj"],
            "x": pair["x_name"],
            "z": pair["z_name"],
            "y": pair["y_name"],
            "clean_margin": clean_margin,
            "erased_margin": erased_margin,
            "source_gain": clean_margin - erased_margin,
            "error": int(clean_margin < 0),
        })
    args.output.write_text(json.dumps({
        "repo": args.repo,
        "revision": args.revision,
        "training_tokens_b": args.tokens_b,
        "training_axis": args.tokens_b,
        "axis_unit": args.axis_unit,
        "template": args.template,
        "item_mode": args.item_mode,
        "rows": rows,
    }, indent=1) + "\n")
    print(
        f"saved {args.output}: n={len(rows)} "
        f"error={np.mean([row['error'] for row in rows]):.3f} "
        f"source_gain={np.mean([row['source_gain'] for row in rows]):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
