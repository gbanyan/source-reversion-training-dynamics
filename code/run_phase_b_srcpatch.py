"""Causal source-position patching across OLMo training checkpoints.

Prompts differ in exactly one token: the source answer is X in the clean
prompt and a neutral candidate Z in the corrupt prompt.  At each layer, patch
only that token's clean residual into the corrupt run and measure recovery of
the final X-vs-Y margin.  Every matched item is retained in the output; no
outcome-defined filtering is used while collecting features.
"""

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
sys.path.insert(0, str(PILOT))

import gen_items  # noqa: E402
from prior_law import _cached_snapshot  # noqa: E402
from run_patch_srcpos import build_srcpos_pairs, first_id, is_single  # noqa: E402
from template_prompts import render_pair, TEMPLATES  # noqa: E402


def transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return decoder blocks for the open decoder families used here."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        f"Unsupported decoder layout for {type(model).__name__}; "
        "expected model.layers, gpt_neox.layers, or transformer.h"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokens-b", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--n-items", type=int, default=60)
    parser.add_argument("--item-mode", choices=("conflict", "neutral"), default="conflict")
    parser.add_argument("--template", choices=tuple(TEMPLATES), default="A")
    parser.add_argument("--model-tag")
    parser.add_argument("--model-size", type=float)
    parser.add_argument("--family")
    parser.add_argument(
        "--axis-unit",
        default="training_tokens_b",
        help="Label for the numeric checkpoint axis (e.g. checkpoint ordinal).",
    )
    parser.add_argument(
        "--donor-mode", choices=("source", "corrupt"), default="source",
        help="Hidden-state donor for the source-position patch; corrupt is a no-op control.",
    )
    return parser.parse_args()


def build_neutral_pairs(tokenizer, n_target: int, seed: int = 13) -> list[dict]:
    """Token-aligned source-X/source-Z pairs with fictional subjects lacking catalogued facts."""
    rng = np.random.default_rng(seed)
    pairs = []
    cats = list(gen_items.CATS)
    while len(pairs) < n_target:
        cat = cats[rng.integers(len(cats))]
        spec = gen_items.CATS[cat]
        subj = rng.choice(spec["fiction"])
        pool = [name for name in spec["pool"] if is_single(tokenizer, name)]
        if len(pool) < 4:
            continue
        x_name, z_name, y_name, filler = rng.choice(pool, size=4, replace=False)
        candidates = [x_name, z_name, y_name, filler]
        candidates = [candidates[i] for i in rng.permutation(len(candidates))]
        ids = gen_items._ids(rng, len(candidates))
        roster = ", ".join(
            f"<{ids[i]}> {name}" for i, name in enumerate(candidates)
        )
        stem = spec["stem"].format(subj=subj)

        def render(answer: str) -> str:
            source = spec["source"].format(subj=subj, ans=answer)
            return f"Options: {roster}. {source}\n{stem}"

        clean, corrupt = render(x_name), render(z_name)
        clean_ids = tokenizer(clean, return_tensors="pt").input_ids[0]
        corrupt_ids = tokenizer(corrupt, return_tensors="pt").input_ids[0]
        if clean_ids.shape != corrupt_ids.shape:
            continue
        differences = (clean_ids != corrupt_ids).nonzero().flatten().tolist()
        if len(differences) != 1:
            continue
        pairs.append({
            "cat": cat,
            "subj": subj,
            "x_name": str(x_name),
            "z_name": str(z_name),
            "y_name": str(y_name),
            "prompt_conf": clean,
            "prompt_corr": corrupt,
            "src_pos": differences[0],
        })
    return pairs


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
    layers = transformer_layers(model)
    n_layers = len(layers)
    print(
        f"revision={args.revision} tokens={args.tokens_b}B "
        f"layers={n_layers} items={len(pairs)}",
        flush=True,
    )

    def encode(prompt: str) -> torch.Tensor:
        return tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    @torch.no_grad()
    def margin(ids: torch.Tensor, xid: int, yid: int) -> float:
        logits = model(ids).logits[0, -1].float()
        return float(logits[xid] - logits[yid])

    rows = []
    for index, pair in enumerate(pairs):
        clean_prompt, corrupt_prompt, _erased_prompt, _fame = render_pair(
            pair, args.template, args.item_mode
        )
        xid = first_id(tokenizer, pair["x_name"])
        yid = first_id(tokenizer, pair["y_name"])
        clean_ids = encode(clean_prompt)
        corrupt_ids = encode(corrupt_prompt)
        if clean_ids.shape != corrupt_ids.shape:
            raise ValueError(
                f"rendered prompt shape mismatch for template {args.template}: "
                f"{tuple(clean_ids.shape)} vs {tuple(corrupt_ids.shape)}"
            )
        differences = (clean_ids[0] != corrupt_ids[0]).nonzero().flatten().tolist()
        if len(differences) != 1:
            raise ValueError(
                f"expected one rendered source-token difference, got {differences} "
                f"for template {args.template}"
            )
        src_pos = differences[0]

        clean_out = model(clean_ids, output_hidden_states=True)
        clean_logits = clean_out.logits[0, -1].float()
        clean_margin = float(clean_logits[xid] - clean_logits[yid])
        if args.donor_mode == "source":
            donors = [state[0, src_pos].detach().clone() for state in clean_out.hidden_states]
            corrupt_margin = margin(corrupt_ids, xid, yid)
        else:
            corrupt_out = model(corrupt_ids, output_hidden_states=True)
            donors = [state[0, src_pos].detach().clone() for state in corrupt_out.hidden_states]
            corrupt_margin = float(corrupt_out.logits[0, -1, xid] - corrupt_out.logits[0, -1, yid])

        recovery = []
        for layer in range(n_layers):
            donor = donors[layer + 1]

            def patch(_module, _inputs, output, *, vector=donor):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[:, src_pos, :] = vector.to(hidden.dtype)
                return output

            handle = layers[layer].register_forward_hook(patch)
            try:
                recovery.append(margin(corrupt_ids, xid, yid) - corrupt_margin)
            finally:
                handle.remove()

        rows.append({
            "revision": args.revision,
            "model": args.model_tag or args.repo.split("/")[-1],
            "size": args.model_size,
            "family": args.family or args.repo.split("/")[0].lower(),
            "training_tokens_b": args.tokens_b,
            "training_axis": args.tokens_b,
            "axis_unit": args.axis_unit,
            "template": args.template,
            "item_mode": args.item_mode,
            "item_index": index,
            "cat": pair["cat"],
            "subj": pair["subj"],
            "x": pair["x_name"],
            "z": pair["z_name"],
            "y": pair["y_name"],
            "source_position": src_pos,
            "donor_mode": args.donor_mode,
            "clean_margin": clean_margin,
            "corrupt_margin": corrupt_margin,
            "source_effect": clean_margin - corrupt_margin,
            "reversion": int(clean_margin < 0),
            "recovery_by_layer": recovery,
        })
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(pairs)}", flush=True)

    args.output.write_text(json.dumps({
        "repo": args.repo,
        "item_mode": args.item_mode,
        "template": args.template,
        "revision": args.revision,
        "training_tokens_b": args.tokens_b,
        "training_axis": args.tokens_b,
        "axis_unit": args.axis_unit,
        "n_layers": n_layers,
        "rows": rows,
    }, indent=1) + "\n")
    print(f"saved {args.output}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
