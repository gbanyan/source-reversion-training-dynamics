"""Source-position causal patching for a frozen semantic NLP axis."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "pilot"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PILOT))

from prior_law import _cached_snapshot  # noqa: E402
from semantic_items import first_id, load_or_build_pairs, render_pair  # noqa: E402


def transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(f"unsupported decoder layout for {type(model).__name__}")


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
    parser.add_argument("--n-items", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--model-tag")
    parser.add_argument("--model-size", type=float)
    parser.add_argument("--family")
    parser.add_argument(
        "--donor-mode",
        choices=("source", "corrupt", "wrong_position", "random_residual"),
        default="source",
        help="source donor or a predeclared falsification control",
    )
    return parser.parse_args()


def load_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    source = args.repo
    kwargs: dict[str, object] = {"revision": args.revision, "cache_dir": args.cache_dir}
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
    kwargs: dict[str, object] = {"revision": args.revision, "cache_dir": args.cache_dir, "dtype": torch.bfloat16}
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
    layers = transformer_layers(model)
    rows: list[dict[str, object]] = []

    def encode(prompt: str) -> torch.Tensor:
        return tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    def margin(ids: torch.Tensor, xid: int, yid: int) -> float:
        logits = model(ids).logits[0, -1].float()
        return float(logits[xid] - logits[yid])

    for index, pair in enumerate(pairs):
        clean, corrupt, _erased, fame = render_pair(pair, args.template, args.item_mode)
        xid = first_id(tokenizer, pair["x_name"])
        yid = first_id(tokenizer, pair["y_name"])
        clean_ids = encode(clean)
        corrupt_ids = encode(corrupt)
        if clean_ids.shape != corrupt_ids.shape:
            raise ValueError(f"clean/corrupt shape mismatch for item {index}")
        differences = (clean_ids[0] != corrupt_ids[0]).nonzero().flatten().tolist()
        if len(differences) != 1:
            raise ValueError(f"expected one source-token difference, got {differences}")
        source_position = differences[0]

        clean_out = model(clean_ids, output_hidden_states=True)
        clean_margin = float(clean_out.logits[0, -1, xid] - clean_out.logits[0, -1, yid])
        donor_position: int | None = source_position
        if args.donor_mode == "source":
            donors = [state[0, source_position].detach().clone() for state in clean_out.hidden_states]
            corrupt_margin = margin(corrupt_ids, xid, yid)
        elif args.donor_mode == "wrong_position":
            # Use a deterministic adjacent context position as the donor, but
            # still write it at the true source position. This preserves the
            # intervention geometry while removing the source-token content.
            donor_position = source_position - 1 if source_position > 0 else source_position + 1
            if donor_position >= clean_ids.shape[1]:
                raise ValueError(f"no valid wrong-position donor for item {index}")
            donors = [state[0, donor_position].detach().clone() for state in clean_out.hidden_states]
            corrupt_margin = margin(corrupt_ids, xid, yid)
        elif args.donor_mode == "random_residual":
            # Random vectors are norm-matched to the clean residual at each
            # layer, so this tests direction/content rather than scale.
            donors = [state[0, source_position].detach().clone() for state in clean_out.hidden_states]
            donor_position = None
            corrupt_margin = margin(corrupt_ids, xid, yid)
        else:
            corrupt_out = model(corrupt_ids, output_hidden_states=True)
            donors = [state[0, source_position].detach().clone() for state in corrupt_out.hidden_states]
            corrupt_margin = float(corrupt_out.logits[0, -1, xid] - corrupt_out.logits[0, -1, yid])

        recovery: list[float] = []
        for layer in range(len(layers)):
            donor = donors[layer + 1]
            if args.donor_mode == "random_residual":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(args.seed + index * 1009 + layer)
                random = torch.randn(
                    donor.shape, generator=generator, dtype=torch.float32
                ).to(donor.device, dtype=donor.dtype)
                ref_norm = donor.float().norm()
                random_norm = random.float().norm().clamp_min(1e-6)
                donor = random * (ref_norm / random_norm).to(random.dtype)

            def patch(_module, _inputs, output, *, vector=donor):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[:, source_position, :] = vector.to(hidden.dtype)
                return output

            handle = layers[layer].register_forward_hook(patch)
            try:
                recovery.append(margin(corrupt_ids, xid, yid) - corrupt_margin)
            finally:
                handle.remove()

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
            "source_position": source_position,
            "donor_mode": args.donor_mode,
            "donor_position": donor_position,
            "donor_seed": args.seed if args.donor_mode == "random_residual" else None,
            "clean_margin": clean_margin,
            "corrupt_margin": corrupt_margin,
            "source_effect": clean_margin - corrupt_margin,
            "reversion": int(clean_margin < 0),
            "recovery_by_layer": recovery,
        })
        if (index + 1) % 10 == 0:
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
        "donor_mode": args.donor_mode,
        "n_items": len(rows),
        "n_layers": len(layers),
        "rows": rows,
    }, indent=1) + "\n")
    print(f"saved {args.output}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
