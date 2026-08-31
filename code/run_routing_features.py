"""Extract pre-final source-routing features from OLMo checkpoints.

Only layers through normalized depth 0.75 are retained as predictor features.
The final conflict margin is recorded strictly as the target, never a feature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "pilot"
sys.path.insert(0, str(PILOT))

import gen_items  # noqa: E402
from run_phase_b_srcpatch import build_neutral_pairs  # noqa: E402
from run_patch_srcpos import build_srcpos_pairs, first_id  # noqa: E402
from template_prompts import render_pair, TEMPLATES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokens-b", type=int, required=True)
    parser.add_argument("--template", choices=tuple(TEMPLATES), required=True)
    parser.add_argument("--item-mode", choices=("conflict", "neutral"), default="conflict")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--n-items", type=int, default=120)
    parser.add_argument("--diagnostic-all-layers", action="store_true")
    return parser.parse_args()




@torch.no_grad()
def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.repo, revision=args.revision, cache_dir=args.cache_dir
    )
    base_pairs = (
        build_srcpos_pairs(tokenizer, n_target=args.n_items)
        if args.item_mode == "conflict"
        else build_neutral_pairs(tokenizer, n_target=args.n_items)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.repo,
        revision=args.revision,
        cache_dir=args.cache_dir,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    last_feature_layer = int(np.floor(.75 * (n_layers - 1)))
    extraction_last_layer = n_layers - 1 if args.diagnostic_all_layers else last_feature_layer
    layer_inputs = {}
    handles = []

    def capture(layer: int):
        def hook(_module, inputs):
            layer_inputs[layer] = inputs[0].detach()
        return hook

    for layer in range(max(extraction_last_layer, last_feature_layer + 1) + 1):
        handles.append(
            model.model.layers[layer].register_forward_pre_hook(capture(layer))
        )

    unembed = model.lm_head.weight.detach().float()
    norm_weight = model.model.norm.weight.detach().float()
    rows = []
    skipped = 0
    for pair in base_pairs:
        clean, corrupt, erased, fame = render_pair(
            pair, args.template, args.item_mode
        )
        clean_ids = tokenizer(clean, return_tensors="pt").input_ids
        corrupt_ids = tokenizer(corrupt, return_tensors="pt").input_ids
        if clean_ids.shape != corrupt_ids.shape:
            skipped += 1
            continue
        differences = (clean_ids[0] != corrupt_ids[0]).nonzero().flatten().tolist()
        if len(differences) != 1:
            skipped += 1
            continue
        source_position = differences[0]
        xid = first_id(tokenizer, pair["x_name"])
        yid = first_id(tokenizer, pair["y_name"])
        direction = norm_weight * (unembed[xid] - unembed[yid])

        layer_inputs.clear()
        clean_ids = clean_ids.to("cuda")
        output = model(
            clean_ids, output_attentions=True, output_hidden_states=True,
            use_cache=False
        )
        clean_margin = float(
            output.logits[0, -1, xid] - output.logits[0, -1, yid]
        )
        attention_source = np.empty((extraction_last_layer + 1, n_heads))
        source_path_readout = np.empty_like(attention_source)
        for layer in range(extraction_last_layer + 1):
            attention = output.attentions[layer][0, :, -1, source_position].float()
            hidden_source = layer_inputs[layer][0, source_position]
            value = model.model.layers[layer].self_attn.v_proj(hidden_source)
            value = value.float().reshape(n_heads, head_dim)
            o_weight = model.model.layers[layer].self_attn.o_proj.weight.detach().float()
            scores = []
            for head in range(n_heads):
                head_value = attention[head] * value[head]
                head_slice = slice(head * head_dim, (head + 1) * head_dim)
                contribution = head_value @ o_weight[:, head_slice].T
                scores.append(float(contribution @ direction))
            attention_source[layer] = attention.cpu().numpy()
            source_path_readout[layer] = scores

        prefinal_hidden = layer_inputs[last_feature_layer + 1][0, -1]
        prefinal_logits = model.lm_head(model.model.norm(prefinal_hidden)).float()
        prefinal_margin = float(prefinal_logits[xid] - prefinal_logits[yid])
        clean_logit_lens = []
        for hidden in output.hidden_states[1:]:
            lens_logits = model.lm_head(
                model.model.norm(hidden[0, -1])
            ).float()
            clean_logit_lens.append(float(lens_logits[xid] - lens_logits[yid]))
        erased_ids = tokenizer(erased, return_tensors="pt").input_ids.to("cuda")
        erased_output = model(
            erased_ids, output_hidden_states=True, use_cache=False
        )
        erased_logits = erased_output.logits[0, -1].float()
        erased_margin = float(erased_logits[xid] - erased_logits[yid])
        erased_prefinal_hidden = layer_inputs[last_feature_layer + 1][0, -1]
        erased_prefinal_logits = model.lm_head(
            model.model.norm(erased_prefinal_hidden)
        ).float()
        erased_prefinal_margin = float(
            erased_prefinal_logits[xid] - erased_prefinal_logits[yid]
        )
        erased_logit_lens = []
        for hidden in erased_output.hidden_states[1:]:
            lens_logits = model.lm_head(
                model.model.norm(hidden[0, -1])
            ).float()
            erased_logit_lens.append(float(lens_logits[xid] - lens_logits[yid]))

        rows.append({
            "revision": args.revision,
            "training_tokens_b": args.tokens_b,
            "template": args.template,
            "cat": pair["cat"],
            "fame": fame,
            "subj": pair["subj"],
            "x": pair["x_name"],
            "z": pair["z_name"],
            "y": pair["y_name"],
            "source_position": source_position,
            "clean_margin_target_only": clean_margin,
            "reversion_target": int(clean_margin < 0),
            "erased_margin": erased_margin,
            "prefinal_margin_l11": prefinal_margin,
            "erased_prefinal_margin_l11": erased_prefinal_margin,
            "clean_logit_lens_by_layer": clean_logit_lens,
            "erased_logit_lens_by_layer": erased_logit_lens,
            "attention_to_source": attention_source.tolist(),
            "source_path_readout": source_path_readout.tolist(),
        })
        if len(rows) % 10 == 0:
            print(f"  retained={len(rows)} skipped={skipped}", flush=True)

    for handle in handles:
        handle.remove()
    args.output.write_text(json.dumps({
        "repo": args.repo,
        "revision": args.revision,
        "training_tokens_b": args.tokens_b,
        "template": args.template,
        "item_mode": args.item_mode,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "last_feature_layer": last_feature_layer,
        "extraction_last_layer": extraction_last_layer,
        "diagnostic_all_layers": args.diagnostic_all_layers,
        "rows": rows,
        "skipped": skipped,
    }, indent=1) + "\n")
    print(
        f"saved {args.output}: retained={len(rows)} skipped={skipped} "
        f"reversion={np.mean([row['reversion_target'] for row in rows]):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
