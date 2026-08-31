"""A/B test the batched candidate scorer against the original scalar scorer."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot"))
from prior_law import DEV, logps_for
import gen_items


@torch.no_grad()
def scalar_logps_for(model, tok, stem, cands):
    stem_ids = tok(stem, return_tensors="pt").input_ids.to(DEV)
    base = stem_ids.shape[1]
    values = []
    for name in cands:
        cont = tok(" " + name, add_special_tokens=False).input_ids
        ids = torch.cat([stem_ids, torch.tensor([cont], device=DEV)], 1)
        logits = model(ids).logits[0].float()
        score = sum(
            torch.log_softmax(logits[base + j - 1], -1)[token].item()
            for j, token in enumerate(cont)
        )
        values.append(score / max(1, len(cont)))
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="EleutherAI/pythia-1.4b-deduped")
    parser.add_argument("--revision", default="step143000")
    parser.add_argument("--cache-dir")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    kwargs = {"revision": args.revision, "cache_dir": args.cache_dir}
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    tok = AutoTokenizer.from_pretrained(args.repo, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.repo, torch_dtype=torch.bfloat16, **kwargs
    ).to(DEV).eval()
    pairs = gen_items.generate_pairs(n_per_fame=max(1, (args.n + 3) // 4))[:args.n]
    diffs = []
    for conf, _erased in pairs:
        stem = conf["prompt_F2_stem"]
        cands = conf["f2_candidates"]
        scalar = scalar_logps_for(model, tok, stem, cands)
        batched = logps_for(model, tok, stem, cands)
        diffs.extend(np.abs(scalar - batched).tolist())
    print({
        "pairs": len(pairs),
        "max_abs": max(diffs),
        "mean_abs": statistics.mean(diffs),
        "within_1e-4": max(diffs) <= 1e-4,
    })


if __name__ == "__main__":
    main()
