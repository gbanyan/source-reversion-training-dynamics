"""
prior_law.py — Step 1 of the PORS deepening.

QUESTION: is "Qwen2.5-base reverts to the prior / Llama stays faithful" a real FAMILY
difference in deployment, or merely a difference in PARAMETRIC PRIOR STRENGTH? If reversion
is a single monotone function of prior strength shared across families, the generality limit
becomes a LAW ("deployment failure scales with prior strength") and Llama-faithful is just the
low-prior-strength tail. If Llama reverts LESS at matched prior strength, it is a genuine
family deployment difference.

Per MATCHED conflict/erased pair (same fact, same counterfactual X, same prior Y, same
candidate menu — from gen_items.generate_pairs):
  prior_strength = lp_erased[Y] - mean(lp_erased[fillers])   # pull to the true answer, NO source
  reversion      = argmax(lp_conflict) == Y                  # deploy-failure error WITH source present
  m_conflict     = lp_conflict[X] - lp_conflict[Y]           # decision margin with source
  m_erased       = lp_erased[X]   - lp_erased[Y]             # decision margin without source
  dm             = m_conflict - m_erased                     # source contribution (Tier-1 reported +5.5)

Readout = F2 candidate-string likelihood (length-normalized), identical to sweep.py/run_models.py.
Saves per-item rows to prior_law_rows.json for the one-curve analysis (done locally).

Run with a local Python environment and a CUDA-capable device:
``python prior_law.py [n_per_fame=40]``
"""
import json, sys, gc, os, numpy as np, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import gen_items

LADDER = [
    ("qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B",   0.5, "qwen"),
    ("qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B",   1.5, "qwen"),
    ("qwen2.5-3B",   "Qwen/Qwen2.5-3B",     3,   "qwen"),
    ("qwen2.5-7B",   "Qwen/Qwen2.5-7B",     7,   "qwen"),
    ("qwen2.5-14B",  "Qwen/Qwen2.5-14B",    14,  "qwen"),
    ("llama3.2-1B",  "unsloth/Llama-3.2-1B", 1,  "llama"),
    ("llama3.2-3B",  "unsloth/Llama-3.2-3B", 3,  "llama"),
    ("llama3.1-8B",  "unsloth/Llama-3.1-8B", 8,  "llama"),
]
DEV = "cuda"


@torch.no_grad()
def logps_for(model, tok, stem, cands):
    """Length-normalized logprob of each candidate NAME in the answer slot of `stem`."""
    stem_ids = tok(stem, return_tensors="pt").input_ids.to(DEV)
    base = stem_ids.shape[1]
    out = []
    for name in cands:
        cont = tok(" " + name, add_special_tokens=False).input_ids
        ids = torch.cat([stem_ids, torch.tensor([cont], device=DEV)], 1)
        logits = model(ids).logits[0].float()
        lp = sum(
            torch.log_softmax(logits[base + j - 1], -1)[token].item()
            for j, token in enumerate(cont)
        )
        out.append(lp / max(1, len(cont)))
    return np.asarray(out)


def _cached_snapshot(repo, revision, cache_dir):
    """Resolve a locally materialized Hub snapshot, including manual refs.

    Hugging Face marks a snapshot incomplete when it was assembled manually
    without a cached tree listing.  Passing the snapshot directory directly
    bypasses that metadata check while retaining the normal cache layout.
    This is opt-in via MODEL_LOCAL_SNAPSHOT/TOKENIZER_LOCAL_SNAPSHOT.
    """
    if not revision or not cache_dir:
        return None
    storage = Path(cache_dir) / ("models--" + repo.replace("/", "--"))
    ref = storage / "refs" / revision
    commit = ref.read_text().strip() if ref.is_file() else revision
    snapshot = storage / "snapshots" / commit
    if snapshot.is_dir() and (snapshot / "config.json").exists():
        return str(snapshot)
    return None


def run_one(repo, pairs, *, revision=None, cache_dir=None):
    """Evaluate one repository/revision; optional args preserve old callers."""
    load_kwargs = {"revision": revision, "cache_dir": cache_dir}
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
    # Tokenizer files are shared across checkpoint revisions.  When a runner
    # supplies TOKENIZER_REVISION, use that cached snapshot and avoid one HF
    # Hub HEAD request per checkpoint (which can trigger anonymous rate limits).
    tokenizer_kwargs = {"cache_dir": cache_dir}
    tokenizer_revision = os.environ.get("TOKENIZER_REVISION")
    if tokenizer_revision:
        tokenizer_kwargs["revision"] = tokenizer_revision
    if os.environ.get("TOKENIZER_LOCAL_ONLY"):
        tokenizer_kwargs["local_files_only"] = True
    tokenizer_source = repo
    if os.environ.get("TOKENIZER_LOCAL_SNAPSHOT"):
        tokenizer_source = _cached_snapshot(
            repo, tokenizer_revision or revision, cache_dir
        ) or repo
        if tokenizer_source != repo:
            tokenizer_kwargs = {}
    tok = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    model_source = repo
    if os.environ.get("MODEL_LOCAL_SNAPSHOT"):
        model_source = _cached_snapshot(repo, revision, cache_dir) or repo
        if model_source != repo:
            load_kwargs = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_source, torch_dtype=torch.bfloat16, **load_kwargs
    ).to(DEV).eval()
    rows = []
    for conf, era in pairs:
        cands = conf["f2_candidates"]
        xi = cands.index(conf["x_name"]); yi = cands.index(conf["y_name"])
        fillers = [i for i in range(len(cands)) if i not in (xi, yi)]
        lc = logps_for(model, tok, conf["prompt_F2_stem"], cands)
        le = logps_for(model, tok, era["prompt_F2_stem"], cands)
        rows.append(dict(
            cat=conf["cat"], fame=int(conf["fame"]), subj=conf["subj"],
            x=conf["x_name"], y=conf["y_name"],
            prior_strength=float(le[yi] - np.mean(le[fillers])),
            m_conflict=float(lc[xi] - lc[yi]),
            m_erased=float(le[xi] - le[yi]),
            dm=float((lc[xi] - lc[yi]) - (le[xi] - le[yi])),
            reversion=int(np.argmax(lc) == yi),
            faithful=int(np.argmax(lc) == xi),
        ))
    del model; gc.collect(); torch.cuda.empty_cache()
    return rows


def main():
    n_per_fame = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    pairs = gen_items.generate_pairs(n_per_fame=n_per_fame)
    print(f"pairs={len(pairs)}\n", flush=True)
    print(f"{'model':14s} {'sz':>4s} {'fam':6s} {'reversion':>9s} {'prior_str':>9s} {'dm':>7s}", flush=True)
    out = []
    for tag, repo, sz, fam in LADDER:
        try:
            rows = run_one(repo, pairs)
        except Exception as e:
            print(f"{tag:14s} {sz:>4} {fam:6s}  ERROR {str(e)[:60]}", flush=True)
            continue
        rev = float(np.mean([r["reversion"] for r in rows]))
        ps = float(np.mean([r["prior_strength"] for r in rows]))
        dm = float(np.mean([r["dm"] for r in rows]))
        for r in rows:
            r.update(model=tag, size=sz, family=fam)
        out.extend(rows)
        print(f"{tag:14s} {sz:>4} {fam:6s} {rev:9.2f} {ps:9.2f} {dm:7.2f}", flush=True)
    json.dump(out, open("prior_law_rows.json", "w"), indent=1)
    print(f"\nsaved prior_law_rows.json  ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
