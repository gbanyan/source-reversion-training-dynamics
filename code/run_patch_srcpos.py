"""
run_patch_srcpos.py — Step-2 follow-up (a): SOURCE-POSITION patch, isolating the source
clause from the last-token decision state.

The whole-residual last-token patch in run_patch.py transplants source-info AND downstream
decision state, so it measures where the source-driven SHIFT becomes causal, not a
source-specific component (the PATCH-RESULT.md caveat). This script patches ONLY the
source-answer token position, far upstream of the decision token, so any recovery at the
final token must have PROPAGATED forward from the source clause — a clean source-isolated
causal test.

Design (token-aligned by construction):
  conflict : "Options: <roster>. In this document, <subj> is located in X.  <stem>"   (source = X, counterfactual)
  corrupt  : "Options: <roster>. In this document, <subj> is located in Z.  <stem>"   (source = Z, a neutral filler candidate)
Roster, subject and stem are IDENTICAL; X and Z are both filtered to SINGLE tokens, so the
two prompts differ in EXACTLY ONE token position (the source answer). All positions align ->
patching the source-answer position is unambiguous, no length mismatch.

DENOISING: run the corrupt prompt (source says Z; no support for X), patch the source-answer
token's residual at layer L with the conflict run's residual at that same position, read the
final-token X-vs-Y first-token margin. Recovered margin dL = margin_patched(L) - margin_corrupt.
Compare against the run_patch.py LAST-token patch: if the upstream source-position patch also
recovers margin (especially through the back half), source content propagates causally from its
own position, not just as a last-token decision transplant.

Split error/faithful by whether the full CONFLICT run reverts to the prior Y (margin<0).

Run with a local Python environment and a CUDA-capable device:
``python run_patch_srcpos.py [n_target=60]``
Sanity (tokenizer only, no model): ``python run_patch_srcpos.py --check``
"""
import json, os, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gen_items

REPO = os.environ.get("REPO", "Qwen/Qwen2.5-14B")
TAG = REPO.split("/")[-1].lower()
DEV = "cuda"


def is_single(tok, name):
    return len(tok(" " + name, add_special_tokens=False).input_ids) == 1


def first_id(tok, name):
    return tok(" " + name, add_special_tokens=False).input_ids[0]


def build_srcpos_pairs(tok, n_target=60, K=4, seed=3):
    """Matched (conflict source=X, corrupt source=Z) prompts, token-identical except the
    source-answer position. X, Z single-token, both in the roster, X!=Y!=Z."""
    rng = np.random.default_rng(seed)
    pairs = []
    cats = list(gen_items.CATS)
    while len(pairs) < n_target:
        cat = cats[rng.integers(len(cats))]
        c = gen_items.CATS[cat]
        fame = (3, 2)[rng.integers(2)]
        fb = [f for f in c["facts"] if f[2] == fame]
        subj, Y, _ = fb[rng.integers(len(fb))]
        if not is_single(tok, Y):
            continue
        pool = [a for a in c["pool"] if a != Y and is_single(tok, a)]
        if len(pool) < 3:
            continue
        pick = list(rng.choice(pool, size=3, replace=False))
        X, Z, filler = pick[0], pick[1], pick[2]      # source-X (counterfactual), corrupt-Z, extra filler
        cand_names = [X, Y, Z, filler][:K]
        cand_names = [cand_names[i] for i in rng.permutation(len(cand_names))]
        ids = gen_items._ids(rng, len(cand_names))
        cand = [{"id": ids[i], "name": cand_names[i]} for i in range(len(cand_names))]
        roster = ", ".join(f"<{c0['id']}> {c0['name']}" for c0 in cand)
        stem = c["stem"].format(subj=subj)

        def render(ans):
            src = c["source"].format(subj=subj, ans=ans)
            return f"Options: {roster}. {src}\n{stem}"

        p_conf, p_corr = render(X), render(Z)
        a = tok(p_conf, return_tensors="pt").input_ids[0]
        b = tok(p_corr, return_tensors="pt").input_ids[0]
        if a.shape != b.shape:
            continue
        diff = (a != b).nonzero().flatten().tolist()
        if len(diff) != 1:                            # must differ in exactly one position
            continue
        pairs.append(dict(cat=cat, subj=subj, x_name=X, z_name=Z, y_name=Y,
                          prompt_conf=p_conf, prompt_corr=p_corr, src_pos=diff[0]))
    return pairs


@torch.no_grad()
def resid_at(model, tok, prompt, pos):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model(ids, output_hidden_states=True)
    return [h[0, pos].detach().clone() for h in out.hidden_states], ids


@torch.no_grad()
def margin(model, tok, ids, xid, yid, layer=None, pos=None, donor=None):
    handle = None
    if layer is not None:
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs[:, pos, :] = donor.to(hs.dtype)
            return out
        handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        logits = model(ids).logits[0, -1].float()
    finally:
        if handle:
            handle.remove()
    return float(logits[xid] - logits[yid])


def main():
    args = sys.argv[1:]
    check = "--check" in args
    n_target = next((int(a) for a in args if a.isdigit()), 60)
    tok = AutoTokenizer.from_pretrained(REPO)
    pairs = build_srcpos_pairs(tok, n_target=n_target)
    print(f"{TAG}: built {len(pairs)} token-aligned src-position pairs", flush=True)
    if check:
        p = pairs[0]
        print("EXAMPLE conflict:\n", p["prompt_conf"])
        print("EXAMPLE corrupt :\n", p["prompt_corr"])
        print("src_pos:", p["src_pos"], "X:", p["x_name"], "Z:", p["z_name"], "Y:", p["y_name"])
        cats = {}
        for q in pairs:
            cats[q["cat"]] = cats.get(q["cat"], 0) + 1
        print("cats:", cats)
        return

    model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(DEV).eval()
    nL = model.config.num_hidden_layers
    print(f"{TAG} layers={nL}", flush=True)

    err_curves, faith_curves = [], []
    n_err = n_faith = 0
    for pi, p in enumerate(pairs):
        xid, yid = first_id(tok, p["x_name"]), first_id(tok, p["y_name"])
        sp = p["src_pos"]
        donor, _ = resid_at(model, tok, p["prompt_conf"], sp)          # clean source-X states
        corr_ids = tok(p["prompt_corr"], return_tensors="pt").input_ids.to(DEV)
        m_corr = margin(model, tok, corr_ids, xid, yid)               # corrupt baseline (source=Z)
        m_conf = margin(model, tok,
                        tok(p["prompt_conf"], return_tensors="pt").input_ids.to(DEV), xid, yid)
        is_err = m_conf < 0
        curve = [margin(model, tok, corr_ids, xid, yid, layer=L, pos=sp, donor=donor[L + 1]) - m_corr
                 for L in range(nL)]
        (err_curves if is_err else faith_curves).append(curve)
        n_err += is_err; n_faith += (not is_err)
        if (pi + 1) % 20 == 0:
            print(f"  {pi+1}/{len(pairs)}  err={n_err} faith={n_faith}", flush=True)

    res = dict(repo=REPO, mode="source-answer-position patch (single token, upstream)",
               layers=nL, n_err=int(n_err), n_faith=int(n_faith),
               err_curve=np.mean(err_curves, 0).tolist() if err_curves else [],
               faith_curve=np.mean(faith_curves, 0).tolist() if faith_curves else [])
    out = f"patch_srcpos_{TAG}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nsaved {out}  n_err={n_err} n_faith={n_faith}", flush=True)
    for tag, cv in [("ERROR", res["err_curve"]), ("FAITHFUL", res["faith_curve"])]:
        if cv:
            idx = [int(f * (nL - 1)) for f in (0, .25, .5, .7, .85, 1.0)]
            print(f"  {tag:9s} recovered-margin @layers {idx}: " +
                  " ".join(f"{cv[i]:+.2f}" for i in idx), flush=True)


if __name__ == "__main__":
    main()
