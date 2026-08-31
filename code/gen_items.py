"""
gen_items.py — counterfactual knowledge-conflict entity-selection items (pilot SPEC v3).

CPU / local. The earlier in-context-priming design produced ~0 source-unfaithful errors
(models lock onto the explicit source clause; arbitrary names carry no prior). Fix: use a
STRONG PARAMETRIC PRIOR (a famous fact) contradicted by a source-stated COUNTERFACTUAL.
Models reliably revert to the parametric prior there (the knowledge-conflict phenomenon),
so errors exist — and it is still a SOURCE-FAITHFULNESS test (the faithful answer = what the
passage states; a human reports the stated value even when counterfactual).

Item: a fact (subject, relation, true answer T). Candidates are SAME-TYPE entities, each
tagged with a per-item RANDOMIZED id; ALL tagged incl. the prior-favored decoy.
  conflict (main)   : source states a COUNTERFACTUAL X (!=T); prior favors Y=T.  faithful=X, error=Y
  agree (pull_to_x) : source states the TRUE answer T;        prior favors Y=T.  faithful=prior=T (ceiling)
  source_erased     : NO source statement; only the prior.    measures prior strength (decorrelator)
  neutral (capacity): FICTIONAL subject (no prior); source states X.            faithful=X (pure source)

Two answer formats: F1 (ID-pointer) and F2 (candidate-string likelihood, no menu).
"""
import numpy as np

# ---- fact banks (strong, single-token-ish answers; same-type candidate pools) -------------
LOC_FACTS = [  # (landmark, true city, fame 2-3)
    ("the Eiffel Tower", "Paris", 3), ("the Colosseum", "Rome", 3),
    ("Big Ben", "London", 3), ("the Brandenburg Gate", "Berlin", 3),
    ("the Acropolis", "Athens", 3), ("the Sagrada Familia", "Barcelona", 3),
    ("the Kremlin", "Moscow", 3), ("the Charles Bridge", "Prague", 2),
    ("Tower Bridge", "London", 3), ("the Pantheon", "Rome", 2),
    ("the Louvre", "Paris", 3), ("the Reichstag", "Berlin", 2),
    ("the Prado Museum", "Madrid", 2), ("Schonbrunn Palace", "Vienna", 2),
    ("the Blue Mosque", "Istanbul", 2), ("the Parthenon", "Athens", 3),
]
CITY_POOL = ["Paris", "Rome", "London", "Berlin", "Athens", "Barcelona", "Moscow",
             "Prague", "Madrid", "Vienna", "Istanbul", "Lisbon", "Dublin", "Oslo", "Cairo"]
LOC_FICTION = ["the Vorlon Spire", "the Keth Monument", "the Aldwin Arch",
               "the Brennon Obelisk", "the Yarrow Gate", "the Calden Tower"]

AUTH_FACTS = [  # (work, true author, fame)
    ("Romeo and Juliet", "Shakespeare", 3), ("Hamlet", "Shakespeare", 3),
    ("War and Peace", "Tolstoy", 3), ("Pride and Prejudice", "Austen", 3),
    ("The Odyssey", "Homer", 3), ("Nineteen Eighty-Four", "Orwell", 3),
    ("Don Quixote", "Cervantes", 3), ("The Divine Comedy", "Dante", 3),
    ("Crime and Punishment", "Dostoevsky", 3), ("Faust", "Goethe", 2),
    ("The Trial", "Kafka", 2), ("Ulysses", "Joyce", 2),
]
AUTH_POOL = ["Shakespeare", "Tolstoy", "Austen", "Homer", "Orwell", "Cervantes",
             "Dante", "Dostoevsky", "Goethe", "Kafka", "Joyce", "Dickens", "Hugo", "Twain"]
AUTH_FICTION = ["the novel Karenwald", "the play Mirethorn", "the poem Valdris",
                "the novel Helstrom", "the play Auberon", "the novel Drennick"]

CATS = {
    "loc": dict(facts=LOC_FACTS, pool=CITY_POOL, fiction=LOC_FICTION,
                source="In this document, {subj} is located in {ans}.",
                stem="According to the document, {subj} is located in",
                query="According to the document, in which city is {subj} located?"),
    "auth": dict(facts=AUTH_FACTS, pool=AUTH_POOL, fiction=AUTH_FICTION,
                 source="In this document, {subj} was written by {ans}.",
                 stem="According to the document, {subj} was written by",
                 query="According to the document, who wrote {subj}?"),
}
SOURCE_ERASED = "The document does not provide this information."

ID_LETTERS = list("abcdefghjkmnpqrtvwxyz"); ID_DIGITS = list("234678")


def _ids(rng, k):
    # Preserve draw order.  Returning list(set(...)) made tag-to-candidate
    # assignment depend on Python's per-process hash seed, so nominally frozen
    # items changed across checkpoint workers.
    seen = set()
    ordered = []
    while len(ordered) < k:
        candidate = rng.choice(ID_LETTERS) + rng.choice(ID_DIGITS)
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def build_item(rng, cat, K=4, variant="conflict", fame=None):
    c = CATS[cat]
    if variant == "neutral":
        subj = rng.choice(c["fiction"]); true_ans = None
    else:
        fb = [f for f in c["facts"] if (fame is None or f[2] == fame)]
        subj, true_ans, fm = fb[rng.integers(len(fb))]
    pool = [a for a in c["pool"] if a != true_ans]

    # choose answers
    if variant == "agree":
        X = true_ans                               # source states the TRUE answer (no conflict)
    else:  # conflict / neutral / source_erased: X is a distinct counterfactual / arbitrary city
        X = rng.choice(pool)
    Y = true_ans if true_ans else rng.choice([a for a in pool if a != X])  # prior decoy
    specials = [X] if X == Y else [X, Y]           # agree: faithful==prior == one entity
    fillers = list(rng.choice([a for a in pool if a not in (X, Y)],
                              size=max(0, K - len(specials)), replace=False))
    answers = specials + fillers
    answers = [answers[i] for i in rng.permutation(len(answers))]   # randomize positions
    ids = _ids(rng, len(answers))
    cand = [{"id": ids[i], "name": answers[i]} for i in range(len(answers))]
    xid = next(c0["id"] for c0 in cand if c0["name"] == X)
    yid = next(c0["id"] for c0 in cand if c0["name"] == Y)

    roster = ", ".join(f"<{c0['id']}> {c0['name']}" for c0 in cand)
    if variant == "source_erased":
        src = SOURCE_ERASED; faithful = None
    else:
        src = c["source"].format(subj=subj, ans=X); faithful = X
    body = f"Options: {roster}. {src}"
    query = c["query"].format(subj=subj) + \
        f" Reply with exactly one tag from {{{', '.join('<'+c0['id']+'>' for c0 in cand)}}}."
    return {
        "cat": cat, "variant": variant, "fame": (fame or 0), "subj": subj, "true_ans": true_ans,
        "candidates": cand, "x_name": X, "x_id": xid, "y_name": Y, "y_id": yid,
        "faithful_id": (xid if faithful else None), "faithful_name": faithful,
        "prior_id": yid, "prior_name": Y,
        "passage": body,
        "prompt_F1": body + "\n" + query + "\nAnswer: <",
        "prompt_F2_stem": body + "\n" + c["stem"].format(subj=subj),
        "f2_candidates": [c0["name"] for c0 in cand],
        "f2_faithful": faithful, "f2_prior": Y,
    }


def generate(n_per_cell=8, K=4, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for cat in CATS:
        for variant in ("conflict", "agree", "source_erased", "neutral"):
            fames = (3, 2) if variant != "neutral" else (None,)
            for fame in fames:
                for _ in range(n_per_cell):
                    items.append(build_item(rng, cat, K=K, variant=variant, fame=fame))
    rng.shuffle(items)
    return items


def build_pair(rng, cat, K=4, fame=3):
    """A MATCHED conflict/erased pair: same fact, candidates, X (counterfactual), Y (prior).
    Used for the Tier-1 source-presence margin-shift (delta-m) test."""
    c = CATS[cat]
    fb = [f for f in c["facts"] if f[2] == fame]
    subj, true_ans, _ = fb[rng.integers(len(fb))]
    pool = [a for a in c["pool"] if a != true_ans]
    X = rng.choice(pool)                       # counterfactual source answer
    Y = true_ans                               # prior
    fillers = list(rng.choice([a for a in pool if a != X], size=max(0, K - 2), replace=False))
    answers = [X, Y] + fillers
    answers = [answers[i] for i in rng.permutation(len(answers))]
    ids = _ids(rng, len(answers))
    cand = [{"id": ids[i], "name": answers[i]} for i in range(len(answers))]
    roster = ", ".join(f"<{c0['id']}> {c0['name']}" for c0 in cand)
    stem = c["stem"].format(subj=subj)

    def render(src_present):
        src = c["source"].format(subj=subj, ans=X) if src_present else SOURCE_ERASED
        return dict(cat=cat, variant=("conflict" if src_present else "source_erased"),
                    fame=fame, subj=subj, x_name=X, y_name=Y,
                    f2_candidates=[c0["name"] for c0 in cand],
                    f2_faithful=(X if src_present else None), f2_prior=Y,
                    prompt_F2_stem=f"Options: {roster}. {src}\n{stem}")
    return render(True), render(False)


def generate_pairs(n_per_fame=20, K=4, seed=1):
    """Matched conflict/erased pairs across cats x fame for the delta-m test."""
    rng = np.random.default_rng(seed)
    pairs = []
    for cat in CATS:
        for fame in (3, 2):
            for _ in range(n_per_fame):
                pairs.append(build_pair(rng, cat, K=K, fame=fame))
    rng.shuffle(pairs)
    return pairs


def check_invariants(items):
    import collections
    fails = []

    def chk(c, m):
        (None if c else fails.append(m))

    for it in items:
        ids = [c["id"] for c in it["candidates"]]; nm = [c["name"] for c in it["candidates"]]
        chk(len(set(ids)) == len(ids), "dup id"); chk(len(set(nm)) == len(nm), "dup name")
        chk(it["y_id"] in ids and it["x_id"] in ids, "X/Y not tagged")
        if it["variant"] == "conflict":
            chk(it["x_name"] != it["y_name"], "conflict X==Y (no conflict)")
            chk(it["y_name"] == it["true_ans"], "prior Y != true answer")
        if it["variant"] == "source_erased":
            chk(it["faithful_id"] is None, "erased has faithful")
        else:
            chk(it["faithful_id"] is not None, "missing faithful")
    fids = [it["faithful_id"] for it in items if it["faithful_id"]]
    chk(len(set(fids)) > 8, "faithful id not randomized")
    pos = [next(i for i, c in enumerate(it["candidates"]) if c["id"] == it["x_id"]) for it in items]
    chk(0.10 < np.mean([p == 0 for p in pos]) < 0.45, "X position not ~uniform")
    vc = collections.Counter(it["variant"] for it in items)
    for v in ("conflict", "agree", "source_erased", "neutral"):
        chk(vc[v] > 0, f"variant {v} missing")
    return fails


if __name__ == "__main__":
    import collections
    items = generate(n_per_cell=8)
    fails = check_invariants(items)
    print(f"gen_items — {len(items)} items; variants:",
          dict(collections.Counter(it['variant'] for it in items)),
          "; cats:", dict(collections.Counter(it['cat'] for it in items)))
    for v in ("conflict", "source_erased", "neutral"):
        s = next(it for it in items if it["variant"] == v)
        print(f"\n--- {v} ({s['cat']}) ---\n{s['prompt_F1']}")
        print(f"  [faithful={s['faithful_name']} <{s['faithful_id']}>  prior/decoy={s['prior_name']} <{s['prior_id']}>]")
    print(f"\nINVARIANTS: {'PASS' if not fails else 'FAIL ' + str(fails)}")
    raise SystemExit(0 if not fails else 1)
