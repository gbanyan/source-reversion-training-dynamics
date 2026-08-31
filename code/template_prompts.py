"""Shared deterministic A/B/C rendering for conflict and neutral probes."""

from __future__ import annotations

import gen_items


TEMPLATES = {
    "A": {
        "loc_source": "In this document, {subj} is located in {ans}.",
        "loc_stem": "According to the document, {subj} is located in",
        "auth_source": "In this document, {subj} was written by {ans}.",
        "auth_stem": "According to the document, {subj} was written by",
        "erased": "The document does not provide this information.",
    },
    "B": {
        "loc_source": "The passage identifies {ans} as the location of {subj}.",
        "loc_stem": "Based on the passage, the location of {subj} is",
        "auth_source": "The passage attributes {subj} to {ans}.",
        "auth_stem": "Based on the passage, {subj} is attributed to",
        "erased": "The passage does not specify this fact.",
    },
    "C": {
        "loc_source": "For this account, {subj} can be found in {ans}.",
        "loc_stem": "In this account, {subj} can be found in",
        "auth_source": "For this account, {ans} is the author of {subj}.",
        "auth_stem": "In this account, the author of {subj} is",
        "erased": "This account leaves the requested fact unstated.",
    },
}


def fame_for(cat: str, subj: str) -> int:
    return next(
        fame for subject, _answer, fame in gen_items.CATS[cat]["facts"]
        if subject == subj
    )


def roster_prefix(prompt: str) -> str:
    return prompt.split(". In this document,", 1)[0]


def render_pair(pair: dict, family: str, item_mode: str) -> tuple[str, str, str, int]:
    spec = TEMPLATES[family]
    cat = pair["cat"]
    source_pattern = spec[f"{cat}_source"]
    stem_pattern = spec[f"{cat}_stem"]
    prefix = roster_prefix(pair["prompt_conf"])
    stem = stem_pattern.format(subj=pair["subj"])

    def with_answer(answer: str) -> str:
        source = source_pattern.format(subj=pair["subj"], ans=answer)
        return f"{prefix}. {source}\n{stem}"

    clean = with_answer(pair["x_name"])
    corrupt = with_answer(pair["z_name"])
    erased = f"{prefix}. {spec['erased']}\n{stem}"
    fame = fame_for(cat, pair["subj"]) if item_mode == "conflict" else 0
    return clean, corrupt, erased, fame
