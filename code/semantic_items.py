"""Frozen semantic-axis item builders for the post-replication NLP suite.

The primary identity-axis generator remains untouched.  This module keeps the
meaning extensions separate so an exploratory item bank cannot silently alter
the authoritative results.  Builders are tokenizer-aware: only candidate
answers that occupy one first-token position are retained, and every template
is checked for an exactly-one-token source/corrupt difference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import gen_items


TYPE_CATS: dict[str, dict[str, Any]] = {
    "geography_type": {
        "facts": [
            ("the Nile", "river", 3),
            ("Mount Everest", "mountain", 3),
            ("Paris", "city", 3),
            ("France", "country", 3),
            ("the Sahara", "desert", 3),
            ("Iceland", "island", 2),
            ("the Amazon", "river", 3),
        ],
        "pool": [
            "river", "mountain", "city", "country", "desert", "island", "lake",
            "forest", "valley", "company", "language", "village",
        ],
        "fiction": [
            "the Varnell Archive", "the Keth Vessel", "the Ordan Festival",
            "the Mirethorn Ledger", "the Calden Workshop", "the Yarrow Signal",
        ],
    },
    "culture_type": {
        "facts": [
            ("The Beatles", "band", 3),
            ("Hamlet", "play", 3),
            ("The Odyssey", "poem", 3),
            ("the Mona Lisa", "painting", 3),
            ("The Times", "newspaper", 2),
            ("The Godfather", "film", 3),
        ],
        "pool": [
            "band", "play", "poem", "painting", "newspaper", "film", "novel",
            "opera", "album", "company", "language", "sculpture",
        ],
        "fiction": [
            "the Varnell Ballad", "the Keth Chronicle", "the Ordan Portrait",
            "the Mirethorn Gazette", "the Calden Drama", "the Yarrow Record",
        ],
    },
    "person_type": {
        "facts": [
            ("Bach", "composer", 3),
            ("Einstein", "scientist", 3),
            ("Shakespeare", "author", 3),
            ("Edison", "inventor", 3),
            ("Lincoln", "president", 3),
            ("Darwin", "naturalist", 2),
        ],
        "pool": [
            "composer", "scientist", "author", "inventor", "president", "naturalist",
            "artist", "poet", "engineer", "company", "language", "athlete",
        ],
        "fiction": [
            "the Varnell Scholar", "the Keth Composer", "the Ordan Engineer",
            "the Mirethorn Author", "the Calden Naturalist", "the Yarrow Artist",
        ],
    },
}


REL_CATS: dict[str, dict[str, Any]] = {
    "headquarters": {
        "facts": [
            ("Apple", "Cupertino", 3),
            ("Microsoft", "Redmond", 3),
            ("NATO", "Brussels", 2),
            ("BMW", "Munich", 2),
            ("Samsung", "Suwon", 2),
            ("Airbus", "Toulouse", 2),
        ],
        "pool": [
            "Cupertino", "Redmond", "Brussels", "Munich", "Suwon", "Toulouse",
            "Seattle", "Dublin", "Paris", "London", "Berlin", "Tokyo", "Rome",
        ],
        "fiction": [
            "the Varnell Cooperative", "the Keth Observatory", "the Ordan Guild",
            "the Mirethorn Company", "the Calden Trust", "the Yarrow Bureau",
        ],
    },
    "inventor": {
        "facts": [
            ("the telephone", "Bell", 3),
            ("the light bulb", "Edison", 3),
            ("the steam engine", "Watt", 2),
            ("the printing press", "Gutenberg", 3),
        ],
        "pool": [
            "Bell", "Edison", "Watt", "Gutenberg", "Tesla", "Newton", "Fulton",
            "Morse", "Marconi", "Darwin", "Galileo", "Faraday",
        ],
        "fiction": [
            "the Varnell Engine", "the Keth Recorder", "the Ordan Printer",
            "the Mirethorn Lamp", "the Calden Telegraph", "the Yarrow Device",
        ],
    },
    "birthplace": {
        "facts": [
            ("Einstein", "Ulm", 3),
            ("Shakespeare", "Stratford", 3),
            ("Mozart", "Salzburg", 3),
            ("Napoleon", "Ajaccio", 2),
            ("Gandhi", "Porbandar", 2),
            ("Darwin", "Shrewsbury", 2),
        ],
        "pool": [
            "Ulm", "Stratford", "Salzburg", "Ajaccio", "Porbandar", "Shrewsbury",
            "London", "Paris", "Berlin", "Rome", "Vienna", "Madrid",
        ],
        "fiction": [
            "the Varnell Poet", "the Keth Explorer", "the Ordan Scholar",
            "the Mirethorn Composer", "the Calden Pilot", "the Yarrow Engineer",
        ],
    },
    "currency": {
        "facts": [
            ("Japan", "yen", 3),
            ("Britain", "pound", 3),
            ("India", "rupee", 3),
            ("Russia", "ruble", 2),
            ("Mexico", "peso", 2),
            ("Thailand", "baht", 2),
        ],
        "pool": [
            "yen", "pound", "rupee", "ruble", "peso", "baht", "dollar", "euro",
            "franc", "krone", "won", "dinar",
        ],
        "fiction": [
            "the Varnell Republic", "the Keth Isles", "the Ordan Union",
            "the Mirethorn Kingdom", "the Calden State", "the Yarrow Territory",
        ],
    },
    "language": {
        "facts": [
            ("Brazil", "Portuguese", 3),
            ("Japan", "Japanese", 3),
            ("Germany", "German", 3),
            ("Egypt", "Arabic", 3),
            ("Iran", "Persian", 2),
            ("Italy", "Italian", 3),
        ],
        "pool": [
            "Portuguese", "Japanese", "German", "Arabic", "Persian", "Italian",
            "English", "French", "Spanish", "Russian", "Turkish", "Hindi",
        ],
        "fiction": [
            "the Varnell Region", "the Keth Province", "the Ordan Coast",
            "the Mirethorn Valley", "the Calden Isles", "the Yarrow Basin",
        ],
    },
}


NAT_CATS: dict[str, dict[str, Any]] = {
    "location_context": {
        "facts": list(gen_items.LOC_FACTS),
        "pool": list(gen_items.CITY_POOL),
        "fiction": list(gen_items.LOC_FICTION),
    },
    "authorship_context": {
        "facts": list(gen_items.AUTH_FACTS),
        "pool": list(gen_items.AUTH_POOL),
        "fiction": list(gen_items.AUTH_FICTION),
    },
}


AXIS_CATS: dict[str, dict[str, dict[str, Any]]] = {
    "type": TYPE_CATS,
    "relation": REL_CATS,
    "naturalistic": NAT_CATS,
}


TYPE_WORDING = {
    "A": (
        "The profile classifies {subj} as type {ans}.",
        "According to the profile, the type of {subj} is",
    ),
    "B": (
        "The passage assigns the type {ans} to {subj}.",
        "Based on the passage, the type assigned to {subj} is",
    ),
    "C": (
        "For this account, the category recorded for {subj} is {ans}.",
        "In this account, the category of {subj} is",
    ),
}


REL_WORDING = {
    "headquarters": {
        "A": ("The document states that {subj} has headquarters in {ans}.", "According to the document, the headquarters of {subj} is in"),
        "B": ("The passage lists {ans} as the headquarters city of {subj}.", "Based on the passage, the headquarters city of {subj} is"),
        "C": ("For this account, the headquarters of {subj} are in {ans}.", "In this account, {subj}'s headquarters are in"),
    },
    "inventor": {
        "A": ("The document states that {subj} was invented by {ans}.", "According to the document, {subj} was invented by"),
        "B": ("The passage credits {ans} with inventing {subj}.", "Based on the passage, the inventor of {subj} is"),
        "C": ("For this account, {ans} is named as the inventor of {subj}.", "In this account, the inventor of {subj} is"),
    },
    "birthplace": {
        "A": ("The document states that {subj} was born in {ans}.", "According to the document, {subj} was born in"),
        "B": ("The passage records {ans} as the birthplace of {subj}.", "Based on the passage, the birthplace of {subj} is"),
        "C": ("For this account, {subj}'s birthplace is {ans}.", "In this account, the birthplace of {subj} is"),
    },
    "currency": {
        "A": ("The document states that {subj} uses {ans} as its currency.", "According to the document, the currency used by {subj} is"),
        "B": ("The passage names {ans} as the currency of {subj}.", "Based on the passage, the currency of {subj} is"),
        "C": ("For this account, {ans} is the currency used by {subj}.", "In this account, the currency used by {subj} is"),
    },
    "language": {
        "A": ("The document states that {subj} uses {ans} as its primary language.", "According to the document, the primary language of {subj} is"),
        "B": ("The passage identifies {ans} as the primary language of {subj}.", "Based on the passage, the primary language of {subj} is"),
        "C": ("For this account, the primary language used by {subj} is {ans}.", "In this account, the primary language of {subj} is"),
    },
}


NAT_WORDING = {
    "location_context": {
        "A": (
            "The field note concerns {subj}. It records that {subj} is located in {ans}. The note was filed for review.",
            "According to the field note, {subj} is located in",
        ),
        "B": (
            "A short report reviews {subj}. In its findings, the report lists {ans} as the location of {subj}. The remaining details concern its history.",
            "Based on the report, the location of {subj} is",
        ),
        "C": (
            "The case file describes {subj}. One recorded line reads, 'Location: {ans}.' Other notes discuss the site.",
            "In the case file, the location of {subj} is",
        ),
    },
    "authorship_context": {
        "A": (
            "The field note concerns {subj}. It records that {subj} was written by {ans}. The note was filed for review.",
            "According to the field note, {subj} was written by",
        ),
        "B": (
            "A short report reviews {subj}. In its findings, the report names {ans} as the author of {subj}. The remaining details concern its history.",
            "Based on the report, the author of {subj} is",
        ),
        "C": (
            "The case file describes {subj}. One recorded line reads, 'Author: {ans}.' Other notes discuss the work.",
            "In the case file, the author of {subj} is",
        ),
    },
}


ERASED = {
    "A": "The document does not provide this information.",
    "B": "The passage does not specify this fact.",
    "C": "This account leaves the requested fact unstated.",
}


def is_single(tokenizer: Any, name: str) -> bool:
    ids = tokenizer(" " + name, add_special_tokens=False).input_ids
    return len(ids) == 1


def first_id(tokenizer: Any, name: str) -> int:
    ids = tokenizer(" " + name, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"answer is not one token: {name!r} -> {ids}")
    return int(ids[0])


def _wording(axis: str, cat: str, template: str) -> tuple[str, str]:
    if axis == "type":
        return TYPE_WORDING[template]
    if axis == "relation":
        return REL_WORDING[cat][template]
    return NAT_WORDING[cat][template]


def render_pair(
    pair: dict[str, Any], template: str, item_mode: str
) -> tuple[str, str, str, int]:
    """Render clean/corrupt/erased prompts and return (prompts..., fame)."""
    del item_mode  # retained for a runner-compatible signature
    source, stem = _wording(pair["axis"], pair["cat"], template)
    roster = pair["roster"]

    def with_answer(answer: str) -> str:
        return f"Options: {roster}. {source.format(subj=pair['subj'], ans=answer)}\n{stem.format(subj=pair['subj'])}"

    clean = with_answer(pair["x_name"])
    corrupt = with_answer(pair["z_name"])
    erased = f"Options: {roster}. {ERASED[template]}\n{stem.format(subj=pair['subj'])}"
    return clean, corrupt, erased, int(pair.get("fame", 0))


def _token_diff(tokenizer: Any, a: str, b: str) -> list[int]:
    ids_a = tokenizer(a, return_tensors="pt").input_ids[0]
    ids_b = tokenizer(b, return_tensors="pt").input_ids[0]
    if ids_a.shape != ids_b.shape:
        return [-1]
    return (ids_a != ids_b).nonzero().flatten().tolist()


def validate_pairs(tokenizer: Any, pairs: list[dict[str, Any]]) -> None:
    """Re-check a frozen item file against the active tokenizer/templates."""
    for index, pair in enumerate(pairs):
        for template in ("A", "B", "C"):
            clean, corrupt, _erased, _fame = render_pair(pair, template, pair.get("item_mode", "conflict"))
            diff = _token_diff(tokenizer, clean, corrupt)
            if len(diff) != 1:
                raise ValueError(
                    f"frozen item {index} is not one-token aligned for template {template}: {diff}"
                )
        for name in (pair["x_name"], pair["y_name"], pair["z_name"]):
            if not is_single(tokenizer, str(name)):
                raise ValueError(f"frozen answer is not one token: {name!r}")


def load_or_build_pairs(
    tokenizer: Any,
    axis: str,
    item_mode: str,
    n_target: int,
    seed: int,
    path: Path | None,
) -> list[dict[str, Any]]:
    if path is not None and path.exists():
        pairs = json.loads(path.read_text())
        if len(pairs) != n_target:
            raise ValueError(f"frozen item file has {len(pairs)} rows, expected {n_target}: {path}")
        validate_pairs(tokenizer, pairs)
        return pairs
    pairs = build_pairs(tokenizer, axis, item_mode, n_target, seed=seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pairs, indent=2) + "\n")
    return pairs


def build_pairs(
    tokenizer: Any,
    axis: str,
    item_mode: str,
    n_target: int = 120,
    seed: int = 1301,
) -> list[dict[str, Any]]:
    """Build deterministic tokenizer-valid pairs for one semantic axis."""
    if axis not in AXIS_CATS:
        raise ValueError(f"unknown semantic axis: {axis}")
    if item_mode not in {"conflict", "neutral"}:
        raise ValueError(f"unsupported item mode: {item_mode}")

    rng = np.random.default_rng(seed)
    cats = list(AXIS_CATS[axis])
    pairs: list[dict[str, Any]] = []
    attempts = 0
    while len(pairs) < n_target and attempts < n_target * 500:
        attempts += 1
        cat = cats[int(rng.integers(len(cats)))]
        spec = AXIS_CATS[axis][cat]
        if item_mode == "conflict":
            subj, prior, fame = spec["facts"][int(rng.integers(len(spec["facts"])))]
            y_name = str(prior)
            if not is_single(tokenizer, y_name):
                continue
            subject = str(subj)
        else:
            subject = str(spec["fiction"][int(rng.integers(len(spec["fiction"])))] )
            fame = 0
            valid = [str(name) for name in spec["pool"] if is_single(tokenizer, str(name))]
            if len(valid) < 4:
                continue
            y_name = str(valid[int(rng.integers(len(valid)))])

        valid = [str(name) for name in spec["pool"] if str(name) != y_name and is_single(tokenizer, str(name))]
        if len(valid) < 3:
            continue
        x_name, z_name, filler = [str(x) for x in rng.choice(valid, size=3, replace=False)]
        names = [x_name, y_name, z_name, filler]
        names = [names[i] for i in rng.permutation(len(names))]
        ids = gen_items._ids(rng, len(names))
        roster = ", ".join(f"<{ids[i]}> {names[i]}" for i in range(len(names)))
        pair: dict[str, Any] = {
            "axis": axis,
            "cat": cat,
            "item_mode": item_mode,
            "fame": int(fame),
            "subj": subject,
            "x_name": x_name,
            "z_name": z_name,
            "y_name": y_name,
            "candidate_names": names,
            "candidate_ids": ids,
            "roster": roster,
            "item_seed": int(seed),
        }
        aligned = True
        for template in ("A", "B", "C"):
            clean, corrupt, _erased, _fame = render_pair(pair, template, item_mode)
            if _token_diff(tokenizer, clean, corrupt) == [-1] or len(_token_diff(tokenizer, clean, corrupt)) != 1:
                aligned = False
                break
        if not aligned:
            continue
        pair["item_key"] = f"{axis}:{cat}:{subject}:{y_name}"
        pairs.append(pair)

    if len(pairs) < n_target:
        raise RuntimeError(
            f"could only build {len(pairs)}/{n_target} tokenizer-aligned {axis}/{item_mode} items"
        )
    return pairs
