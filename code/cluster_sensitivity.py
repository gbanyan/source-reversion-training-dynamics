#!/usr/bin/env python3
"""Fact-cluster sensitivity checks using already-produced JSON outputs.

The primary analyses use unique item clusters.  This supplementary pass keeps
the estimand fixed but gives every factual subject/answer tuple equal weight,
then bootstraps those fact clusters.  It never runs model inference or changes
the primary summaries.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from analyze_phase_a import peak_excursion


HERE = Path(__file__).resolve().parent
OUT = HERE / "cluster_sensitivity.json"
N_BOOT = 10_000
TRAJECTORY_SEED = 20260831
PATCH_SEED = 20260901


def ci(draws: np.ndarray) -> list[float]:
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.asarray([
        values[rng.integers(0, len(values), len(values))].mean()
        for _ in range(N_BOOT)
    ])


def fact_key(row: dict) -> tuple[str, str, str]:
    return row["cat"], row["subj"], row["y"]


def item_key(row: dict) -> tuple[str, str, str, str]:
    return row["cat"], row["subj"], row["x"], row["y"]


def pair_key(row: dict) -> tuple[str, str, str, str, str]:
    return row["cat"], row["subj"], row["x"], row["z"], row["y"]


def load_rows(names: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        value = json.loads((HERE / name).read_text())
        rows.extend(value if isinstance(value, list) else value["rows"])
    return rows


def trajectory_matrix(rows: list[dict]) -> tuple[list[int], list[tuple], np.ndarray]:
    tokens = sorted({int(row["training_tokens_b"]) for row in rows})
    items = sorted({item_key(row) for row in rows})
    grouped: dict[tuple[int, tuple], list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["training_tokens_b"]), item_key(row)].append(float(row["reversion"]))
    matrix = np.empty((len(items), len(tokens)), dtype=float)
    for i, item in enumerate(items):
        for j, token in enumerate(tokens):
            matrix[i, j] = np.mean(grouped[token, item])
    return tokens, items, matrix


def fact_weighted_matrix(items: list[tuple], matrix: np.ndarray) -> tuple[list[tuple], np.ndarray]:
    by_fact: dict[tuple, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        by_fact[(item[0], item[1], item[3])].append(index)
    facts = sorted(by_fact)
    return facts, np.asarray([matrix[by_fact[fact]].mean(axis=0) for fact in facts])


def trajectory_report(rows: list[dict], seed: int) -> dict:
    tokens, items, item_matrix = trajectory_matrix(rows)
    facts, fact_matrix = fact_weighted_matrix(items, item_matrix)
    curve = fact_matrix.mean(axis=0)
    peak = peak_excursion(curve)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        peak_excursion(
            fact_matrix[rng.integers(0, len(fact_matrix), len(fact_matrix))].mean(axis=0)
        )["excursion"]
        for _ in range(N_BOOT)
    ])
    return {
        "cluster_key": "(category, subject, Y)",
        "n_item_clusters": len(items),
        "n_fact_clusters": len(facts),
        "fact_weighted_curve": [float(value) for value in curve],
        "fact_weighted_excursion": float(peak["excursion"]),
        "fact_weighted_extrema_tokens_b": [
            tokens[peak["before_index"]],
            tokens[peak["peak_index"]],
            tokens[peak["after_index"]],
        ],
        "fact_cluster_bootstrap_95": ci(draws),
        "bootstrap_seed": seed,
        "bootstrap_draws": N_BOOT,
    }


def patch_group(
    early_name: str,
    late_name: str,
    early_token: int,
    late_token: int,
    seed: int,
) -> dict:
    early_rows = load_rows([early_name])
    late_rows = load_rows([late_name])
    early = {pair_key(row): row for row in early_rows}
    late = {pair_key(row): row for row in late_rows}
    keys = sorted(set(early) & set(late))
    if not keys:
        raise ValueError(f"no matched rows: {early_name}, {late_name}")
    early_values = np.asarray([
        np.mean(early[key]["recovery_by_layer"][: len(early[key]["recovery_by_layer"]) * 3 // 4])
        for key in keys
    ], dtype=float)
    late_values = np.asarray([
        np.mean(late[key]["recovery_by_layer"][: len(late[key]["recovery_by_layer"]) * 3 // 4])
        for key in keys
    ], dtype=float)
    delta_values = late_values - early_values
    by_fact: dict[tuple, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        by_fact[(key[0], key[1], key[4])].append(index)
    facts = sorted(by_fact)
    early_fact = np.asarray([early_values[by_fact[fact]].mean() for fact in facts])
    late_fact = np.asarray([late_values[by_fact[fact]].mean() for fact in facts])
    delta_fact = late_fact - early_fact
    rng = np.random.default_rng(seed)
    early_draws = bootstrap_mean(early_fact, rng)
    late_draws = bootstrap_mean(late_fact, rng)
    delta_draws = bootstrap_mean(delta_fact, rng)
    return {
        "early_checkpoint": early_token,
        "late_checkpoint": late_token,
        "n_matched_item_rows": len(keys),
        "n_fact_clusters": len(facts),
        "cluster_key": "(category, subject, Y)",
        "item_weighted": {
            "q_early": float(early_values.mean()),
            "q_late": float(late_values.mean()),
            "delta_q": float(delta_values.mean()),
        },
        "fact_weighted": {
            "q_early": float(early_fact.mean()),
            "q_late": float(late_fact.mean()),
            "delta_q": float(delta_fact.mean()),
            "q_early_bootstrap_95": ci(early_draws),
            "q_late_bootstrap_95": ci(late_draws),
            "delta_q_bootstrap_95": ci(delta_draws),
        },
        "bootstrap_seed": seed,
        "bootstrap_draws": N_BOOT,
    }


def main() -> None:
    one_rows = load_rows(["phase_a_det_shard0.json", "phase_a_det_shard1.json"])
    seven_rows = load_rows(["phase_a_7b_shard0.json", "phase_a_7b_shard1.json"])
    patch_specs = {
        "1B_conflict_A": ("phase_b_det_conflict_2496_A.json", "phase_b_det_conflict_2999_A.json", 2496, 2999),
        "1B_conflict_B": ("phase_b_det_conflict_2496_B.json", "phase_b_det_conflict_2999_B.json", 2496, 2999),
        "1B_conflict_C": ("phase_b_det_conflict_2496_C.json", "phase_b_det_conflict_2999_C.json", 2496, 2999),
        "1B_neutral_A": ("phase_b_det_neutral_2496_A.json", "phase_b_det_neutral_2999_A.json", 2496, 2999),
        "1B_neutral_B": ("phase_b_det_neutral_2496_B.json", "phase_b_det_neutral_2999_B.json", 2496, 2999),
        "1B_neutral_C": ("phase_b_det_neutral_2496_C.json", "phase_b_det_neutral_2999_C.json", 2496, 2999),
        "7B_conflict_A": ("phase_b_7b_conflict_3201_A.json", "phase_b_7b_conflict_3896_A.json", 3201, 3896),
        "7B_conflict_B": ("phase_b_7b_conflict_3201_B.json", "phase_b_7b_conflict_3896_B.json", 3201, 3896),
        "7B_neutral_A": ("phase_b_7b_neutral_3201_A.json", "phase_b_7b_neutral_3896_A.json", 3201, 3896),
        "7B_neutral_B": ("phase_b_7b_neutral_3201_B.json", "phase_b_7b_neutral_3896_B.json", 3201, 3896),
    }
    patches = {}
    for index, (name, (early, late, early_token, late_token)) in enumerate(patch_specs.items()):
        patches[name] = patch_group(early, late, early_token, late_token, PATCH_SEED + index)
    result = {
        "status": "FACT-CLUSTER-SENSITIVITY",
        "scope": "supplementary sensitivity using existing canonical raw outputs; no model inference",
        "cluster_key": "(category, subject, Y)",
        "trajectory": {
            "1B": trajectory_report(one_rows, TRAJECTORY_SEED),
            "7B": trajectory_report(seven_rows, TRAJECTORY_SEED + 1),
        },
        "patch": patches,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
