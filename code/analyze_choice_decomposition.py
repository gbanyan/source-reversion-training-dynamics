"""Derive candidate-choice decomposition from the audited trajectory rows.

The Phase-A runners retain whether the source candidate (``faithful``) or the
parametric candidate (``reversion``) is the top-scoring option.  Because the
candidate roster also contains distractors, the remaining mass is a useful
explicit ``other`` category rather than an assumed complement of either
binary outcome.  This script applies the same unique-item aggregation used by
``analyze_phase_a.py`` and writes a deterministic, machine-readable summary.

This is a derived analysis only: raw model outputs are not changed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def item_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["cat"]),
        str(row["subj"]),
        str(row["x"]),
        str(row["y"]),
    )


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"expected a row list: {path}")
        rows.extend(payload)
    if not rows:
        raise ValueError("no rows supplied")
    required = {"training_tokens_b", "faithful", "reversion", "cat", "subj", "x", "y"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"rows are missing fields: {sorted(missing)}")
    if any(int(row["faithful"]) + int(row["reversion"]) > 1 for row in rows):
        raise ValueError("a conflict row cannot be both faithful and reversion")
    return rows


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[int], list[tuple[str, str, str, str]], np.ndarray]:
    tokens = sorted({int(row["training_tokens_b"]) for row in rows})
    keys = sorted({item_key(row) for row in rows})
    grouped: dict[tuple[int, tuple[str, str, str, str]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["training_tokens_b"]), item_key(row)].append(row)
    values = np.empty((len(keys), len(tokens), 3), dtype=float)
    for i, key in enumerate(keys):
        for j, token in enumerate(tokens):
            grouped_rows = grouped[token, key]
            if not grouped_rows:
                raise ValueError(f"missing item/token cell: {key} at {token}B")
            px = float(np.mean([int(row["faithful"]) for row in grouped_rows]))
            py = float(np.mean([int(row["reversion"]) for row in grouped_rows]))
            values[i, j] = (px, py, 1.0 - px - py)
    return tokens, keys, values


def ci_for(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = rng.integers(0, values.shape[0], size=values.shape[0])
        means[draw] = float(values[sample].mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def transition(values: np.ndarray, tokens: list[int], before: int, after: int,
               seed: int, draws: int) -> dict[str, Any]:
    if before not in tokens or after not in tokens:
        raise ValueError(f"transition {before}->{after} is not in the trajectory")
    before_i, after_i = tokens.index(before), tokens.index(after)
    delta = values[:, after_i, :] - values[:, before_i, :]
    result: dict[str, Any] = {
        "before_tokens_b": before,
        "after_tokens_b": after,
        "delta": [float(x) for x in delta.mean(axis=0)],
        "bootstrap_95": {},
    }
    for metric_i, name in enumerate(("p_x", "p_y", "p_other")):
        result["bootstrap_95"][name] = ci_for(delta[:, metric_i], seed + metric_i, draws)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    rows = load_rows(args.rows)
    tokens, keys, values = aggregate(rows)
    means = values.mean(axis=0)
    point_ci: dict[str, list[list[float]]] = {"p_x": [], "p_y": [], "p_other": []}
    for j in range(len(tokens)):
        for i, name in enumerate(point_ci):
            point_ci[name].append(ci_for(values[:, j, i], args.seed + j * 3 + i, args.draws))
    transitions = {}
    for before, after in ((63, 1909), (1909, 1993), (2496, 2999), (1901, 3201), (3201, 3896)):
        if before in tokens and after in tokens:
            transitions[f"{before}B_to_{after}B"] = transition(
                values, tokens, before, after, args.seed + before + after, args.draws
            )
    result = {
        "status": "DERIVED-CHOICE-DECOMPOSITION",
        "row_sources": [str(path) for path in args.rows],
        "n_raw_rows": len(rows),
        "n_unique_items": len(keys),
        "tokens_b": tokens,
        "aggregation": "mean within unique (cat, subject, X, Y) item clusters",
        "definitions": {
            "p_x": "top-scoring explicit source candidate X (faithful flag)",
            "p_y": "top-scoring parametric candidate Y (reversion flag)",
            "p_other": "top-scoring distractor; 1 - p_x - p_y",
        },
        "curves": {
            "p_x": [float(x) for x in means[:, 0]],
            "p_y": [float(x) for x in means[:, 1]],
            "p_other": [float(x) for x in means[:, 2]],
        },
        "point_bootstrap_95": point_ci,
        "transitions": transitions,
        "bootstrap": {"draws": args.draws, "seed": args.seed, "unit": "unique item cluster"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
