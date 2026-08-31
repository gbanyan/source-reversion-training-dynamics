#!/usr/bin/env python3
"""Summarize layerwise source-token patch results with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_mean(values: list[float], seed: int, draws: int = 2000) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    return [mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(draws)]


def ci(values: list[float], seed: int) -> list[float]:
    boot = bootstrap_mean(values, seed)
    return [quantile(boot, 0.025), quantile(boot, 0.975)]


def summarize_file(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    effects = [float(row["source_effect"]) for row in rows]
    matrix = [row["recovery_by_layer"] for row in rows]
    n_layers = len(matrix[0]) if matrix else 0
    layer_mean = [mean([float(row[layer]) for row in matrix]) for layer in range(n_layers)]
    layer_ci = [
        ci([float(row[layer]) for row in matrix], 1000 + layer)
        for layer in range(n_layers)
    ]
    return {
        "file": str(path),
        "revision": payload["revision"],
        "training_tokens_b": payload["training_tokens_b"],
        "template": payload["template"],
        "item_mode": payload["item_mode"],
        "n": len(rows),
        "n_layers": n_layers,
        "source_effect_mean": mean(effects),
        "source_effect_ci95": ci(effects, 77),
        "recovery_by_layer_mean": layer_mean,
        "recovery_by_layer_ci95": layer_ci,
        "recovery_final_mean": layer_mean[-1] if layer_mean else float("nan"),
        "recovery_max_mean": max(layer_mean) if layer_mean else float("nan"),
        "clean_margin_mean": mean([float(row["clean_margin"]) for row in rows]),
        "corrupt_margin_mean": mean([float(row["corrupt_margin"]) for row in rows]),
    }


def paired_contrast(conflict: dict, neutral: dict) -> dict:
    # Both files are generated from the same deterministic item order.
    c = json.loads(Path(conflict["file"]).read_text())["rows"]
    n = json.loads(Path(neutral["file"]).read_text())["rows"]
    if len(c) != len(n):
        raise ValueError(f"paired row count mismatch: {conflict['file']} vs {neutral['file']}")
    deltas = [float(a["source_effect"]) - float(b["source_effect"]) for a, b in zip(c, n)]
    return {
        "revision": conflict["revision"],
        "training_tokens_b": conflict["training_tokens_b"],
        "template": conflict["template"],
        "n": len(deltas),
        "conflict_minus_neutral_source_effect_mean": mean(deltas),
        "ci95": ci(deltas, 9000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.patch_dir.glob("pythia1p4_step*_patch_*.json"))
    files = [path for path in files if len(json.loads(path.read_text()).get("rows", [])) == 60]
    summaries = [summarize_file(path) for path in files]
    by_key = {(row["revision"], row["template"], row["item_mode"]): row for row in summaries}
    contrasts = []
    for revision, template, mode in sorted(by_key):
        if mode != "conflict":
            continue
        neutral = by_key.get((revision, template, "neutral"))
        if neutral is not None:
            contrasts.append(paired_contrast(by_key[(revision, template, mode)], neutral))
    result = {"summaries": summaries, "conflict_minus_neutral": contrasts}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
