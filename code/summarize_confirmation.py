#!/usr/bin/env python3
"""Summarize a frozen Pythia source-patch confirmation suite.

The summary keeps baseline clean--corrupt margins separate from the causal
patch recovery.  Item-bootstrap intervals are deterministic and are used only
for descriptive, checkpoint-selected confirmation reporting.
"""

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


def bootstrap_mean(values: list[float], seed: int, draws: int = 4000) -> list[float]:
    if not values:
        return []
    rng = random.Random(seed)
    n = len(values)
    return [mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(draws)]


def ci95(values: list[float], seed: int) -> list[float]:
    boot = bootstrap_mean(values, seed)
    return [quantile(boot, 0.025), quantile(boot, 0.975)]


def summarize_source(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    effects = [float(row["source_effect"]) for row in rows]
    matrix = [[float(value) for value in row["recovery_by_layer"]] for row in rows]
    n_layers = len(matrix[0]) if matrix else 0
    layer_means = [mean([row[layer] for row in matrix]) for layer in range(n_layers)]
    layer_cis = [
        ci95([row[layer] for row in matrix], 1000 + layer)
        for layer in range(n_layers)
    ]
    max_layer = max(range(n_layers), key=lambda layer: layer_means[layer]) if n_layers else None
    return {
        "file": str(path),
        "repo": payload.get("repo"),
        "revision": payload["revision"],
        "training_tokens_b": payload["training_tokens_b"],
        "training_axis": payload.get("training_axis", payload["training_tokens_b"]),
        "axis_unit": payload.get("axis_unit", "training_tokens_b"),
        "template": payload["template"],
        "item_mode": payload["item_mode"],
        "donor_mode": rows[0].get("donor_mode", "source") if rows else "source",
        "n": len(rows),
        "n_layers": n_layers,
        "source_effect_mean": mean(effects),
        "source_effect_ci95": ci95(effects, 77),
        "clean_margin_mean": mean([float(row["clean_margin"]) for row in rows]),
        "corrupt_margin_mean": mean([float(row["corrupt_margin"]) for row in rows]),
        "reversion_rate": mean([float(row["reversion"]) for row in rows]),
        "recovery_by_layer_mean": layer_means,
        "recovery_by_layer_ci95": layer_cis,
        "recovery_max_layer": max_layer,
        "recovery_max_mean": layer_means[max_layer] if max_layer is not None else float("nan"),
        "recovery_max_ci95": layer_cis[max_layer] if max_layer is not None else [float("nan"), float("nan")],
        "recovery_final_mean": layer_means[-1] if layer_means else float("nan"),
    }


def summarize_corrupt(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    recoveries = [float(value) for row in rows for value in row["recovery_by_layer"]]
    return {
        "file": str(path),
        "repo": payload.get("repo"),
        "revision": payload["revision"],
        "training_tokens_b": payload["training_tokens_b"],
        "training_axis": payload.get("training_axis", payload["training_tokens_b"]),
        "axis_unit": payload.get("axis_unit", "training_tokens_b"),
        "template": payload["template"],
        "item_mode": payload["item_mode"],
        "donor_mode": rows[0].get("donor_mode", "corrupt") if rows else "corrupt",
        "n": len(rows),
        "recovery_max_abs": max((abs(value) for value in recoveries), default=0.0),
        "recovery_mean_abs": mean([abs(value) for value in recoveries]),
    }


def paired_contrast(conflict: dict, neutral: dict) -> dict:
    c = json.loads(Path(conflict["file"]).read_text())["rows"]
    n = json.loads(Path(neutral["file"]).read_text())["rows"]
    if len(c) != len(n):
        raise ValueError(f"paired row count mismatch: {conflict['file']} vs {neutral['file']}")
    deltas = [float(a["source_effect"]) - float(b["source_effect"]) for a, b in zip(c, n)]
    return {
        "revision": conflict["revision"],
        "training_tokens_b": conflict["training_tokens_b"],
        "training_axis": conflict.get("training_axis", conflict["training_tokens_b"]),
        "axis_unit": conflict.get("axis_unit", "training_tokens_b"),
        "template": conflict["template"],
        "n": len(deltas),
        "conflict_minus_neutral_source_effect_mean": mean(deltas),
        "ci95": ci95(deltas, 9000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", default="CONFIRMATION")
    args = parser.parse_args()

    source_files = sorted(args.patch_dir.glob("*_source.json"))
    corrupt_files = sorted(args.patch_dir.glob("*_corrupt.json"))
    source_summaries = [summarize_source(path) for path in source_files]
    corrupt_summaries = [summarize_corrupt(path) for path in corrupt_files]

    by_key = {
        (row["revision"], row["template"], row["item_mode"]): row
        for row in source_summaries
    }
    contrasts = []
    for revision, template, mode in sorted(by_key):
        if mode == "conflict" and (revision, template, "neutral") in by_key:
            contrasts.append(
                paired_contrast(
                    by_key[(revision, template, "conflict")],
                    by_key[(revision, template, "neutral")],
                )
            )

    result = {
        "status": args.status,
        "source_files": len(source_files),
        "corrupt_files": len(corrupt_files),
        "source_summaries": source_summaries,
        "corrupt_summaries": corrupt_summaries,
        "conflict_minus_neutral": contrasts,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
