"""Summarize semantic-axis behavior and source-patch JSON files.

This is descriptive and checkpoint-selected; it does not turn checkpoints into
independent statistical replicates. Bootstrap resampling is over the frozen
semantic ``item_key`` carried by each runner row (``axis:category:subject:Y``),
with deterministic halves based on the same key hash. The fallback key is kept
for legacy rows that predate explicit semantic keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def item_key(row: dict[str, Any]) -> str:
    """Return the frozen semantic fact/candidate key used for clustering."""
    return str(row.get("item_key") or f"{row.get('cat')}:{row.get('subj')}:{row.get('x')}:{row.get('y')}")


def cluster_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[item_key(row)].append(float(row[field]))
    return [mean(values) for values in grouped.values()]


def bootstrap(values: list[float], seed: int, draws: int = 4000) -> list[float]:
    if not values:
        return []
    # A tiny local LCG keeps this script dependency-free and deterministic.
    state = seed & 0xFFFFFFFF
    out: list[float] = []
    for _ in range(draws):
        sample: list[float] = []
        for _ in values:
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            sample.append(values[state % len(values)])
        out.append(mean(sample))
    return out


def ci95(values: list[float], seed: int) -> list[float]:
    draws = bootstrap(values, seed)
    return [quantile(draws, 0.025), quantile(draws, 0.975)]


def half(row: dict[str, Any]) -> int:
    return int(hashlib.sha256(item_key(row).encode()).hexdigest(), 16) % 2


def load_jsons(directory: Path, suffix: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        payloads.append(json.loads(path.read_text()))
    return payloads


def behavior_summary(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    rows = payload.get("rows", [])
    return {
        "file": str(path),
        "repo": payload.get("repo"),
        "axis": payload.get("axis"),
        "revision": payload.get("revision"),
        "training_axis": payload.get("training_axis", payload.get("training_tokens_b")),
        "axis_unit": payload.get("axis_unit", "training_tokens_b"),
        "template": payload.get("template"),
        "item_mode": payload.get("item_mode"),
        "n": len(rows),
        "error_mean": mean(cluster_values(rows, "reversion")),
        "error_ci95": ci95(cluster_values(rows, "reversion"), 100),
        "source_gain_mean": mean(cluster_values(rows, "source_gain")),
        "source_gain_ci95": ci95(cluster_values(rows, "source_gain"), 101),
        "erased_shift_mean": mean(cluster_values(rows, "erased_shift")),
        "semantic_half": {
            str(parity): {
                "n_items": len({item_key(row) for row in rows if half(row) == parity}),
                "error_mean": mean([float(row["reversion"]) for row in rows if half(row) == parity]),
                "source_gain_mean": mean([float(row["source_gain"]) for row in rows if half(row) == parity]),
            }
            for parity in (0, 1)
        },
    }


def patch_summary(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    rows = payload.get("rows", [])
    source_effects = [float(row["source_effect"]) for row in rows]
    matrix = [[float(value) for value in row["recovery_by_layer"]] for row in rows]
    n_layers = len(matrix[0]) if matrix else 0
    layer_means = [mean([row[layer] for row in matrix]) for layer in range(n_layers)]
    max_layer = max(range(n_layers), key=lambda layer: layer_means[layer]) if n_layers else None
    max_values = [max(row) for row in matrix]
    return {
        "file": str(path),
        "repo": payload.get("repo"),
        "axis": payload.get("axis"),
        "revision": payload.get("revision"),
        "training_axis": payload.get("training_axis", payload.get("training_tokens_b")),
        "axis_unit": payload.get("axis_unit", "training_tokens_b"),
        "template": payload.get("template"),
        "item_mode": payload.get("item_mode"),
        "donor_mode": payload.get("donor_mode", rows[0].get("donor_mode", "source") if rows else "source"),
        "n": len(rows),
        "source_effect_mean": mean(source_effects),
        "source_effect_ci95": ci95(source_effects, 200),
        "recovery_by_layer_mean": layer_means,
        "recovery_max_layer": max_layer,
        "recovery_max_mean": mean(max_values),
        "recovery_max_ci95": ci95(max_values, 201),
        "recovery_final_mean": layer_means[-1] if layer_means else float("nan"),
        "recovery_max_abs": max((abs(value) for row in matrix for value in row), default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", default="SEMANTIC-AXIS-SUMMARY")
    args = parser.parse_args()

    behavior = []
    for path in sorted(args.behavior_dir.glob("*.json")):
        if path.name.endswith("summary.json"):
            continue
        payload = json.loads(path.read_text())
        rows = payload.get("rows", [])
        if not rows or "erased_margin" not in rows[0]:
            continue
        behavior.append(behavior_summary(payload, path))
    patches = []
    for path in sorted(args.patch_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload.get("rows", [])
        if not rows or "recovery_by_layer" not in rows[0]:
            continue
        patches.append(patch_summary(payload, path))
    result = {
        "status": args.status,
        "behavior_files": len(behavior),
        "patch_files": len(patches),
        "behavior": behavior,
        "patch": patches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
