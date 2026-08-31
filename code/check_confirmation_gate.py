#!/usr/bin/env python3
"""Apply the predeclared behavioral and causal gate to a confirmation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


TEMPLATES = ("A", "B", "C")
MODES = ("conflict", "neutral")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def bootstrap_mean(values: list[float], seed: int, draws: int = 4000) -> list[float]:
    if not values:
        return []
    rng = random.Random(seed)
    return [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]


def ci95(values: list[float], seed: int) -> list[float]:
    boot = bootstrap_mean(values, seed)
    return [quantile(boot, 0.025), quantile(boot, 0.975)]


def item_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("cat", "")),
        str(row.get("subj", "")),
        str(row.get("x", "")),
        str(row.get("y", "")),
    )


def half(key: tuple[str, str, str, str]) -> int:
    encoded = "\x1f".join(key).encode()
    return int(hashlib.sha256(encoded).hexdigest(), 16) % 2


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def behavior_summary(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload is None or not isinstance(payload.get("rows"), list):
        return {"file": str(path), "valid": False, "reason": "invalid-json"}
    rows = payload["rows"]
    required = {"source_gain", "cat", "subj", "x", "y", "item_index"}
    valid = len(rows) == 120 and all(required <= set(row) for row in rows)
    gains = [float(row["source_gain"]) for row in rows] if valid else []
    by_half = {
        str(part): [
            float(row["source_gain"])
            for row in rows
            if half(item_key(row)) == part
        ]
        for part in (0, 1)
    }
    return {
        "file": str(path),
        "valid": valid,
        "revision": payload.get("revision"),
        "template": payload.get("template"),
        "item_mode": payload.get("item_mode"),
        "axis_unit": payload.get("axis_unit"),
        "n": len(rows),
        "source_gain_mean": sum(gains) / len(gains) if gains else None,
        "source_gain_ci95": ci95(gains, 1200) if gains else [None, None],
        "source_gain_half_mean": {
            key: (sum(values) / len(values) if values else None)
            for key, values in by_half.items()
        },
    }


def patch_summary(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload is None or not isinstance(payload.get("rows"), list):
        return {"file": str(path), "valid": False, "reason": "invalid-json"}
    rows = payload["rows"]
    valid = len(rows) == 60 and all(
        isinstance(row.get("recovery_by_layer"), list)
        and row.get("recovery_by_layer")
        for row in rows
    )
    if not valid:
        return {
            "file": str(path),
            "valid": False,
            "revision": payload.get("revision"),
            "template": payload.get("template"),
            "item_mode": payload.get("item_mode"),
            "donor_mode": payload.get("donor_mode"),
            "n": len(rows),
        }
    n_layers = len(rows[0]["recovery_by_layer"])
    valid = all(len(row["recovery_by_layer"]) == n_layers for row in rows)
    if not valid:
        return {"file": str(path), "valid": False, "reason": "layer-shape"}
    layer_values = [
        [float(row["recovery_by_layer"][layer]) for row in rows]
        for layer in range(n_layers)
    ]
    layer_means = [sum(values) / len(values) for values in layer_values]
    max_layer = max(range(n_layers), key=lambda layer: layer_means[layer])
    return {
        "file": str(path),
        "valid": True,
        "revision": payload.get("revision"),
        "template": payload.get("template"),
        "item_mode": payload.get("item_mode"),
        "donor_mode": payload.get("donor_mode"),
        "axis_unit": payload.get("axis_unit"),
        "n": len(rows),
        "n_layers": n_layers,
        "recovery_by_layer_mean": layer_means,
        "recovery_max_layer": max_layer,
        "recovery_max_mean": layer_means[max_layer],
        "recovery_max_ci95": ci95(layer_values[max_layer], 2200 + max_layer),
        "recovery_max_abs": max(
            (abs(value) for values in layer_values for value in values),
            default=0.0,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--peak-revision", required=True)
    parser.add_argument("--revisions", nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    behavior_files = sorted(args.behavior_dir.glob("*.json"))
    source_files = sorted(args.patch_dir.glob("*_source.json"))
    corrupt_files = sorted(args.patch_dir.glob("*_corrupt.json"))
    behaviors = [behavior_summary(path) for path in behavior_files]
    sources = [patch_summary(path) for path in source_files]
    corrupt = [patch_summary(path) for path in corrupt_files]

    expected_behavior = {
        (revision, template, mode)
        for revision in args.revisions
        for template in TEMPLATES
        for mode in MODES
    }
    behavior_keys = {
        (row.get("revision"), row.get("template"), row.get("item_mode"))
        for row in behaviors
        if row.get("valid")
    }
    expected_source = {
        (revision, template, mode)
        for revision in args.revisions
        for template in TEMPLATES
        for mode in MODES
    }
    source_keys = {
        (row.get("revision"), row.get("template"), row.get("item_mode"))
        for row in sources
        if row.get("valid")
    }
    expected_corrupt = {
        (args.peak_revision, template, mode)
        for template in TEMPLATES
        for mode in MODES
    }
    corrupt_keys = {
        (row.get("revision"), row.get("template"), row.get("item_mode"))
        for row in corrupt
        if row.get("valid")
    }

    peak_conflict = [
        row for row in behaviors
        if row.get("valid")
        and row.get("revision") == args.peak_revision
        and row.get("item_mode") == "conflict"
    ]
    positive_behavior = [
        row for row in peak_conflict
        if row.get("source_gain_ci95", [None])[0] is not None
        and row["source_gain_ci95"][0] > 0
    ]
    positive_behavior_deterministic_halves = [
        row for row in peak_conflict
        if all(
            row["source_gain_half_mean"].get(str(part)) is not None
            and row["source_gain_half_mean"][str(part)] > 0
            for part in (0, 1)
        )
    ]
    peak_source = [
        row for row in sources
        if row.get("valid")
        and row.get("revision") == args.peak_revision
        and row.get("item_mode") == "conflict"
    ]
    positive_recovery = [
        row for row in peak_source
        if row.get("recovery_max_ci95", [None])[0] is not None
        and row["recovery_max_ci95"][0] > 0
    ]
    corrupt_noop = all(
        row.get("valid") and row.get("recovery_max_abs", math.inf) <= 1e-9
        for row in corrupt
    )

    checks = [
        {
            "name": "behavior-coverage",
            "passed": len(behavior_keys) == len(expected_behavior)
            and behavior_keys == expected_behavior,
            "detail": f"valid={len(behavior_keys)} expected={len(expected_behavior)}",
        },
        {
            "name": "behavior-schema",
            "passed": len(behaviors) == len(expected_behavior)
            and all(row.get("valid") for row in behaviors),
            "detail": f"files={len(behaviors)} expected={len(expected_behavior)}",
        },
        {
            "name": "source-patch-coverage",
            "passed": len(source_keys) == len(expected_source)
            and source_keys == expected_source,
            "detail": f"valid={len(source_keys)} expected={len(expected_source)}",
        },
        {
            "name": "source-patch-schema",
            "passed": len(sources) == len(expected_source)
            and all(row.get("valid") for row in sources),
            "detail": f"files={len(sources)} expected={len(expected_source)}",
        },
        {
            "name": "peak-behavior-signal",
            "passed": len(positive_behavior) >= 2,
            "detail": f"positive_template_CIs={len(positive_behavior)} required=2",
        },
        {
            "name": "peak-deterministic-item-key-halves",
            "passed": len(positive_behavior_deterministic_halves) >= 2,
            "detail": f"templates_with_both_positive_halves={len(positive_behavior_deterministic_halves)} required=2",
        },
        {
            "name": "peak-source-recovery",
            "passed": len(positive_recovery) >= 2,
            "detail": f"positive_recovery_CIs={len(positive_recovery)} required=2",
        },
        {
            "name": "peak-corrupt-no-op",
            "passed": len(corrupt_keys) == len(expected_corrupt)
            and corrupt_keys == expected_corrupt
            and len(corrupt) == len(expected_corrupt)
            and corrupt_noop,
            "detail": f"valid={len(corrupt_keys)} expected={len(expected_corrupt)} tolerance=1e-09",
        },
    ]
    result = {
        "status": "CONFIRMATION-GATE",
        "passed": all(check["passed"] for check in checks),
        "peak_revision": args.peak_revision,
        "expected_revisions": list(args.revisions),
        "checks": checks,
        "behavior": behaviors,
        "source": sources,
        "corrupt": corrupt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
