#!/usr/bin/env python3
"""Validate and summarize the frozen semantic falsification controls.

The checker is intentionally descriptive about effect size.  It enforces the
complete, predeclared matrix and its metadata, then compares each control with
the already-produced source-donor patch at the same axis/checkpoint/template.
It never selects a checkpoint or declares a control successful because its
effect is large or small.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TEMPLATES = ("A", "B", "C")
MODES = ("conflict", "neutral")
DONOR_MODES = ("wrong_position", "random_residual")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("axis"),
        payload.get("revision"),
        payload.get("training_axis", payload.get("training_tokens_b")),
        payload.get("template"),
        payload.get("item_mode"),
    )


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("item_key"),
        row.get("cat"),
        row.get("subj"),
        row.get("x"),
        row.get("y"),
    )


def item_multiset_hash(rows: list[dict[str, Any]]) -> str:
    """Hash the sampled item multiset; repeated frozen draws are intentional."""
    counts: dict[str, int] = {}
    for row in rows:
        encoded = json.dumps(row_key(row), ensure_ascii=True, separators=(",", ":"))
        counts[encoded] = counts.get(encoded, 0) + 1
    canonical = json.dumps(sorted(counts.items()), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def summarize(path: Path, expected_mode: str | None = None) -> dict[str, Any]:
    payload = load(path)
    base: dict[str, Any] = {"file": str(path), "valid": False}
    if payload is None:
        base["reason"] = "invalid-json"
        return base

    rows = payload.get("rows")
    mode = payload.get("donor_mode")
    if mode is None and rows and isinstance(rows[0], dict):
        mode = rows[0].get("donor_mode")
    base.update(
        {
            "axis": payload.get("axis"),
            "revision": payload.get("revision"),
            "training_axis": payload.get("training_axis", payload.get("training_tokens_b")),
            "template": payload.get("template"),
            "item_mode": payload.get("item_mode"),
            "donor_mode": mode,
            "n": len(rows) if isinstance(rows, list) else None,
        }
    )
    if expected_mode is not None and mode != expected_mode:
        base["reason"] = "donor-mode-mismatch"
        return base
    if not isinstance(rows, list) or len(rows) != 60 or not rows:
        base["reason"] = "row-count"
        return base
    if any(not isinstance(row, dict) for row in rows):
        base["reason"] = "row-schema"
        return base
    matrices = [row.get("recovery_by_layer") for row in rows]
    if any(not isinstance(matrix, list) or not matrix for matrix in matrices):
        base["reason"] = "missing-layer-matrix"
        return base
    n_layers = len(matrices[0])
    if n_layers == 0 or any(len(matrix) != n_layers for matrix in matrices):
        base["reason"] = "layer-shape"
        return base
    required = {"item_key", "cat", "subj", "x", "y", "source_position", "source_effect"}
    if any(not required <= set(row) for row in rows):
        base["reason"] = "row-fields"
        return base
    if any(not all(finite(value) for value in matrix) for matrix in matrices):
        base["reason"] = "nonfinite-recovery"
        return base
    if any(not finite(row["source_effect"]) for row in rows):
        base["reason"] = "nonfinite-source-effect"
        return base
    layer_values = [
        [float(row["recovery_by_layer"][layer]) for row in rows]
        for layer in range(n_layers)
    ]
    layer_means = [mean(values) for values in layer_values]
    max_values = [max(float(value) for value in matrix) for matrix in matrices]
    base.update(
        {
            "valid": True,
            "item_multiset_hash": item_multiset_hash(rows),
            "unique_items": len({row_key(row) for row in rows}),
            "n_layers": n_layers,
            "source_effect_mean": mean([float(row["source_effect"]) for row in rows]),
            "recovery_by_layer_mean": layer_means,
            "recovery_max_mean": mean(max_values),
            "recovery_max_abs": max(
                (abs(float(value)) for matrix in matrices for value in matrix),
                default=0.0,
            ),
            "source_positions": {
                "min": min(int(row["source_position"]) for row in rows),
                "max": max(int(row["source_position"]) for row in rows),
            },
        }
    )
    if mode == "wrong_position":
        base["wrong_position_distinct_fraction"] = mean(
            [
                float(row.get("donor_position") is not None and row.get("donor_position") != row["source_position"])
                for row in rows
            ]
        )
    elif mode == "random_residual":
        base["random_seed_present_fraction"] = mean(
            [float(row.get("donor_seed") is not None) for row in rows]
        )
        base["random_position_none_fraction"] = mean(
            [float(row.get("donor_position") is None) for row in rows]
        )
    return base


def compare(source: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    source_layers = source.get("recovery_by_layer_mean", [])
    control_layers = control.get("recovery_by_layer_mean", [])
    if len(source_layers) != len(control_layers):
        return {"compatible": False, "reason": "layer-count-mismatch"}
    deltas = [float(c) - float(s) for s, c in zip(source_layers, control_layers)]
    source_max = float(source.get("recovery_max_mean", float("nan")))
    control_max = float(control.get("recovery_max_mean", float("nan")))
    ratio = control_max / source_max if finite(source_max) and abs(source_max) > 1e-12 else None
    return {
        "compatible": True,
        "item_multiset_match": source.get("item_multiset_hash")
        == control.get("item_multiset_hash"),
        "max_abs_layer_delta": max((abs(value) for value in deltas), default=0.0),
        "control_minus_source_layer_mean": deltas,
        "source_recovery_max_mean": source_max,
        "control_recovery_max_mean": control_max,
        "control_to_source_max_mean_ratio": ratio,
    }


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--axis-value", type=int, required=True)
    parser.add_argument("--axes", nargs="+", default=["type", "relation", "naturalistic"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    control_summaries: list[dict[str, Any]] = []
    source_summaries: dict[tuple[Any, ...], dict[str, Any]] = {}
    expected = {
        (axis, args.revision, args.axis_value, template, mode)
        for axis in args.axes
        for template in TEMPLATES
        for mode in MODES
    }
    source_paths = []
    control_paths = {donor: [] for donor in DONOR_MODES}
    for axis in args.axes:
        patch_dir = args.semantic_dir / axis / "patches"
        source_paths.extend(sorted(patch_dir.glob("*_source.json")))
        for donor in DONOR_MODES:
            control_paths[donor].extend(sorted(patch_dir.glob(f"*_{donor}.json")))

    for path in source_paths:
        summary = summarize(path, "source")
        if summary.get("valid"):
            source_summaries[key(summary)] = summary
    source_keys = set(source_summaries)
    missing_source = expected - source_keys
    checks.append(
        check(
            "source-reference-coverage",
            not missing_source and len(source_keys) >= len(expected),
            f"valid={len(source_keys)} expected_peak={len(expected)} missing={sorted(missing_source, key=str)[:6]}",
        )
    )

    for donor in DONOR_MODES:
        summaries = [summarize(path, donor) for path in control_paths[donor]]
        control_summaries.extend(summaries)
        valid = [summary for summary in summaries if summary.get("valid")]
        keys = {key(summary) for summary in valid}
        missing = expected - keys
        checks.append(
            check(
                f"{donor}-coverage",
                len(keys) == len(expected) and not missing and len(summaries) == len(expected),
                f"valid={len(keys)} files={len(summaries)} expected={len(expected)} missing={sorted(missing, key=str)[:6]}",
            )
        )
        bad = [summary for summary in summaries if not summary.get("valid")]
        checks.append(
            check(
                f"{donor}-schema",
                not bad,
                f"invalid={[(row.get('file'), row.get('reason')) for row in bad[:6]]}",
            )
        )
        if donor == "wrong_position":
            bad_geometry = [
                summary
                for summary in valid
                if summary.get("wrong_position_distinct_fraction") != 1.0
            ]
            checks.append(
                check(
                    "wrong-position-geometry",
                    not bad_geometry,
                    f"files_with_nonadjacent_or_missing_donor={len(bad_geometry)}",
                )
            )
        else:
            bad_geometry = [
                summary
                for summary in valid
                if summary.get("random_seed_present_fraction") != 1.0
                or summary.get("random_position_none_fraction") != 1.0
            ]
            checks.append(
                check(
                    "random-residual-geometry",
                    not bad_geometry,
                    f"files_with_bad_seed_or_position_metadata={len(bad_geometry)}",
                )
            )

    comparisons: list[dict[str, Any]] = []
    for summary in control_summaries:
        if not summary.get("valid"):
            continue
        reference = source_summaries.get(key(summary))
        if reference is None:
            continue
        comparisons.append(
            {
                "axis": summary.get("axis"),
                "revision": summary.get("revision"),
                "training_axis": summary.get("training_axis"),
                "template": summary.get("template"),
                "item_mode": summary.get("item_mode"),
                "donor_mode": summary.get("donor_mode"),
                "control": summary,
                "source": reference,
                "comparison": compare(reference, summary),
            }
        )
    checks.append(
        check(
            "source-control-pairing",
            len(comparisons) == len(expected) * len(DONOR_MODES),
            f"paired={len(comparisons)} expected={len(expected) * len(DONOR_MODES)}",
        )
    )
    bad_item_multisets = [
        pairing
        for pairing in comparisons
        if not pairing.get("comparison", {}).get("item_multiset_match", False)
    ]
    checks.append(
        check(
            "source-control-item-multiset",
            not bad_item_multisets,
            f"mismatched_pairs={len(bad_item_multisets)}",
        )
    )

    result = {
        "status": "SEMANTIC-FALSIFICATION-GATE",
        "passed": all(item["passed"] for item in checks),
        "family": args.family,
        "semantic_dir": str(args.semantic_dir),
        "revision": args.revision,
        "axis_value": args.axis_value,
        "axes": list(args.axes),
        "checks": checks,
        "controls": control_summaries,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
