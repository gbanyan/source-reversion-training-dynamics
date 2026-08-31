#!/usr/bin/env python3
"""Check a frozen semantic-axis confirmation summary.

The checker is deliberately separate from ``analyze_semantic.py``: it only
reads the already-produced JSON and makes the pre-declared pass/fail decision.
It does not select checkpoints, items, templates, or layers after seeing the
result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _axis_value(row: dict[str, Any]) -> float:
    value = row.get("training_axis", row.get("training_tokens_b"))
    return float(value)


def _lower(row: dict[str, Any], field: str) -> float:
    interval = row.get(field)
    if not isinstance(interval, list) or not interval:
        return float("nan")
    return float(interval[0])


def _check(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: str
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _format_keys(keys: list[tuple[Any, ...]], limit: int = 6) -> str:
    if not keys:
        return "none"
    shown = ", ".join("/".join(map(str, key)) for key in keys[:limit])
    suffix = " ..." if len(keys) > limit else ""
    return shown + suffix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoints", type=int, default=3)
    parser.add_argument("--expected-templates", type=int, default=3)
    parser.add_argument("--behavior-n", type=int, default=120)
    parser.add_argument("--patch-n", type=int, default=60)
    parser.add_argument(
        "--peak-axis",
        type=int,
        default=None,
        help="explicit frozen peak value; defaults to max(training_axis) for legacy summaries",
    )
    parser.add_argument(
        "--min-signal-templates",
        type=int,
        default=2,
        help="minimum templates with a positive peak conflict CI",
    )
    parser.add_argument(
        "--corrupt-tolerance",
        type=float,
        default=1e-9,
        help="absolute tolerance for the exact corrupt-donor no-op check",
    )
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text())
    behavior = payload.get("behavior", [])
    patches = payload.get("patch", [])
    checks: list[dict[str, Any]] = []

    expected_modes = {"conflict", "neutral"}
    expected_templates = {chr(ord("A") + i) for i in range(args.expected_templates)}
    behavior_keys = {
        (row.get("training_axis"), row.get("template"), row.get("item_mode"))
        for row in behavior
    }
    axes = sorted({row.get("training_axis") for row in behavior}, key=str)
    expected_behavior_keys = {
        (axis, template, mode)
        for axis in axes
        for template in expected_templates
        for mode in expected_modes
    }
    missing_behavior = sorted(expected_behavior_keys - behavior_keys, key=str)
    duplicate_behavior = len(behavior_keys) != len(behavior)
    _check(
        checks,
        "behavior-coverage",
        len(axes) == args.expected_checkpoints
        and not missing_behavior
        and not duplicate_behavior,
        f"checkpoints={len(axes)} expected={args.expected_checkpoints}; "
        f"rows={len(behavior)}; missing={_format_keys(missing_behavior)}; "
        f"duplicate_keys={duplicate_behavior}",
    )

    bad_behavior_rows = [
        (row.get("training_axis"), row.get("template"), row.get("item_mode"))
        for row in behavior
        if row.get("n") != args.behavior_n
        or not _finite(row.get("error_mean"))
        or not _finite(row.get("source_gain_mean"))
        or not _finite(_lower(row, "source_gain_ci95"))
    ]
    _check(
        checks,
        "behavior-schema",
        not bad_behavior_rows,
        f"expected_n={args.behavior_n}; invalid={_format_keys(bad_behavior_rows)}",
    )

    source_patches = [row for row in patches if row.get("donor_mode") == "source"]
    corrupt_patches = [row for row in patches if row.get("donor_mode") == "corrupt"]
    source_keys = {
        (row.get("training_axis"), row.get("template"), row.get("item_mode"))
        for row in source_patches
    }
    expected_source_keys = {
        (axis, template, mode)
        for axis in axes
        for template in expected_templates
        for mode in expected_modes
    }
    missing_source = sorted(expected_source_keys - source_keys, key=str)
    _check(
        checks,
        "source-patch-coverage",
        len(source_patches) == args.expected_checkpoints
        * args.expected_templates
        * len(expected_modes)
        and not missing_source,
        f"rows={len(source_patches)} expected="
        f"{args.expected_checkpoints * args.expected_templates * len(expected_modes)}; "
        f"missing={_format_keys(missing_source)}",
    )

    bad_source_rows = [
        (row.get("training_axis"), row.get("template"), row.get("item_mode"))
        for row in source_patches
        if row.get("n") != args.patch_n
        or not _finite(row.get("source_effect_mean"))
        or not _finite(row.get("recovery_max_mean"))
        or not _finite(_lower(row, "recovery_max_ci95"))
    ]
    _check(
        checks,
        "source-patch-schema",
        not bad_source_rows,
        f"expected_n={args.patch_n}; invalid={_format_keys(bad_source_rows)}",
    )

    if axes:
        peak = args.peak_axis if args.peak_axis is not None else max(axes)
        if peak not in axes:
            _check(
                checks,
                "peak-axis-present",
                False,
                f"peak_axis={peak} not in training_axes={axes}",
            )
        peak_conflict = [
            row
            for row in behavior
            if row.get("training_axis") == peak and row.get("item_mode") == "conflict"
        ]
        positive_peak = [
            row for row in peak_conflict if _lower(row, "source_gain_ci95") > 0
        ]
        _check(
            checks,
            "peak-behavior-signal",
            len(positive_peak) >= args.min_signal_templates,
            f"peak_axis={peak}; positive_template_CIs={len(positive_peak)} "
            f"required={args.min_signal_templates}",
        )

        deterministic_half_ok = 0
        deterministic_half_details: list[str] = []
        for row in peak_conflict:
            halves = row.get("semantic_half", {})
            values = [halves.get(str(i), {}).get("source_gain_mean") for i in (0, 1)]
            if all(_finite(value) and float(value) > 0 for value in values):
                deterministic_half_ok += 1
            else:
                deterministic_half_details.append(str(row.get("template")))
        _check(
            checks,
            "peak-deterministic-item-key-halves",
            deterministic_half_ok >= args.min_signal_templates,
            f"templates_with_both_positive_halves={deterministic_half_ok} "
            f"required={args.min_signal_templates}; failed={','.join(deterministic_half_details) or 'none'}",
        )

        peak_source = [
            row
            for row in source_patches
            if row.get("training_axis") == peak and row.get("item_mode") == "conflict"
        ]
        positive_patch = [
            row for row in peak_source if _lower(row, "recovery_max_ci95") > 0
        ]
        _check(
            checks,
            "peak-source-recovery",
            len(positive_patch) >= args.min_signal_templates,
            f"peak_axis={peak}; positive_recovery_CIs={len(positive_patch)} "
            f"required={args.min_signal_templates}",
        )
    else:
        _check(checks, "peak-behavior-signal", False, "no behavior checkpoints")
        _check(checks, "peak-deterministic-item-key-halves", False, "no behavior checkpoints")
        _check(checks, "peak-source-recovery", False, "no behavior checkpoints")

    bad_corrupt_rows = [
        (row.get("training_axis"), row.get("template"), row.get("item_mode"))
        for row in corrupt_patches
        if row.get("n") != args.patch_n
        or not _finite(row.get("recovery_max_abs"))
        or float(row.get("recovery_max_abs")) > args.corrupt_tolerance
    ]
    expected_corrupt = args.expected_templates * len(expected_modes)
    corrupt_at_peak = [
        row for row in corrupt_patches if row.get("training_axis") == peak
    ] if axes else []
    _check(
        checks,
        "peak-corrupt-no-op",
        len(corrupt_patches) == expected_corrupt
        and len(corrupt_at_peak) == expected_corrupt
        and not bad_corrupt_rows,
        f"rows={len(corrupt_patches)} expected={expected_corrupt}; "
        f"at_peak={len(corrupt_at_peak)}; "
        f"tolerance={args.corrupt_tolerance:g}; invalid={_format_keys(bad_corrupt_rows)}",
    )

    result = {
        "status": "SEMANTIC-AXIS-GATE",
        "passed": all(check["passed"] for check in checks),
        "summary": str(args.summary),
        "axis": payload.get("behavior", [{}])[0].get("axis") if behavior else None,
        "repo": payload.get("behavior", [{}])[0].get("repo") if behavior else None,
        "training_axes": axes,
        "peak_axis": args.peak_axis if args.peak_axis is not None else (max(axes) if axes else None),
        "checks": checks,
        "pending_controls": [
            "wrong-position patch",
            "random-residual donor",
            "relation-preserving global candidate permutation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
