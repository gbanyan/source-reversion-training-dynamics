#!/usr/bin/env python3
"""Pool the frozen semantic-axis confirmations across model families.

This is a reporting/robustness pass, not a new checkpoint-selection step.  It
uses the already selected peak for each family, gives each family equal weight,
and never treats checkpoints as independent statistical replicates.  The
output is deliberately machine-readable so paper tables and figures can be
regenerated without re-running a model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


AXES = ("type", "relation", "naturalistic")
TEMPLATES = ("A", "B", "C")
MODES = ("conflict", "neutral")
DONOR_MODES = ("wrong_position", "random_residual")


def load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: Iterable[Any]) -> float:
    numbers = [float(value) for value in values if finite(value)]
    return sum(numbers) / len(numbers) if numbers else float("nan")


def lower(interval: Any) -> float:
    if not isinstance(interval, list) or not interval:
        return float("nan")
    return float(interval[0]) if finite(interval[0]) else float("nan")


def same_axis(left: Any, right: Any) -> bool:
    """Compare JSON numeric/string checkpoint labels without coercing output."""
    if str(left) == str(right):
        return True
    try:
        return math.isfinite(float(left)) and math.isfinite(float(right)) and float(left) == float(right)
    except (TypeError, ValueError):
        return False


def json_safe(value: Any) -> Any:
    """Turn diagnostic NaN/Infinity values into JSON null for pending runs."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def rows_at(rows: list[dict[str, Any]], peak: Any, item_mode: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if same_axis(row.get("training_axis"), peak)
        and row.get("item_mode") == item_mode
    ]


def leave_one(values: dict[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dropped in sorted(values):
        kept = [value for key, value in values.items() if key != dropped]
        result[dropped] = {"mean": mean(kept), "templates": len(kept)}
    return result


def summary_path(axis_dir: Path, family: str, axis: str) -> Path | None:
    exact = axis_dir / f"{family}_{axis}_semantic_summary.json"
    if exact.is_file():
        return exact
    candidates = sorted(axis_dir.glob("*_semantic_summary.json"))
    return candidates[0] if len(candidates) == 1 else None


def family_axis(
    runs_root: Path, family: str, axis: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Read one family/axis and return a report plus missing-artifact notes."""
    missing: list[str] = []
    axis_dir = runs_root / f"{family}_semantic" / axis
    summary_file = summary_path(axis_dir, family, axis)
    behavior_gate_file = axis_dir / f"{family}_{axis}_semantic_gate.json"
    falsification_file = runs_root / f"{family}_semantic" / f"{family}_semantic_falsification_gate.json"
    summary = load(summary_file) if summary_file else None
    behavior_gate = load(behavior_gate_file)
    falsification_gate = load(falsification_file)
    for label, path, payload in (
        ("semantic summary", summary_file, summary),
        ("semantic gate", behavior_gate_file, behavior_gate),
        ("falsification gate", falsification_file, falsification_gate),
    ):
        if path is None or payload is None:
            missing.append(f"{family}/{axis}: missing {label}")

    if summary is None or behavior_gate is None or falsification_gate is None:
        return None, missing

    peak = behavior_gate.get("peak_axis")
    if peak is None:
        missing.append(f"{family}/{axis}: semantic gate has no peak_axis")
        return None, missing
    behavior = summary.get("behavior", [])
    patches = summary.get("patch", [])
    if not isinstance(behavior, list) or not isinstance(patches, list):
        missing.append(f"{family}/{axis}: malformed summary lists")
        return None, missing

    conflict = rows_at(behavior, peak, "conflict")
    source_patch = [
        row
        for row in rows_at(patches, peak, "conflict")
        if row.get("donor_mode", "source") == "source"
    ]
    template_behavior = {
        str(row.get("template")): row for row in conflict if row.get("template") in TEMPLATES
    }
    template_patch = {
        str(row.get("template")): row for row in source_patch if row.get("template") in TEMPLATES
    }
    if set(template_behavior) != set(TEMPLATES):
        missing.append(f"{family}/{axis}: incomplete peak behavior templates")
    if set(template_patch) != set(TEMPLATES):
        missing.append(f"{family}/{axis}: incomplete peak source-patch templates")

    behavior_by_template = {
        template: {
            "source_gain_mean": row.get("source_gain_mean"),
            "source_gain_ci95": row.get("source_gain_ci95"),
            "reversion_mean": row.get("error_mean"),
            "semantic_half": row.get("semantic_half", {}),
        }
        for template, row in sorted(template_behavior.items())
    }
    patch_by_template = {
        template: {
            "recovery_max_mean": row.get("recovery_max_mean"),
            "recovery_max_ci95": row.get("recovery_max_ci95"),
            "recovery_max_layer": row.get("recovery_max_layer"),
            "recovery_by_layer_mean": row.get("recovery_by_layer_mean", []),
        }
        for template, row in sorted(template_patch.items())
    }

    control_by_mode: dict[str, dict[str, dict[str, Any]]] = {mode: {} for mode in DONOR_MODES}
    comparisons = falsification_gate.get("comparisons", [])
    if not isinstance(comparisons, list):
        comparisons = []
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        if not same_axis(comparison.get("training_axis"), peak):
            continue
        if comparison.get("item_mode") != "conflict":
            continue
        template = str(comparison.get("template"))
        donor_mode = comparison.get("donor_mode")
        if template in TEMPLATES and donor_mode in DONOR_MODES:
            control_by_mode[donor_mode][template] = comparison.get("control", {})

    controls = {
        donor_mode: {
            template: {
                "recovery_max_mean": payload.get("recovery_max_mean"),
                "recovery_by_layer_mean": payload.get("recovery_by_layer_mean", []),
                "source_effect_mean": payload.get("source_effect_mean"),
                "source_positions": payload.get("source_positions"),
            }
            for template, payload in sorted(template_map.items())
        }
        for donor_mode, template_map in control_by_mode.items()
    }

    gains = {template: value["source_gain_mean"] for template, value in behavior_by_template.items()}
    gain_lowers = {
        template: lower(value["source_gain_ci95"])
        for template, value in behavior_by_template.items()
    }
    patch_max = {
        template: value["recovery_max_mean"] for template, value in patch_by_template.items()
    }
    patch_lowers = {
        template: lower(value["recovery_max_ci95"])
        for template, value in patch_by_template.items()
    }
    half_values: dict[str, float] = {}
    for template, value in behavior_by_template.items():
        halves = value.get("semantic_half", {})
        for half_id in ("0", "1"):
            half_row = halves.get(half_id, {}) if isinstance(halves, dict) else {}
            half_values[f"{template}:half{half_id}"] = half_row.get("source_gain_mean", float("nan"))

    control_means = {
        donor_mode: mean(
            payload.get("recovery_max_mean")
            for payload in template_map.values()
        )
        for donor_mode, template_map in controls.items()
    }
    source_mean = mean(patch_max.values())
    control_ratios = {
        donor_mode: (
            control_means[donor_mode] / source_mean
            if finite(control_means[donor_mode]) and finite(source_mean) and abs(source_mean) > 1e-12
            else None
        )
        for donor_mode in DONOR_MODES
    }

    report = {
        "family": family,
        "axis": axis,
        "repo": behavior_gate.get("repo"),
        "peak_axis": peak,
        "semantic_gate_passed": bool(behavior_gate.get("passed")),
        "falsification_gate_passed": bool(falsification_gate.get("passed")),
        "behavior": {
            "templates": behavior_by_template,
            "source_gain_mean_equal_template_weight": mean(gains.values()),
            "source_gain_ci_lower_min": min(gain_lowers.values(), default=float("nan")),
            "positive_template_ci_count": sum(
                finite(value) and value > 0 for value in gain_lowers.values()
            ),
            "positive_semantic_half_count": sum(
                finite(value) and value > 0 for value in half_values.values()
            ),
            "semantic_half_values": half_values,
            "leave_one_template_out": leave_one(gains),
        },
        "source_patch": {
            "templates": patch_by_template,
            "recovery_max_mean_equal_template_weight": source_mean,
            "recovery_max_ci_lower_min": min(patch_lowers.values(), default=float("nan")),
            "positive_template_ci_count": sum(
                finite(value) and value > 0 for value in patch_lowers.values()
            ),
            "leave_one_template_out": leave_one(patch_max),
        },
        "controls": controls,
        "control_recovery_max_mean": control_means,
        "control_to_source_recovery_ratio": control_ratios,
    }
    return report, missing


def pool_axis(reports: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    selected = [report for report in reports if report["axis"] == axis]
    family_means = {
        report["family"]: report["behavior"]["source_gain_mean_equal_template_weight"]
        for report in selected
    }
    source_patch_means = {
        report["family"]: report["source_patch"]["recovery_max_mean_equal_template_weight"]
        for report in selected
    }
    return {
        "axis": axis,
        "family_count": len(selected),
        "families": [report["family"] for report in selected],
        "all_family_gates_passed": all(
            report["semantic_gate_passed"] and report["falsification_gate_passed"]
            for report in selected
        ),
        "family_behavior_source_gain_means": family_means,
        "equal_family_weight_mean_source_gain": mean(family_means.values()),
        "family_source_patch_recovery_means": source_patch_means,
        "equal_family_weight_mean_source_patch_recovery": mean(source_patch_means.values()),
        "all_families_positive_template_ci": all(
            report["behavior"]["positive_template_ci_count"] >= 2 for report in selected
        ),
        "all_families_positive_source_patch_ci": all(
            report["source_patch"]["positive_template_ci_count"] >= 2 for report in selected
        ),
        "leave_one_family_out": {
            dropped["family"]: {
                "remaining_families": [item["family"] for item in selected if item is not dropped],
                "source_gain_mean": mean(
                    item["behavior"]["source_gain_mean_equal_template_weight"]
                    for item in selected
                    if item is not dropped
                ),
            }
            for dropped in selected
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["amber", "pythia6p9"])
    parser.add_argument("--axes", nargs="+", default=list(AXES))
    args = parser.parse_args()

    reports: list[dict[str, Any]] = []
    missing: list[str] = []
    for family in args.families:
        for axis in args.axes:
            report, notes = family_axis(args.runs_root, family, axis)
            missing.extend(notes)
            if report is not None:
                reports.append(report)

    pooled = [pool_axis(reports, axis) for axis in args.axes]
    all_gates_passed = all(
        report["semantic_gate_passed"] and report["falsification_gate_passed"]
        for report in reports
    )
    result = {
        "status": "CROSS-FAMILY-SEMANTIC-POOL",
        "passed": not missing
        and len(reports) == len(args.families) * len(args.axes)
        and all(item["family_count"] == len(args.families) for item in pooled)
        and all_gates_passed,
        "families": list(args.families),
        "axes": list(args.axes),
        "unit_of_inference": "model family; checkpoints are not independent replicates",
        "weighting": "equal family weight, then equal template weight within family",
        "all_input_gates_passed": all_gates_passed,
        "missing_or_invalid": missing,
        "family_axis": reports,
        "pooled_axis": pooled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_result = json_safe(result)
    args.output.write_text(json.dumps(safe_result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(safe_result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
