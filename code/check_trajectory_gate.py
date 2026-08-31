"""Apply the predeclared independent-family coarse trajectory gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    manifest = json.loads(args.manifest.read_text())
    curve = summary["nonmonotonic_excursion"]
    halves = summary.get("item_half_splits", {})
    half_excursions = {}
    for key, value in halves.items():
        try:
            excursion = float(value["excursion"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(excursion):
            half_excursions[str(key)] = excursion
    try:
        curve_excursion = float(curve["excursion"])
        bootstrap_lower = float(curve["bootstrap_95"][0])
    except (KeyError, TypeError, ValueError):
        curve_excursion = float("nan")
        bootstrap_lower = float("nan")
    has_peak = all(
        curve.get(key) is not None
        for key in ("before_tokens_b", "peak_tokens_b", "after_tokens_b")
    )
    passed = (
        has_peak
        and math.isfinite(curve_excursion)
        and curve_excursion >= 0.10
        and math.isfinite(bootstrap_lower)
        and bootstrap_lower > 0.05
        and len(half_excursions) >= 2
        and all(value >= 0.05 for value in half_excursions.values())
    )
    axis_to_revision = {
        int(point.get("axis_value", point.get("tokens_b"))): point["revision"]
        for point in manifest["checkpoints"]
    }
    selected = {
        "before_axis": int(curve["before_tokens_b"]) if has_peak else None,
        "peak_axis": int(curve["peak_tokens_b"]) if has_peak else None,
        "after_axis": int(curve["after_tokens_b"]) if has_peak else None,
    }
    if has_peak and not all(value in axis_to_revision for value in selected.values()):
        passed = False
    result = {
        "status": "PASS" if passed else "FAIL",
        "gate": {
            "excursion_threshold": 0.10,
            "bootstrap_lower_threshold": 0.05,
            "half_excursion_threshold": 0.05,
        },
        "excursion": curve,
        "half_excursions": half_excursions,
        "selected": selected,
        "selected_revisions": {
            "before": axis_to_revision.get(selected["before_axis"])
            if has_peak else None,
            "peak": axis_to_revision.get(selected["peak_axis"])
            if has_peak else None,
            "after": axis_to_revision.get(selected["after_axis"])
            if has_peak else None,
        },
        "axis_unit": summary.get("axis_unit", manifest.get("checkpoint_unit", "training_tokens_b")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
