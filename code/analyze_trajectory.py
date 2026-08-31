"""Architecture-independent summary for a frozen checkpoint trajectory.

The script deliberately reports the same descriptive excursion statistic used
for the OLMo-2 audit.  A family-screen result is not a confirmatory p-value:
the family and its trajectory are selected before any causal follow-up.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def peak_excursion(curve: np.ndarray) -> dict:
    best = {"excursion": float("-inf"), "status": "INSUFFICIENT_TRAJECTORY"}
    finite = np.isfinite(curve)
    for peak in range(1, len(curve) - 1):
        if not finite[peak]:
            continue
        before_candidates = np.flatnonzero(finite[:peak])
        after_candidates = np.flatnonzero(finite[peak + 1 :]) + peak + 1
        if not len(before_candidates) or not len(after_candidates):
            continue
        before = int(before_candidates[np.argmin(curve[before_candidates])])
        after = int(after_candidates[np.argmin(curve[after_candidates])])
        excursion = min(curve[peak] - curve[before], curve[peak] - curve[after])
        if excursion > best["excursion"]:
            best = {
                "excursion": float(excursion),
                "before_index": before,
                "peak_index": peak,
                "after_index": after,
                "status": "OK",
            }
    return best


def item_key(row: dict) -> tuple[str, str, str, str]:
    return row["cat"], row["subj"], row["x"], row["y"]


def axis_value(row: dict) -> int:
    """Read the numeric checkpoint axis, retaining legacy compatibility."""
    return int(row.get("training_axis", row["training_tokens_b"]))


def aggregate(rows: list[dict]) -> tuple[list[int], list[tuple], dict[str, np.ndarray]]:
    tokens = sorted({axis_value(row) for row in rows})
    keys = sorted({item_key(row) for row in rows})
    grouped: dict[tuple[int, tuple], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[axis_value(row), item_key(row)].append(row)
    metrics = {}
    for name in ("reversion", "prior_strength", "dm"):
        matrix = np.full((len(keys), len(tokens)), np.nan)
        for i, key in enumerate(keys):
            for j, token in enumerate(tokens):
                values = [float(row[name]) for row in grouped[token, key]]
                if values:
                    matrix[i, j] = np.mean(values)
        metrics[name] = matrix
    return tokens, keys, metrics


def bootstrap_excursion(matrix: np.ndarray, n_boot: int = 10_000) -> np.ndarray:
    rng = np.random.default_rng(20260829)
    values = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        sample = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        values[draw] = peak_excursion(np.nanmean(matrix[sample], axis=0))["excursion"]
    return values


def summarize(matrix: np.ndarray, tokens: list[int]) -> dict:
    curve = np.nanmean(matrix, axis=0)
    peak = peak_excursion(curve)
    if "before_index" not in peak:
        return {
            "status": peak["status"],
            "curve": [float(value) for value in curve],
            "excursion": None,
            "before_tokens_b": None,
            "peak_tokens_b": None,
            "after_tokens_b": None,
            "bootstrap_95": [None, None],
            "bootstrap_p_excursion_le_0_05": None,
        }
    boot = bootstrap_excursion(matrix)
    return {
        "status": peak["status"],
        "curve": [float(value) for value in curve],
        "excursion": float(peak["excursion"]),
        "before_tokens_b": tokens[peak["before_index"]],
        "peak_tokens_b": tokens[peak["peak_index"]],
        "after_tokens_b": tokens[peak["after_index"]],
        "bootstrap_95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "bootstrap_p_excursion_le_0_05": float(np.mean(boot <= .05)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())
    tokens, keys, metrics = aggregate(rows)
    reversion = metrics["reversion"]
    split = {}
    for parity in (0, 1):
        mask = np.asarray([
            int(hashlib.sha256("\x1f".join(key).encode()).hexdigest(), 16) % 2 == parity
            for key in keys
        ])
        split[str(parity)] = {
            "n_items": int(mask.sum()),
            **summarize(reversion[mask], tokens),
        }
    summary = {
        "status": "INDEPENDENT-FAMILY-SCREEN",
        "rows": len(rows),
        "unique_items": len(keys),
        "axis_unit": rows[0].get("axis_unit", "training_tokens_b"),
        "axis_values": tokens,
        "tokens_b": tokens,
        "curve": [float(value) for value in np.nanmean(reversion, axis=0)],
        "nonmonotonic_excursion": summarize(reversion, tokens),
        "item_half_splits": split,
        "mean_prior_strength": [float(value) for value in np.nanmean(metrics["prior_strength"], axis=0)],
        "mean_source_contribution": [float(value) for value in np.nanmean(metrics["dm"], axis=0)],
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
