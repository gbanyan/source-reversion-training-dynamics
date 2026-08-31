"""Paired A/B/C audit of the 7B 3201B-to-3896B transition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "template_7b_summary.json"


def load(mode: str, token: int, template: str) -> list[dict]:
    return json.loads((
        HERE / f"template_7b_{mode}_{token}_{template}.json"
    ).read_text())["rows"]


def key(row: dict) -> tuple[str, str, str, str, str]:
    return row["cat"], row["subj"], row["x"], row["z"], row["y"]


def paired(high: list[dict], low: list[dict], field: str) -> dict:
    high_by_key = {key(row): row for row in high}
    low_by_key = {key(row): row for row in low}
    keys = sorted(set(high_by_key) & set(low_by_key))
    differences = np.asarray([
        high_by_key[item][field] - low_by_key[item][field] for item in keys
    ], dtype=float)
    rng = np.random.default_rng(20260824)
    draws = np.empty(10_000)
    for draw in range(len(draws)):
        sample = rng.integers(0, len(keys), size=len(keys))
        draws[draw] = differences[sample].mean()
    return {
        "n_unique_items": len(keys),
        "high_minus_low": float(differences.mean()),
        "bootstrap_95": [
            float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
        ],
        "bootstrap_p_le_0": float(np.mean(draws <= 0)),
    }


def main() -> None:
    result = {}
    for template in "ABC":
        conflict_high = load("conflict", 3201, template)
        conflict_low = load("conflict", 3896, template)
        neutral_high = load("neutral", 3201, template)
        neutral_low = load("neutral", 3896, template)
        result[template] = {
            "conflict_error_3201": float(np.mean([row["error"] for row in conflict_high])),
            "conflict_error_3896": float(np.mean([row["error"] for row in conflict_low])),
            "conflict_error_drop": paired(conflict_high, conflict_low, "error"),
            "neutral_error_3201": float(np.mean([row["error"] for row in neutral_high])),
            "neutral_error_3896": float(np.mean([row["error"] for row in neutral_low])),
            "neutral_error_drop": paired(neutral_high, neutral_low, "error"),
            "neutral_source_gain_3201": float(np.mean([
                row["source_gain"] for row in neutral_high
            ])),
            "neutral_source_gain_3896": float(np.mean([
                row["source_gain"] for row in neutral_low
            ])),
            "neutral_source_gain_increase": paired(
                neutral_low, neutral_high, "source_gain"
            ),
        }
    summary = {
        "status": "PARTIAL-TEMPLATE-REPLICATION",
        "repo": "allenai/OLMo-2-1124-7B",
        "transition": "3201B to 3896B",
        "templates": result,
        "gate": {
            "all_templates_conflict_drop_lower_gt_0_05": all(
                values["conflict_error_drop"]["bootstrap_95"][0] > .05
                for values in result.values()
            ),
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
