"""Audit deterministic template replication and the pre-final routing gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
TOKENS = (1909, 1993, 2496, 2999)
TEMPLATES = ("A", "B", "C")
OUT = HERE / "deterministic_template_summary.json"


def load(mode: str, token: int, template: str) -> list[dict]:
    return json.loads((
        HERE / f"routing_diag_{mode}_{token}_{template}.json"
    ).read_text())["rows"]


def bootstrap_rate_drop(high_rows: list[dict], low_rows: list[dict]) -> dict:
    def key(row: dict) -> tuple[str, str, str, str, str]:
        return row["cat"], row["subj"], row["x"], row["z"], row["y"]

    high = {key(row): row for row in high_rows}
    low = {key(row): row for row in low_rows}
    keys = sorted(set(high) & set(low))
    differences = np.asarray([
        high[item]["reversion_target"] - low[item]["reversion_target"]
        for item in keys
    ], dtype=float)
    rng = np.random.default_rng(20260824)
    draws = np.empty(10_000)
    for draw in range(len(draws)):
        sample = rng.integers(0, len(keys), size=len(keys))
        draws[draw] = differences[sample].mean()
    return {
        "n_unique_items": len(keys),
        "rate_drop": float(differences.mean()),
        "paired_bootstrap_95": [
            float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
        ],
        "bootstrap_p_drop_le_0": float(np.mean(draws <= 0)),
    }


def main() -> None:
    conflict = {
        (token, template): load("conflict", token, template)
        for token in TOKENS for template in TEMPLATES
    }
    neutral = {
        (token, template): load("neutral", token, template)
        for token in TOKENS for template in TEMPLATES
    }

    group_rates = []
    neutral_lens_gain = []
    neutral_attention = []
    neutral_path = []
    group_names = []
    groups = {}
    for token in TOKENS:
        for template in TEMPLATES:
            conflict_rows = conflict[token, template]
            neutral_rows = neutral[token, template]
            rate = float(np.mean([
                row["reversion_target"] for row in conflict_rows
            ]))
            lens = np.asarray([
                np.asarray(row["clean_logit_lens_by_layer"])
                - np.asarray(row["erased_logit_lens_by_layer"])
                for row in neutral_rows
            ]).mean(axis=0)
            # hidden_states[-1] is already final-normalized in Transformers;
            # replace the double-normalized diagnostic value with exact output gain.
            lens[-1] = np.mean([
                row["clean_margin_target_only"] - row["erased_margin"]
                for row in neutral_rows
            ])
            attention = np.asarray([
                row["attention_to_source"] for row in neutral_rows
            ]).mean(axis=(0, 2))
            path = np.asarray([
                row["source_path_readout"] for row in neutral_rows
            ]).sum(axis=2).mean(axis=0)
            name = f"{token}_{template}"
            group_names.append(name)
            group_rates.append(rate)
            neutral_lens_gain.append(lens)
            neutral_attention.append(attention)
            neutral_path.append(path)
            groups[name] = {
                "conflict_reversion": rate,
                "neutral_final_source_gain": float(lens[-1]),
                "neutral_error_rate": float(np.mean([
                    row["reversion_target"] for row in neutral_rows
                ])),
            }

    group_rates = np.asarray(group_rates)
    neutral_lens_gain = np.asarray(neutral_lens_gain)
    neutral_attention = np.asarray(neutral_attention)
    neutral_path = np.asarray(neutral_path)

    layerwise = {}
    for metric_name, matrix in (
        ("neutral_logit_lens_source_gain", neutral_lens_gain),
        ("neutral_attention_to_source", neutral_attention),
        ("neutral_source_path_readout", neutral_path),
    ):
        tests = []
        for layer in range(matrix.shape[1]):
            result = spearmanr(group_rates, matrix[:, layer])
            tests.append({
                "layer": layer,
                "rho": float(result.statistic),
                "p_uncorrected": float(result.pvalue),
                "p_bonferroni_48": float(min(1, result.pvalue * 48)),
            })
        layerwise[metric_name] = tests

    transitions = {
        "early_1909_to_1993": {
            template: bootstrap_rate_drop(
                conflict[1909, template], conflict[1993, template]
            )
            for template in TEMPLATES
        },
        "late_2496_to_2999": {
            template: bootstrap_rate_drop(
                conflict[2496, template], conflict[2999, template]
            )
            for template in TEMPLATES
        },
    }
    final_gain = spearmanr(group_rates, neutral_lens_gain[:, -1])
    prefinal_tests = [
        test
        for metric in layerwise.values()
        for test in metric[:12]
    ]
    summary = {
        "status": "DETERMINISTIC-TEMPLATE-AUDIT",
        "groups": groups,
        "transition_replication": transitions,
        "final_neutral_gain_correlation": {
            "rho": float(final_gain.statistic),
            "p": float(final_gain.pvalue),
        },
        "prefinal_gate": {
            "layers_allowed": [0, 11],
            "n_tests": len(prefinal_tests),
            "any_bonferroni_significant": any(
                test["p_bonferroni_48"] < .05 for test in prefinal_tests
            ),
            "verdict": "FAIL",
            "reason": "No L0-L11 neutral routing measure predicts conflict rate across checkpoint-template groups.",
        },
        "layerwise_spearman": layerwise,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
