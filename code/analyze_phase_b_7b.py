"""Analyze the preregistered 7B source-token causal replication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TOKENS = (3201, 3896)
TEMPLATES = ("A", "B")
MODES = ("conflict", "neutral")
OUT = HERE / "phase_b_7b_summary.json"


def key(row: dict) -> tuple[str, str, str, str, str]:
    return row["cat"], row["subj"], row["x"], row["z"], row["y"]


def ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000) -> np.ndarray:
    return np.asarray([
        values[rng.integers(0, len(values), len(values))].mean()
        for _ in range(n_boot)
    ])


def main() -> None:
    rng = np.random.default_rng(20260824)
    absolute_rng = np.random.default_rng(20260825)
    groups: dict[str, dict] = {mode: {} for mode in MODES}
    deltas: dict[tuple[str, str], np.ndarray] = {}
    n_layers = None
    early_last = None

    for mode in MODES:
        for template in TEMPLATES:
            payloads = {
                token: json.loads(
                    (HERE / f"phase_b_7b_{mode}_{token}_{template}.json").read_text()
                )
                for token in TOKENS
            }
            n_layers = payloads[TOKENS[0]]["n_layers"]
            early_last = int(np.floor(.75 * (n_layers - 1)))
            indexed = {
                token: {key(row): row for row in payloads[token]["rows"]}
                for token in TOKENS
            }
            keys = sorted(indexed[TOKENS[0]])
            if any(sorted(indexed[token]) != keys for token in TOKENS):
                raise ValueError(f"item mismatch: {mode}/{template}")
            matrices = {
                token: np.asarray([
                    indexed[token][item]["recovery_by_layer"] for item in keys
                ], dtype=float)
                for token in TOKENS
            }
            early = {
                token: matrices[token][:, :early_last + 1].mean(axis=1)
                for token in TOKENS
            }
            delta = early[TOKENS[1]] - early[TOKENS[0]]
            deltas[(mode, template)] = delta
            draws = np.asarray([
                delta[rng.integers(0, len(delta), len(delta))].mean()
                for _ in range(10_000)
            ])
            checkpoints = {}
            for token in TOKENS:
                rows = [indexed[token][item] for item in keys]
                absolute_bootstrap = bootstrap_mean(early[token], absolute_rng)
                checkpoints[str(token)] = {
                    "reversion_rate": float(np.mean([row["reversion"] for row in rows])),
                    "mean_source_effect": float(np.mean([row["source_effect"] for row in rows])),
                    "mean_early_depth_recovery": float(early[token].mean()),
                    "mean_early_depth_recovery_bootstrap_95": ci(absolute_bootstrap),
                    "recovery_curve": [float(x) for x in matrices[token].mean(axis=0)],
                }
            groups[mode][template] = {
                "n_items": len(keys),
                "checkpoints": checkpoints,
                "low_3896_minus_high_3201_early_recovery": float(delta.mean()),
                "paired_bootstrap_95": ci(draws),
                "bootstrap_p_delta_le_0": float(np.mean(draws <= 0)),
            }

    did = {}
    for template in TEMPLATES:
        conflict = deltas[("conflict", template)]
        neutral = deltas[("neutral", template)]
        draws = np.empty(10_000)
        for i in range(len(draws)):
            c = conflict[rng.integers(0, len(conflict), len(conflict))].mean()
            n = neutral[rng.integers(0, len(neutral), len(neutral))].mean()
            draws[i] = c - n
        did[template] = {
            "conflict_delta_minus_neutral_delta": float(conflict.mean() - neutral.mean()),
            "independent_bootstrap_95": ci(draws),
            "bootstrap_p_le_0": float(np.mean(draws <= 0)),
        }

    summary = {
        "status": "INDEPENDENT-SIZE-CAUSAL-REPLICATION",
        "transition": "7B 3201B high-reversion to 3896B low-reversion",
        "scope": "single source-answer-token residual patch; behavior-replicating templates A/B",
        "n_layers": n_layers,
        "pre_final_feature_layers": [0, early_last],
        "bootstrap_draws": 10_000,
        "absolute_bootstrap_seed": 20260825,
        "absolute_bootstrap_unit": "matched item",
        "groups": groups,
        "conflict_specific_difference_in_differences": did,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
