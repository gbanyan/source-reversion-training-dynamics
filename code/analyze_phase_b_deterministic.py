"""Analyze deterministic source-token patching at the 2496B->2999B transition."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
TOKENS = (2496, 2999)
TEMPLATES = ("A", "B", "C")
MODES = ("conflict", "neutral")
OUT = HERE / "phase_b_deterministic_summary.json"
PLOT = HERE / "phase_b_deterministic_curve.png"


def key(row: dict) -> tuple[str, str, str, str, str]:
    return row["cat"], row["subj"], row["x"], row["z"], row["y"]


def ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000) -> np.ndarray:
    return np.asarray([
        values[rng.integers(0, len(values), len(values))].mean()
        for _ in range(n_boot)
    ])


def load(mode: str, token: int, template: str) -> dict:
    path = HERE / f"phase_b_det_{mode}_{token}_{template}.json"
    return json.loads(path.read_text())


def main() -> None:
    payloads = {
        (mode, token, template): load(mode, token, template)
        for mode in MODES for token in TOKENS for template in TEMPLATES
    }
    n_layers = payloads[("conflict", TOKENS[0], "A")]["n_layers"]
    early_last = int(np.floor(.75 * (n_layers - 1)))
    rng = np.random.default_rng(20260824)
    absolute_rng = np.random.default_rng(20260825)
    groups: dict[str, dict] = {}
    item_deltas: dict[tuple[str, str], np.ndarray] = {}

    for mode in MODES:
        groups[mode] = {}
        for template in TEMPLATES:
            indexed = {
                token: {key(row): row for row in payloads[(mode, token, template)]["rows"]}
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
            curves = {token: matrices[token].mean(axis=0) for token in TOKENS}
            early = {
                token: matrices[token][:, :early_last + 1].mean(axis=1)
                for token in TOKENS
            }
            delta = early[TOKENS[1]] - early[TOKENS[0]]
            item_deltas[(mode, template)] = delta
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
                    "recovery_curve": [float(x) for x in curves[token]],
                }
            groups[mode][template] = {
                "n_items": len(keys),
                "checkpoints": checkpoints,
                "low_2999_minus_high_2496_early_recovery": float(delta.mean()),
                "paired_bootstrap_95": ci(draws),
                "bootstrap_p_delta_le_0": float(np.mean(draws <= 0)),
                "curve_delta": [float(x) for x in curves[TOKENS[1]] - curves[TOKENS[0]]],
            }

    did = {}
    for template in TEMPLATES:
        conflict = item_deltas[("conflict", template)]
        neutral = item_deltas[("neutral", template)]
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
        "status": "DETERMINISTIC-CAUSAL-AUDIT",
        "transition": "2496B high-reversion to 2999B low-reversion",
        "scope": "single source-answer-token residual patch; all matched items",
        "n_layers": n_layers,
        "pre_final_feature_layers": [0, early_last],
        "bootstrap_draws": 10_000,
        "absolute_bootstrap_seed": 20260825,
        "absolute_bootstrap_unit": "matched item",
        "groups": groups,
        "conflict_specific_difference_in_differences": did,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for row_index, mode in enumerate(MODES):
        for column, template in enumerate(TEMPLATES):
            ax = axes[row_index, column]
            for token in TOKENS:
                curve = groups[mode][template]["checkpoints"][str(token)]["recovery_curve"]
                ax.plot(range(n_layers), curve, marker="o", ms=3, label=f"{token}B")
            ax.axvline(early_last, color="0.5", ls="--", lw=1)
            ax.set_title(f"{mode} / template {template}")
            ax.grid(alpha=.2)
            if column == 0:
                ax.set_ylabel("source-donor patch effect (X-Y logit diff.)")
            if row_index == 1:
                ax.set_xlabel("layer")
            if row_index == 0 and column == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT, dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
