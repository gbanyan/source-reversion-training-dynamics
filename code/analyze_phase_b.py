"""Analyze matched source-position patching at two training transitions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
TOKENS = (1909, 1993, 2496, 2999)
OUT = HERE / "phase_b_srcpatch_summary.json"
PLOT = HERE / "phase_b_srcpatch_curve.png"


def item_key(row: dict) -> tuple[str, str, str, str, str]:
    return row["cat"], row["subj"], row["x"], row["z"], row["y"]


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000) -> np.ndarray:
    return np.asarray([
        values[rng.integers(0, len(values), len(values))].mean()
        for _ in range(n_boot)
    ])


def main() -> None:
    payloads = {
        token: json.loads((HERE / f"phase_b_srcpatch_{token}.json").read_text())
        for token in TOKENS
    }
    indexed = {
        token: {item_key(row): row for row in payload["rows"]}
        for token, payload in payloads.items()
    }
    neutral_payloads = {
        token: json.loads((HERE / f"phase_b_neutral_{token}.json").read_text())
        for token in TOKENS
    }
    neutral_indexed = {
        token: {item_key(row): row for row in payload["rows"]}
        for token, payload in neutral_payloads.items()
    }
    keys = sorted(indexed[TOKENS[0]])
    if any(sorted(rows) != keys for rows in indexed.values()):
        raise ValueError("checkpoint item sets do not match")
    neutral_keys = sorted(neutral_indexed[TOKENS[0]])
    if any(sorted(rows) != neutral_keys for rows in neutral_indexed.values()):
        raise ValueError("neutral checkpoint item sets do not match")

    n_layers = payloads[TOKENS[0]]["n_layers"]
    early_last = int(np.floor(.75 * (n_layers - 1)))
    matrices = {
        token: np.asarray([
            indexed[token][key]["recovery_by_layer"] for key in keys
        ])
        for token in TOKENS
    }
    curves = {token: matrix.mean(axis=0) for token, matrix in matrices.items()}
    neutral_matrices = {
        token: np.asarray([
            neutral_indexed[token][key]["recovery_by_layer"] for key in neutral_keys
        ])
        for token in TOKENS
    }
    neutral_curves = {
        token: matrix.mean(axis=0) for token, matrix in neutral_matrices.items()
    }

    rng = np.random.default_rng(20260824)
    absolute_rng = np.random.default_rng(20260825)
    comparisons = {}
    neutral_comparisons = {}
    difference_in_differences = {}
    # Each pair is ordered high-reversion, low-reversion. Positive delta means
    # causal source recovery is stronger in the faithful checkpoint.
    for high, low in ((1909, 1993), (2496, 2999)):
        high_early = matrices[high][:, :early_last + 1].mean(axis=1)
        low_early = matrices[low][:, :early_last + 1].mean(axis=1)
        paired_delta = low_early - high_early
        draws = np.empty(10_000)
        for draw in range(len(draws)):
            sample = rng.integers(0, len(keys), size=len(keys))
            draws[draw] = paired_delta[sample].mean()
        comparisons[f"{high}_to_{low}"] = {
            "direction": "low-reversion minus high-reversion",
            "early_depth_mean_delta": float(paired_delta.mean()),
            "paired_item_bootstrap_95": interval(draws),
            "bootstrap_p_delta_le_0": float(np.mean(draws <= 0)),
            "curve_delta": [float(x) for x in curves[low] - curves[high]],
        }
        neutral_high = neutral_matrices[high][:, :early_last + 1].mean(axis=1)
        neutral_low = neutral_matrices[low][:, :early_last + 1].mean(axis=1)
        neutral_delta = neutral_low - neutral_high
        neutral_draws = np.empty(10_000)
        did_draws = np.empty(10_000)
        for draw in range(len(neutral_draws)):
            conflict_sample = rng.integers(0, len(keys), size=len(keys))
            neutral_sample = rng.integers(
                0, len(neutral_keys), size=len(neutral_keys)
            )
            neutral_draws[draw] = neutral_delta[neutral_sample].mean()
            did_draws[draw] = (
                paired_delta[conflict_sample].mean()
                - neutral_delta[neutral_sample].mean()
            )
        neutral_comparisons[f"{high}_to_{low}"] = {
            "low_minus_high": float(neutral_delta.mean()),
            "bootstrap_95": interval(neutral_draws),
        }
        difference_in_differences[f"{high}_to_{low}"] = {
            "conflict_delta_minus_neutral_delta": float(
                paired_delta.mean() - neutral_delta.mean()
            ),
            "bootstrap_95": interval(did_draws),
            "bootstrap_p_le_0": float(np.mean(did_draws <= 0)),
        }

    checkpoints = {}
    for token in TOKENS:
        rows = [indexed[token][key] for key in keys]
        absolute_bootstrap = bootstrap_mean(
            matrices[token][:, :early_last + 1].mean(axis=1), absolute_rng
        )
        checkpoints[str(token)] = {
            "reversion": float(np.mean([row["reversion"] for row in rows])),
            "source_effect": float(np.mean([row["source_effect"] for row in rows])),
            "early_depth_recovery": float(
                matrices[token][:, :early_last + 1].mean()
            ),
            "early_depth_recovery_bootstrap_95": interval(absolute_bootstrap),
            "recovery_curve": [float(x) for x in curves[token]],
            "category_early_depth_recovery": {
                category: float(matrices[token][
                    np.asarray([key[0] == category for key in keys]),
                    :early_last + 1,
                ].mean())
                for category in sorted({key[0] for key in keys})
            },
        }

    neutral_checkpoints = {}
    for token in TOKENS:
        rows = [neutral_indexed[token][key] for key in neutral_keys]
        neutral_checkpoints[str(token)] = {
            "x_loses_to_arbitrary_decoy": float(
                np.mean([row["reversion"] for row in rows])
            ),
            "source_effect": float(np.mean([row["source_effect"] for row in rows])),
            "early_depth_recovery": float(
                neutral_matrices[token][:, :early_last + 1].mean()
            ),
            "recovery_curve": [float(x) for x in neutral_curves[token]],
        }

    summary = {
        "status": "PHASE-B-SOURCE-PATCH",
        "scope": "source-answer-token residual patch; all matched items",
        "n_items": len(keys),
        "n_neutral_items": len(neutral_keys),
        "n_layers": n_layers,
        "pre_final_feature_layers": [0, early_last],
        "bootstrap_draws": 10_000,
        "absolute_bootstrap_seed": 20260825,
        "absolute_bootstrap_unit": "matched item",
        "checkpoints": checkpoints,
        "matched_transition_comparisons": comparisons,
        "neutral_control_checkpoints": neutral_checkpoints,
        "neutral_transition_comparisons": neutral_comparisons,
        "conflict_specific_difference_in_differences": difference_in_differences,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    for token in TOKENS:
        ax.plot(range(n_layers), curves[token], marker="o", label=f"{token}B")
        ax.plot(
            range(n_layers), neutral_curves[token], ls="--", alpha=.45,
            label=f"{token}B neutral",
        )
    ax.axvline(early_last, color="0.4", ls="--", lw=1, label="0.75-depth cutoff")
    ax.set_xlabel("layer")
    ax.set_ylabel("source-donor patch effect (X-Y logit diff.)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT, dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
