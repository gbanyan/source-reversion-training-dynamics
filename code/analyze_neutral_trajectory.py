"""Relate neutral-subject source gain to conflict reversion over 24 checkpoints."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
OUT = HERE / "neutral_trajectory_summary.json"
PLOT = HERE / "neutral_vs_conflict_trajectory.png"


def loo_linear(y: np.ndarray, features: np.ndarray) -> dict:
    predictions = np.empty_like(y)
    design = np.column_stack([np.ones(len(y)), features])
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        coefficients = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
        predictions[held_out] = design[held_out] @ coefficients
    return {
        "mae": float(np.mean(np.abs(y - predictions))),
        "rmse": float(np.sqrt(np.mean((y - predictions) ** 2))),
        "predictions": [float(value) for value in predictions],
    }


def main() -> None:
    rows = []
    for shard in (0, 1):
        rows.extend(json.loads(
            (HERE / f"neutral_trajectory_shard{shard}.json").read_text()
        ))
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["training_tokens_b"])].append(row)

    phase_a = json.loads((HERE / "phase_a_summary.json").read_text())
    tokens = phase_a["tokens_b"]
    reversion = np.asarray(phase_a["curves"]["reversion"])
    prior = np.asarray(phase_a["curves"]["prior_strength"])
    neutral_gain = np.asarray([
        np.mean([row["source_effect"] for row in grouped[token]])
        for token in tokens
    ])
    neutral_error = np.asarray([
        np.mean([row["clean_error"] for row in grouped[token]])
        for token in tokens
    ])

    rho_gain = spearmanr(reversion, neutral_gain)
    rho_error = spearmanr(reversion, neutral_error)
    baseline = loo_linear(reversion, prior[:, None])
    augmented = loo_linear(reversion, np.column_stack([prior, neutral_gain]))
    summary = {
        "status": "FULL-TRAJECTORY-NEUTRAL-CONTROL",
        "n_checkpoints": len(tokens),
        "n_items_per_checkpoint": len(grouped[tokens[0]]),
        "tokens_b": tokens,
        "conflict_reversion": [float(value) for value in reversion],
        "neutral_source_gain": [float(value) for value in neutral_gain],
        "neutral_error_rate": [float(value) for value in neutral_error],
        "checkpoint_spearman": {
            "reversion_vs_neutral_source_gain": {
                "rho": float(rho_gain.statistic), "p": float(rho_gain.pvalue),
            },
            "reversion_vs_neutral_error_rate": {
                "rho": float(rho_error.statistic), "p": float(rho_error.pvalue),
            },
        },
        "leave_one_checkpoint_out_rate_prediction": {
            "prior_only": baseline,
            "prior_plus_neutral_source_gain": augmented,
            "mae_improvement": baseline["mae"] - augmented["mae"],
            "rmse_improvement": baseline["rmse"] - augmented["rmse"],
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    fig, left = plt.subplots(figsize=(9, 5))
    right = left.twinx()
    left.plot(tokens, reversion, "o-", color="C3", label="conflict reversion")
    right.plot(tokens, neutral_gain, "s--", color="C0", label="neutral source gain")
    left.set_xlabel("stage-1 training tokens (billions)")
    left.set_ylabel("conflict reversion rate", color="C3")
    right.set_ylabel("neutral-subject source margin gain", color="C0")
    left.grid(alpha=.25)
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
