"""Analyze the independent OLMo-2 7B trajectory replication scan."""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from analyze_phase_a import aggregate, bootstrap_excursion, peak_excursion
from analyze_neutral_trajectory import loo_linear


HERE = Path(__file__).resolve().parent
OUT = HERE / "phase_a_7b_summary.json"


def main() -> None:
    paths = [HERE / "phase_a_7b_shard0.json", HERE / "phase_a_7b_shard1.json"]
    rows = [row for path in paths for row in json.loads(path.read_text())]
    tokens, keys, metrics = aggregate(rows)
    curves = {name: matrix.mean(axis=0) for name, matrix in metrics.items()}
    peak = peak_excursion(curves["reversion"])
    boot = bootstrap_excursion(metrics["reversion"])

    one_b = json.loads((HERE / "phase_a_summary.json").read_text())
    progress_1b = np.asarray(one_b["tokens_b"]) / max(one_b["tokens_b"])
    progress_7b = np.asarray(tokens) / max(tokens)
    aligned_1b = np.interp(
        progress_7b, progress_1b, one_b["curves"]["reversion"]
    )
    cross_size = spearmanr(curves["reversion"], aligned_1b)
    neutral_paths = [HERE / "neutral_7b_shard0.json", HERE / "neutral_7b_shard1.json"]
    neutral_rows = [
        row for path in neutral_paths for row in json.loads(path.read_text())
    ]
    neutral_grouped = defaultdict(list)
    for row in neutral_rows:
        neutral_grouped[int(row["training_tokens_b"])].append(row)
    neutral_gain = np.asarray([
        np.mean([row["source_effect"] for row in neutral_grouped[token]])
        for token in tokens
    ])
    neutral_error = np.asarray([
        np.mean([row["clean_error"] for row in neutral_grouped[token]])
        for token in tokens
    ])
    rho_neutral_gain = spearmanr(curves["reversion"], neutral_gain)
    rho_neutral_error = spearmanr(curves["reversion"], neutral_error)
    prior_only = loo_linear(curves["reversion"], curves["prior_strength"][:, None])
    prior_plus_gain = loo_linear(
        curves["reversion"],
        np.column_stack([curves["prior_strength"], neutral_gain]),
    )
    estimate = peak["excursion"]
    lower = float(np.quantile(boot, .025))
    summary = {
        "status": (
            "INDEPENDENT-SIZE-SIGNAL"
            if estimate >= .10 and lower > .05
            else "INDEPENDENT-SIZE-NO-SIGNAL"
        ),
        "repo": "allenai/OLMo-2-1124-7B",
        "n_raw_rows": len(rows),
        "n_unique_items": len(keys),
        "tokens_b": tokens,
        "curves": {
            name: [float(value) for value in curve]
            for name, curve in curves.items()
        },
        "nonmonotonic_excursion": {
            "estimate": estimate,
            "bootstrap_95": [lower, float(np.quantile(boot, .975))],
            "before_tokens_b": tokens[peak["before_index"]],
            "peak_tokens_b": tokens[peak["peak_index"]],
            "after_tokens_b": tokens[peak["after_index"]],
        },
        "progress_aligned_cross_size": {
            "one_b_interpolated_reversion": [float(value) for value in aligned_1b],
            "spearman_rho": float(cross_size.statistic),
            "p": float(cross_size.pvalue),
        },
        "neutral_control": {
            "source_gain": [float(value) for value in neutral_gain],
            "error_rate": [float(value) for value in neutral_error],
            "reversion_vs_source_gain": {
                "rho": float(rho_neutral_gain.statistic),
                "p": float(rho_neutral_gain.pvalue),
            },
            "reversion_vs_neutral_error": {
                "rho": float(rho_neutral_error.statistic),
                "p": float(rho_neutral_error.pvalue),
            },
            "loo_rate_prediction": {
                "prior_only": prior_only,
                "prior_plus_neutral_gain": prior_plus_gain,
            },
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
