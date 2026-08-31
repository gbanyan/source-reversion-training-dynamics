"""Analyze the OLMo training-trajectory discovery scan.

The bootstrap unit is a unique fact tuple, not the repeated sampled row.  The
non-monotonic excursion statistic is the largest peak that rises above both an
earlier and a later trough.  Because this exact statistic was finalized after
seeing the discovery curve, its interval is descriptive and must not be called
confirmatory.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROWS_PATHS = [
    HERE / "phase_a_rows.json",
    HERE / "phase_a_dense_rows.json",
    HERE / "phase_a_dense_rows_external.json",
]
DETERMINISTIC_PATHS = [
    HERE / "phase_a_det_shard0.json",
    HERE / "phase_a_det_shard1.json",
]
SUMMARY_PATH = HERE / "phase_a_summary.json"
PLOT_PATH = HERE / "phase_a_curve.png"


def item_key(row: dict) -> tuple[str, str, str, str]:
    return row["cat"], row["subj"], row["x"], row["y"]


def aggregate(rows: list[dict]) -> tuple[list[int], list[tuple], dict[str, np.ndarray]]:
    tokens = sorted({int(row["training_tokens_b"]) for row in rows})
    keys = sorted({item_key(row) for row in rows})
    grouped: dict[tuple[int, tuple], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["training_tokens_b"]), item_key(row)].append(row)

    metrics = {}
    for metric in ("reversion", "prior_strength", "dm"):
        matrix = np.empty((len(keys), len(tokens)), dtype=float)
        for i, key in enumerate(keys):
            for j, token in enumerate(tokens):
                values = [float(row[metric]) for row in grouped[token, key]]
                if not values:
                    raise ValueError(f"missing {metric}: tokens={token}, item={key}")
                matrix[i, j] = np.mean(values)
        metrics[metric] = matrix
    return tokens, keys, metrics


def peak_excursion(curve: np.ndarray) -> dict:
    best = {"excursion": float("-inf")}
    for peak in range(1, len(curve) - 1):
        before = int(np.argmin(curve[:peak]))
        after = peak + 1 + int(np.argmin(curve[peak + 1 :]))
        excursion = min(curve[peak] - curve[before], curve[peak] - curve[after])
        if excursion > best["excursion"]:
            best = {
                "excursion": float(excursion),
                "before_index": before,
                "peak_index": peak,
                "after_index": after,
            }
    return best


def bootstrap_excursion(matrix: np.ndarray, n_boot: int = 10_000) -> np.ndarray:
    rng = np.random.default_rng(20260823)
    values = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        sample = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        values[draw] = peak_excursion(matrix[sample].mean(axis=0))["excursion"]
    return values


def subset_summary(matrix: np.ndarray, tokens: list[int]) -> dict:
    curve = matrix.mean(axis=0)
    peak = peak_excursion(curve)
    return {
        "n_items": int(matrix.shape[0]),
        "curve": [float(value) for value in curve],
        "excursion": peak["excursion"],
        "before_tokens_b": tokens[peak["before_index"]],
        "peak_tokens_b": tokens[peak["peak_index"]],
        "after_tokens_b": tokens[peak["after_index"]],
    }


def main() -> None:
    available_paths = (
        DETERMINISTIC_PATHS
        if all(path.exists() for path in DETERMINISTIC_PATHS)
        else [path for path in ROWS_PATHS if path.exists()]
    )
    rows = [
        row
        for path in available_paths
        for row in json.loads(path.read_text())
    ]
    tokens, keys, metrics = aggregate(rows)
    reversion = metrics["reversion"]
    curves = {name: matrix.mean(axis=0) for name, matrix in metrics.items()}
    peak = peak_excursion(curves["reversion"])
    boot = bootstrap_excursion(reversion)

    category_results = {}
    for category in sorted({key[0] for key in keys}):
        mask = np.asarray([key[0] == category for key in keys])
        category_results[category] = subset_summary(reversion[mask], tokens)

    split_results = {}
    for split in (0, 1):
        mask = np.asarray([
            int(hashlib.sha256("\x1f".join(key).encode()).hexdigest(), 16) % 2 == split
            for key in keys
        ])
        split_results[str(split)] = subset_summary(reversion[mask], tokens)

    rho_prior = spearmanr(curves["reversion"], curves["prior_strength"])
    rho_dm = spearmanr(curves["reversion"], curves["dm"])
    summary = {
        "status": (
            "DETERMINISTIC-REPLICATION"
            if available_paths == DETERMINISTIC_PATHS
            else "DISCOVERY-SIGNAL"
        ),
        "scope": (
            "OLMo-2-0425-1B stage-1 trajectory; rerun after deterministic item-generator fix"
            if available_paths == DETERMINISTIC_PATHS
            else "OLMo-2-0425-1B stage-1 trajectory; descriptive, not confirmatory"
        ),
        "row_sources": [path.name for path in available_paths],
        "n_raw_rows": len(rows),
        "n_unique_items": len(keys),
        "tokens_b": tokens,
        "curves": {name: [float(value) for value in curve] for name, curve in curves.items()},
        "nonmonotonic_excursion": {
            "estimate": peak["excursion"],
            "bootstrap_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "bootstrap_p_excursion_le_0_05": float(np.mean(boot <= 0.05)),
            "before_tokens_b": tokens[peak["before_index"]],
            "peak_tokens_b": tokens[peak["peak_index"]],
            "after_tokens_b": tokens[peak["after_index"]],
        },
        "checkpoint_spearman": {
            "reversion_vs_prior": {"rho": float(rho_prior.statistic), "p": float(rho_prior.pvalue)},
            "reversion_vs_dm": {"rho": float(rho_dm.statistic), "p": float(rho_dm.pvalue)},
        },
        "category_robustness": category_results,
        "deterministic_half_split_robustness": split_results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    fig, left = plt.subplots(figsize=(9, 5))
    right = left.twinx()
    left.plot(tokens, curves["reversion"], "o-", color="C3", lw=2.2, label="reversion")
    right.plot(tokens, curves["prior_strength"], "s--", color="C0", label="prior strength")
    right.plot(tokens, curves["dm"], "^--", color="C2", label="source contribution (dm)")
    left.set_xlabel("stage-1 training tokens (billions)")
    left.set_ylabel("reversion rate", color="C3")
    right.set_ylabel("candidate-margin units")
    left.set_ylim(0, 0.5)
    left.grid(alpha=0.25)
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
