"""Render a compact, regeneration-safe overview of the confirmatory results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def read(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def main() -> None:
    one = read("phase_a_summary.json")
    seven = read("phase_a_7b_summary.json")
    causal_one = read("phase_b_deterministic_summary.json")
    causal_seven = read("phase_b_7b_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(one["tokens_b"], one["curves"]["reversion"], "o-", ms=4)
    axes[0].axvspan(2496, 2999, color="#2a9d8f", alpha=.12)
    axes[0].set(title="1B deterministic trajectory", xlabel="training tokens (B)", ylabel="reversion rate")

    axes[1].plot(seven["tokens_b"], seven["curves"]["reversion"], "o-", color="#e76f51", ms=5)
    axes[1].axvspan(3201, 3896, color="#e76f51", alpha=.12)
    axes[1].set(title="7B independent trajectory", xlabel="training tokens (B)")
    # Use a shared visual scale so the two trajectories are comparable by eye.
    axes[0].set_ylim(0, 0.55)
    axes[1].set_ylim(0, 0.55)

    labels, means, lows, highs = [], [], [], []
    for template in ("A", "B", "C"):
        group = causal_one["groups"]["conflict"][template]
        mean = group["low_2999_minus_high_2496_early_recovery"]
        low, high = group["paired_bootstrap_95"]
        labels.append(f"1B-{template}")
        means.append(mean); lows.append(mean - low); highs.append(high - mean)
    for template in ("A", "B"):
        group = causal_seven["groups"]["conflict"][template]
        mean = group["low_3896_minus_high_3201_early_recovery"]
        low, high = group["paired_bootstrap_95"]
        labels.append(f"7B-{template}")
        means.append(mean); lows.append(mean - low); highs.append(high - mean)
    x = np.arange(len(labels))
    axes[2].bar(x, means, color=["#457b9d"] * 3 + ["#e76f51"] * 2)
    axes[2].errorbar(x, means, yerr=[lows, highs], fmt="none", ecolor="black", capsize=3)
    axes[2].axhline(0, color="0.3", lw=1)
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    axes[2].set(
        title="Causal patch-effect change",
        ylabel="late-minus-early patch effect (one-step X-Y log-odds)",
    )
    for ax in axes:
        ax.grid(alpha=.2)
        ax.tick_params(axis="both", labelsize=9)
        ax.title.set_fontsize(11)
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
    axes[2].yaxis.label.set_size(9)
    fig.tight_layout()
    fig.savefig(HERE / "paper_results_overview.png", dpi=200)


if __name__ == "__main__":
    main()
