"""Render candidate-choice decomposition figures from derived summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def read(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def plot(ax, payload: dict, title: str, transition: tuple[int, int]) -> None:
    tokens = payload["tokens_b"]
    curves = payload["curves"]
    ax.plot(tokens, curves["p_x"], "o-", label="source X", color="#2a9d8f", ms=3.5)
    ax.plot(tokens, curves["p_y"], "o-", label="prior Y", color="#e76f51", ms=3.5)
    ax.plot(tokens, curves["p_other"], "o-", label="distractor", color="#6c757d", ms=3.5)
    ax.axvspan(*transition, color="#264653", alpha=.10)
    ax.set_title(title)
    ax.set_xlabel("training tokens (B)")
    ax.set_ylabel("top-choice proportion")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.2)


def main() -> None:
    one = read("choice_decomposition_1b.json")
    seven = read("choice_decomposition_7b.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    plot(axes[0], one, "OLMo-2 1B", (2496, 2999))
    plot(axes[1], seven, "OLMo-2 7B", (3201, 3896))
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Candidate-choice decomposition of source reversion", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "choice_decomposition.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
