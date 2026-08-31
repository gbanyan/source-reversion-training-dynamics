"""Render the supplementary Phase-A trajectory figure from frozen JSON.

This plotting-only script does not recompute or overwrite the canonical
summary. It uses two shared-x panels so rates and margin diagnostics are not
placed on a dual y-axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def main() -> None:
    summary = json.loads((HERE / "phase_a_summary.json").read_text())
    tokens = summary["tokens_b"]
    curves = summary["curves"]

    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6), gridspec_kw={"height_ratios": [1, 1.2]}
    )
    top.plot(tokens, curves["reversion"], "o-", color="C3", lw=2.0, ms=4,
             label="reversion")
    top.axvspan(2496, 2999, color="#2a9d8f", alpha=0.12)
    top.set_ylabel("reversion rate")
    top.set_ylim(0, 0.55)
    top.grid(alpha=0.25)
    top.legend(loc="upper right")

    bottom.plot(tokens, curves["prior_strength"], "s--", color="C0", lw=1.8,
                ms=4, label="prior strength")
    bottom.plot(tokens, curves["dm"], "^--", color="C2", lw=1.8, ms=4,
                label="source contribution (dm)")
    bottom.axvspan(2496, 2999, color="#2a9d8f", alpha=0.12)
    bottom.set_xlabel("stage-1 training tokens (billions)")
    bottom.set_ylabel("candidate-margin units")
    bottom.grid(alpha=0.25)
    bottom.legend(loc="upper right")

    fig.suptitle("OLMo-2 1B checkpoint diagnostics", y=0.995)
    fig.tight_layout()
    fig.savefig(HERE / "phase_a_curve.png", dpi=180)
    fig.savefig(HERE / "phase_a_curve.pdf")


if __name__ == "__main__":
    main()
