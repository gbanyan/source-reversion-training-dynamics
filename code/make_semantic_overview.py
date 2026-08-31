#!/usr/bin/env python3
"""Render the cross-family semantic summary from its machine-readable JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    if not summary.get("passed"):
        raise SystemExit("refusing to plot a failed pooled summary")
    output = args.output or args.summary.with_name("semantic_pool_overview.png")

    axes = summary["axes"]
    families = summary["families"]
    family_rows = {
        (row["family"], row["axis"]): row for row in summary["family_axis"]
    }
    pooled_rows = {row["axis"]: row for row in summary["pooled_axis"]}
    x = np.arange(len(axes), dtype=float)
    width = 0.32
    colors = {"amber": "#2a9d8f", "pythia6p9": "#e76f51"}
    labels = {"amber": "Amber", "pythia6p9": "Pythia-6.9B"}

    fig, panels = plt.subplots(1, 2, figsize=(8.6, 3.9), constrained_layout=True)
    measures = (
        (
            "behavior",
            "source_gain_mean_equal_template_weight",
            "Source gain (X-Y logit diff.)",
        ),
        (
            "source_patch",
            "recovery_max_mean_equal_template_weight",
            "Source-donor patch effect (X-Y logit diff.)",
        ),
    )
    template_labels_used: set[str] = set()
    template_markers = {"A": "o", "B": "s", "C": "^"}
    for panel, (section, key, ylabel) in zip(panels, measures):
        for index, family in enumerate(families):
            rows = [family_rows[(family, axis)] for axis in axes]
            values = [row[section][key] for row in rows]
            bars = panel.bar(
                x + (index - 0.5) * width,
                values,
                width,
                color=colors.get(family, "#457b9d"),
                label=labels.get(family, family),
            )
            panel.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
            # Keep the pooled bars readable while exposing the template-level
            # spread and uncertainty used by the semantic gate.  These points
            # are descriptive estimates, not independent checkpoint samples.
            for axis_index, row in enumerate(rows):
                templates = row[section]["templates"]
                for template_index, template in enumerate(("A", "B", "C")):
                    entry = templates[template]
                    if section == "behavior":
                        value = entry["source_gain_mean"]
                        interval = entry["source_gain_ci95"]
                    else:
                        value = entry["recovery_max_mean"]
                        interval = entry["recovery_max_ci95"]
                    xpos = x[axis_index] + (index - 0.5) * width + (template_index - 1) * 0.055
                    panel.errorbar(
                        xpos,
                        value,
                        yerr=[[value - interval[0]], [interval[1] - value]],
                        fmt=template_markers[template],
                        ms=3.0,
                        color="#202020",
                        mfc="white",
                        mec="#202020",
                        capsize=2,
                        lw=0.8,
                        alpha=0.8,
                        label=(f"Template {template} (95% CI)" if template not in template_labels_used else None),
                    )
                    template_labels_used.add(template)
        pooled = [pooled_rows[axis]["equal_family_weight_" + ("mean_source_gain" if section == "behavior" else "mean_source_patch_recovery")] for axis in axes]
        panel.plot(x, pooled, "kD", ms=5, label="Equal-family mean")
        panel.set_xticks(x, [axis.capitalize() for axis in axes])
        panel.set_ylabel(ylabel)
        panel.axhline(0, color="#333333", lw=0.8)
        panel.grid(axis="y", alpha=0.22, lw=0.7)
        panel.set_axisbelow(True)
        handles, legend_labels = panel.get_legend_handles_labels()
        desired = ["Amber", "Pythia-6.9B", "Equal-family mean",
                   "Template A (95% CI)", "Template B (95% CI)",
                   "Template C (95% CI)"]
        by_label = dict(zip(legend_labels, handles))
        ordered = [label for label in desired if label in by_label]
        panel.legend([by_label[label] for label in ordered], ordered,
                     frameon=False, fontsize=7.5, loc="upper left")
        panel.set_ylim(bottom=0)

    fig.suptitle("Cross-family semantic source effects at frozen peaks", fontsize=11)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {output}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
