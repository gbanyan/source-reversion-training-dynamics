"""Freeze a pre-final routing predictor on the early transition and test later."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
TOKENS = (1909, 1993, 2496, 2999)
TEMPLATES = ("A", "B", "C")
OUT = HERE / "routing_predictor_summary.json"
BANDS = ((0, 4), (4, 8), (8, 12))


def load_rows() -> list[dict]:
    rows = []
    for token in TOKENS:
        for template in TEMPLATES:
            payload = json.loads(
                (HERE / f"routing_{token}_{template}.json").read_text()
            )
            rows.extend(payload["rows"])
    return rows


def features(row: dict) -> dict[str, float]:
    attention = np.asarray(row["attention_to_source"], dtype=float)
    path = np.asarray(row["source_path_readout"], dtype=float)
    result = {
        "erased_margin": float(row["erased_margin"]),
        "fame": float(row["fame"]),
        "cat_loc": float(row["cat"] == "loc"),
        "prefinal_margin_l11": float(row["prefinal_margin_l11"]),
    }
    for band_index, (start, stop) in enumerate(BANDS):
        attn_band = attention[start:stop]
        path_band = path[start:stop]
        result[f"attention_mean_b{band_index}"] = float(attn_band.mean())
        result[f"attention_max_b{band_index}"] = float(attn_band.max())
        result[f"path_sum_b{band_index}"] = float(
            path_band.sum(axis=1).mean()
        )
        result[f"path_positive_b{band_index}"] = float(
            np.maximum(path_band, 0).sum(axis=1).mean()
        )
    return result


def fit_logistic(x: np.ndarray, y: np.ndarray, regularization: float = .1) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        scores = weights[0] + z @ weights[1:]
        loss = np.mean(np.logaddexp(0, scores) - y * scores)
        loss += .5 * regularization * np.sum(weights[1:] ** 2)
        probability = 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
        gradient = np.r_[
            np.mean(probability - y),
            z.T @ (probability - y) / len(y) + regularization * weights[1:],
        ]
        return float(loss), gradient

    result = minimize(
        objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B"
    )
    if not result.success:
        raise RuntimeError(result.message)
    return {
        "mean": mean,
        "scale": scale,
        "weights": result.x,
        "regularization": regularization,
    }


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    scores = model["weights"][0] + z @ model["weights"][1:]
    return 1 / (1 + np.exp(-np.clip(scores, -30, 30)))


def evaluate(rows: list[dict], probability: np.ndarray, prevalence: float) -> dict:
    y = np.asarray([row["reversion_target"] for row in rows], dtype=float)
    brier = float(np.mean((probability - y) ** 2))
    prevalence_brier = float(np.mean((prevalence - y) ** 2))
    groups = {}
    for token in sorted({row["training_tokens_b"] for row in rows}):
        for template in sorted({row["template"] for row in rows}):
            mask = np.asarray([
                row["training_tokens_b"] == token and row["template"] == template
                for row in rows
            ])
            if not mask.any():
                continue
            groups[f"{token}_{template}"] = {
                "observed": float(y[mask].mean()),
                "predicted": float(probability[mask].mean()),
                "absolute_error": float(abs(y[mask].mean() - probability[mask].mean())),
            }
    return {
        "n": len(rows),
        "brier": brier,
        "brier_skill_vs_training_prevalence": 1 - brier / prevalence_brier,
        "mean_group_rate_error": float(np.mean([
            group["absolute_error"] for group in groups.values()
        ])),
        "groups": groups,
    }


def main() -> None:
    rows = load_rows()
    feature_rows = [features(row) for row in rows]
    train_mask = np.asarray([
        row["template"] == "A" and row["training_tokens_b"] in (1909, 1993)
        for row in rows
    ])
    train_rows = [row for row, keep in zip(rows, train_mask) if keep]
    y_train = np.asarray([row["reversion_target"] for row in train_rows], dtype=float)
    prevalence = float(y_train.mean())

    base = ["erased_margin", "fame", "cat_loc"]
    attention = [
        name for band in range(3)
        for name in (f"attention_mean_b{band}", f"attention_max_b{band}")
    ]
    path = [
        name for band in range(3)
        for name in (f"path_sum_b{band}", f"path_positive_b{band}")
    ]
    specifications = {
        "baseline": base,
        "baseline_plus_attention": base + attention,
        "baseline_plus_path": base + path,
        "baseline_plus_routing": base + attention + path,
        "baseline_plus_routing_and_l11": (
            base + attention + path + ["prefinal_margin_l11"]
        ),
    }

    split_masks = {
        "train_early_A": train_mask,
        "checkpoint_heldout_late_A": np.asarray([
            row["template"] == "A" and row["training_tokens_b"] in (2496, 2999)
            for row in rows
        ]),
        "template_heldout_early_BC": np.asarray([
            row["template"] in ("B", "C")
            and row["training_tokens_b"] in (1909, 1993)
            for row in rows
        ]),
        "joint_heldout_late_BC": np.asarray([
            row["template"] in ("B", "C")
            and row["training_tokens_b"] in (2496, 2999)
            for row in rows
        ]),
    }

    model_results = {}
    for model_name, names in specifications.items():
        matrix = np.asarray([[feature[name] for name in names] for feature in feature_rows])
        fitted = fit_logistic(matrix[train_mask], y_train)
        probabilities = predict(fitted, matrix)
        model_results[model_name] = {
            "features": names,
            "coefficients_standardized": {
                "intercept": float(fitted["weights"][0]),
                **{
                    name: float(value)
                    for name, value in zip(names, fitted["weights"][1:])
                },
            },
            "evaluations": {
                split: evaluate(
                    [row for row, keep in zip(rows, mask) if keep],
                    probabilities[mask],
                    prevalence,
                )
                for split, mask in split_masks.items()
            },
        }

    # Component selection is frozen using only early checkpoint/template-A data.
    early = {}
    for token in (1909, 1993):
        selected = [
            np.asarray(row["source_path_readout"], dtype=float)
            for row in rows
            if row["training_tokens_b"] == token and row["template"] == "A"
        ]
        early[token] = np.stack(selected)
    mean_delta = early[1993].mean(axis=0) - early[1909].mean(axis=0)
    pooled_sd = np.sqrt((early[1993].var(axis=0) + early[1909].var(axis=0)) / 2)
    standardized = mean_delta / np.maximum(pooled_sd, 1e-6)
    flat = [
        (layer, head, float(standardized[layer, head]), float(mean_delta[layer, head]))
        for layer in range(standardized.shape[0])
        for head in range(standardized.shape[1])
    ]
    top_heads = sorted(flat, key=lambda item: item[2], reverse=True)[:8]

    summary = {
        "protocol": {
            "training": "1909B+1993B, template A only",
            "frozen_tests": [
                "2496B+2999B template A",
                "1909B+1993B templates B/C",
                "2496B+2999B templates B/C",
            ],
            "last_feature_layer": 11,
            "final_conflict_margin_used_as_feature": False,
            "fixed_ridge_regularization": .1,
        },
        "training_prevalence": prevalence,
        "models": model_results,
        "frozen_top_source_heads": [
            {"layer": layer, "head": head, "standardized_delta": effect,
             "raw_delta": raw}
            for layer, head, effect, raw in top_heads
        ],
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
