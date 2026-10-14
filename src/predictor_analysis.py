"""Predictor statistical analysis, ranking evaluation, and action table export.

Evaluates whether parameter/model divergence predictors indicate harmful
aggregation actions better than version staleness alone.

Includes:
- Spearman rank correlation with continuous impact
- Binary classification metrics: AUROC and Average Precision (PR-AUC)
- Robust handling of ties, constant predictors, and degenerate single-class data
- Action-level CSV export and multi-variable comparison routines
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import csv
import math
from pathlib import Path
import numpy as np


def compute_fractional_ranks(values: List[float]) -> np.ndarray:
    """Compute 1-based fractional ranks for an array of numbers, averaging ties."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # Sort indices
    order = np.argsort(arr)
    sorted_arr = arr[order]

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        # Average rank for ties (1-indexed)
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def compute_spearman_correlation(
    x: List[float],
    y: List[float],
) -> Tuple[Optional[float], Optional[str]]:
    """Compute Spearman's rank correlation coefficient between x and y.

    Returns:
        (correlation, reason_if_null)
    """
    if len(x) != len(y):
        raise ValueError(f"Lengths must match: len(x)={len(x)}, len(y)={len(y)}")
    if len(x) < 3:
        return None, "insufficient_observations"

    # Filter out NaNs/Nones if any
    valid_pairs = [(float(xi), float(yi)) for xi, yi in zip(x, y) if xi is not None and yi is not None and not (math.isnan(xi) or math.isnan(yi))]
    if len(valid_pairs) < 3:
        return None, "insufficient_observations"

    x_vals = [p[0] for p in valid_pairs]
    y_vals = [p[1] for p in valid_pairs]

    rx = compute_fractional_ranks(x_vals)
    ry = compute_fractional_ranks(y_vals)

    std_x = float(np.std(rx))
    std_y = float(np.std(ry))

    if std_x < 1e-12:
        return None, "constant_predictor"
    if std_y < 1e-12:
        return None, "constant_impact"

    cov = float(np.mean((rx - np.mean(rx)) * (ry - np.mean(ry))))
    corr = cov / (std_x * std_y)
    # Clamp to [-1.0, 1.0]
    corr = max(-1.0, min(1.0, corr))
    return float(round(corr, 6)), None


def compute_auroc(
    scores: List[float],
    labels: List[int],
) -> Tuple[Optional[float], Optional[str]]:
    """Compute Area Under the Receiver Operating Characteristic curve (AUROC).

    Assumes higher score indicates higher probability of positive label (label=1).

    Uses Mann-Whitney U statistic with proper tie handling.

    Returns:
        (auroc, reason_if_null)
    """
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must have the same length.")

    valid_pairs = [(float(s), int(l)) for s, l in zip(scores, labels) if s is not None and not math.isnan(s)]
    if len(valid_pairs) < 3:
        return None, "insufficient_observations"

    scores_clean = [p[0] for p in valid_pairs]
    labels_clean = [p[1] for p in valid_pairs]

    n_pos = sum(labels_clean)
    n_neg = len(labels_clean) - n_pos

    if n_pos == 0 or n_neg == 0:
        return None, "single_class_outcome"

    if np.std(scores_clean) < 1e-12:
        return None, "constant_predictor"

    ranks = compute_fractional_ranks(scores_clean)
    r_pos = sum(r for r, l in zip(ranks, labels_clean) if l == 1)

    u_stat = r_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc_val = u_stat / (float(n_pos) * float(n_neg))
    auroc_val = max(0.0, min(1.0, auroc_val))
    return float(round(auroc_val, 6)), None


def compute_average_precision(
    scores: List[float],
    labels: List[int],
) -> Tuple[Optional[float], Optional[str]]:
    """Compute Average Precision (area under precision-recall curve) with proper step integration.

    Assumes higher score indicates higher probability of positive label (label=1).

    Returns:
        (average_precision, reason_if_null)
    """
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must have the same length.")

    valid_pairs = [(float(s), int(l)) for s, l in zip(scores, labels) if s is not None and not math.isnan(s)]
    if len(valid_pairs) < 3:
        return None, "insufficient_observations"

    scores_clean = np.array([p[0] for p in valid_pairs], dtype=np.float64)
    labels_clean = np.array([p[1] for p in valid_pairs], dtype=np.int64)

    n_pos = int(np.sum(labels_clean))
    if n_pos == 0 or n_pos == len(labels_clean):
        return None, "single_class_outcome"

    if np.std(scores_clean) < 1e-12:
        return None, "constant_predictor"

    # Sort descending by score
    desc_order = np.argsort(-scores_clean)
    sorted_scores = scores_clean[desc_order]
    sorted_labels = labels_clean[desc_order]

    # Find thresholds at distinct score values
    distinct_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_indices, len(sorted_scores) - 1]

    tps = np.cumsum(sorted_labels)[threshold_indices]
    fps = (threshold_indices + 1) - tps

    recalls = tps / float(n_pos)
    precisions = tps / (tps + fps)

    # Prepend recall 0.0, precision 1.0
    r_diff = np.diff(np.r_[0.0, recalls])
    ap_val = float(np.sum(r_diff * precisions))
    ap_val = max(0.0, min(1.0, ap_val))

    return float(round(ap_val, 6)), None


def evaluate_single_predictor(
    predictor_name: str,
    predictor_values: List[Optional[float]],
    impacts: List[float],
    harmful_labels: List[int],
) -> Dict[str, Any]:
    """Evaluate a single continuous score as a predictor of harmful aggregation impact."""
    # Filter valid observations where predictor is present
    valid_indices = [i for i, v in enumerate(predictor_values) if v is not None and not math.isnan(v)]
    n_valid = len(valid_indices)

    if n_valid < 3:
        return {
            "predictor_name": predictor_name,
            "valid_observations": n_valid,
            "spearman_correlation": None,
            "spearman_reason": "insufficient_observations",
            "auroc": None,
            "auroc_reason": "insufficient_observations",
            "average_precision": None,
            "ap_reason": "insufficient_observations",
        }

    sub_scores = [float(predictor_values[i]) for i in valid_indices]
    sub_impacts = [float(impacts[i]) for i in valid_indices]
    sub_labels = [int(harmful_labels[i]) for i in valid_indices]

    spearman_corr, spearman_reason = compute_spearman_correlation(sub_scores, sub_impacts)
    auroc_val, auroc_reason = compute_auroc(sub_scores, sub_labels)
    ap_val, ap_reason = compute_average_precision(sub_scores, sub_labels)

    return {
        "predictor_name": predictor_name,
        "valid_observations": n_valid,
        "spearman_correlation": spearman_corr,
        "spearman_reason": spearman_reason,
        "auroc": auroc_val,
        "auroc_reason": auroc_reason,
        "average_precision": ap_val,
        "ap_reason": ap_reason,
    }


def analyze_action_observations(
    actions: List[Dict[str, Any]],
    predictors_to_evaluate: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Perform comprehensive predictor analysis across recorded aggregation actions."""
    if predictors_to_evaluate is None:
        predictors_to_evaluate = [
            "staleness",
            "update_l2_norm",
            "relative_update_norm",
            "current_model_divergence",
            "cosine_disagreement",
            "local_loss",
        ]

    n_actions = len(actions)
    if n_actions == 0:
        return {
            "action_count": 0,
            "harmful_count": 0,
            "harmful_prevalence": None,
            "predictors": {},
            "status": "no_actions_recorded",
        }

    impacts = [a["impact"] for a in actions]
    harmful_labels = [1 if a["is_harmful"] else 0 for a in actions]
    harmful_count = sum(harmful_labels)
    prevalence = round(float(harmful_count / n_actions), 6)

    pred_results: Dict[str, Any] = {}
    for pname in predictors_to_evaluate:
        p_vals = [a.get(pname) for a in actions]
        pred_results[pname] = evaluate_single_predictor(
            predictor_name=pname,
            predictor_values=p_vals,
            impacts=impacts,
            harmful_labels=harmful_labels,
        )

    return {
        "action_count": n_actions,
        "harmful_count": harmful_count,
        "harmful_prevalence": prevalence,
        "predictors": pred_results,
    }


def export_actions_to_csv(
    actions: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Export action observations to CSV with clean standard column headers."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "action_id",
        "scenario",
        "algorithm",
        "seed",
        "virtual_time",
        "client_id",
        "staleness",
        "update_l2_norm",
        "relative_update_norm",
        "current_model_divergence",
        "cosine_disagreement",
        "cosine_available",
        "local_loss",
        "val_loss_before",
        "val_loss_after",
        "impact",
        "is_harmful",
    ]

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for a in actions:
            row = dict(a)
            # Ensure None values export cleanly as empty strings
            for k in fieldnames:
                if row.get(k) is None:
                    row[k] = ""
            writer.writerow(row)
