"""Fairness, representation, and disaggregated evaluation metrics for AFL.

Computes:
- Jain's Fairness Index across all configured clients (including zero contributors)
- Contribution shares by client hardware speed tier
- Training label exposure among applied updates
- Per-class held-out accuracy, macro accuracy, worst-class accuracy, and support counts
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_jains_fairness_index(
    contributions: List[Union[int, float]],
) -> Tuple[Optional[float], Optional[str]]:
    """Compute Jain's Fairness Index across all configured clients.

    Formula:
        J(x) = (sum_{i=1}^N x_i)^2 / (N * sum_{i=1}^N x_i^2)

    Args:
        contributions: List of client contribution counts (must include zero-contributors).

    Returns:
        (jains_index, reason_if_null)
        If total sum is zero, returns (None, "zero_total_contributions").
    """
    n = len(contributions)
    if n == 0:
        return None, "empty_population"

    total = float(sum(contributions))
    if total <= 0.0:
        return None, "zero_total_contributions"

    sum_squares = float(sum(float(x) ** 2 for x in contributions))
    if sum_squares <= 0.0:
        return None, "zero_total_contributions"

    index_val = (total ** 2) / (float(n) * sum_squares)
    return float(round(index_val, 6)), None


def compute_tier_shares(
    client_applied: Dict[int, int],
    client_tiers: Dict[int, str],
) -> Dict[str, float]:
    """Calculate the share of applied updates contributed by each speed tier."""
    total_applied = sum(client_applied.values())
    if total_applied == 0:
        return {t: 0.0 for t in set(client_tiers.values())}

    tier_counts: Dict[str, int] = {}
    for cid, count in client_applied.items():
        tier = client_tiers.get(cid, "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + count

    return {tier: round(count / total_applied, 6) for tier, count in tier_counts.items()}


def compute_label_exposure(
    client_applied: Dict[int, int],
    client_label_counts: Dict[int, Dict[int, int]],
) -> Dict[str, Any]:
    """Compute the distribution of training labels exposed to the global model through applied updates.

    Args:
        client_applied: Map from client_id to applied update count.
        client_label_counts: Map from client_id to dict {class_id: sample_count_on_client}.

    Returns:
        Dict with total_samples_exposed, per_class_counts, and per_class_exposure_fractions.
    """
    exposed_per_class: Dict[int, int] = {}
    total_samples = 0

    for cid, applied_count in client_applied.items():
        if applied_count <= 0:
            continue
        c_labels = client_label_counts.get(cid, {})
        for label, count in c_labels.items():
            samples = applied_count * count
            exposed_per_class[label] = exposed_per_class.get(label, 0) + samples
            total_samples += samples

    fractions = {}
    if total_samples > 0:
        for label, count in exposed_per_class.items():
            fractions[str(label)] = round(count / total_samples, 6)

    return {
        "total_exposed_samples": total_samples,
        "per_class_exposed_counts": {str(k): v for k, v in exposed_per_class.items()},
        "per_class_exposure_fractions": fractions,
    }


def evaluate_per_class_held_out(
    model: nn.Module,
    data_loader: DataLoader,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate accuracy on a held-out dataset disaggregated by class label without mutating RNG state."""
    import numpy as np
    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    try:
        if device is None:
            device = torch.device("cpu")

        was_training = model.training
        model.eval()

        correct_by_class = [0] * num_classes
        support_by_class = [0] * num_classes

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                preds = torch.argmax(outputs, dim=1)

                for t, p in zip(targets.view(-1), preds.view(-1)):
                    t_idx = int(t.item())
                    if 0 <= t_idx < num_classes:
                        support_by_class[t_idx] += 1
                        if t_idx == int(p.item()):
                            correct_by_class[t_idx] += 1

        if was_training:
            model.train()

        per_class_acc: Dict[str, Optional[float]] = {}
        per_class_sup: Dict[str, int] = {}
        valid_accuracies: List[float] = []
        worst_class: Optional[int] = None
        worst_acc: float = 1.0

        for c in range(num_classes):
            sup = support_by_class[c]
            per_class_sup[str(c)] = sup
            if sup > 0:
                acc = float(correct_by_class[c] / sup)
                per_class_acc[str(c)] = round(acc, 6)
                valid_accuracies.append(acc)
                if acc < worst_acc:
                    worst_acc = acc
                    worst_class = c
            else:
                per_class_acc[str(c)] = None

        macro_acc = round(float(sum(valid_accuracies) / len(valid_accuracies)), 6) if valid_accuracies else None
        worst_acc_val = round(worst_acc, 6) if valid_accuracies else None

        return {
            "per_class_accuracy": per_class_acc,
            "per_class_support": per_class_sup,
            "macro_accuracy": macro_acc,
            "worst_class_accuracy": worst_acc_val,
            "worst_class_id": worst_class,
        }
    finally:
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)


def compile_fairness_summary(
    all_client_ids: List[int],
    client_arrivals: Dict[int, int],
    client_applied: Dict[int, int],
    client_samples_uploaded: Dict[int, int],
    client_tiers: Dict[int, str],
    client_label_counts: Dict[int, Dict[int, int]],
    held_out_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile a complete fairness and representation diagnostic report."""
    contributions = [client_applied.get(cid, 0) for cid in all_client_ids]
    jains_val, jains_reason = compute_jains_fairness_index(contributions)
    tier_shares = compute_tier_shares(client_applied, client_tiers)
    label_exposure = compute_label_exposure(client_applied, client_label_counts)

    return {
        "jains_fairness_index": jains_val,
        "jains_fairness_reason": jains_reason,
        "total_configured_clients": len(all_client_ids),
        "zero_contribution_clients_count": sum(1 for c in contributions if c == 0),
        "client_arrivals": {str(cid): client_arrivals.get(cid, 0) for cid in all_client_ids},
        "client_applied": {str(cid): client_applied.get(cid, 0) for cid in all_client_ids},
        "client_samples_uploaded": {str(cid): client_samples_uploaded.get(cid, 0) for cid in all_client_ids},
        "tier_contribution_shares": tier_shares,
        "label_exposure": label_exposure,
        "held_out_class_metrics": held_out_metrics,
    }
