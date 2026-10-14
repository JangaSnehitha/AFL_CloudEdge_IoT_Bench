"""Harmful-impact ground truth evaluation for Asynchronous Federated Learning.

Ground truth is defined exclusively by change in validation loss on a fixed
centralized validation split:
    impact = val_loss_before - val_loss_after

- Positive impact: validation loss improved (decreased).
- Negative impact: validation loss worsened (increased).
- Harmful label: impact < -tolerance.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """Evaluate mean cross-entropy loss deterministically without mutating state or RNG.

    Args:
        model: PyTorch model.
        data_loader: DataLoader for held-out validation or test set.
        device: Torch device (defaults to CPU).

    Returns:
        Mean scalar loss over all batches.
    """
    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    try:
        if device is None:
            device = torch.device("cpu")

        was_training = model.training
        model.eval()

        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                total_loss += float(loss.item())
                total_samples += int(targets.size(0))

        if was_training:
            model.train()

        if total_samples == 0:
            return 0.0

        return float(total_loss / total_samples)
    finally:
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)


def compute_action_impact(
    val_loss_before: float,
    val_loss_after: float,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Calculate continuous impact and discrete harmful status.

    Args:
        val_loss_before: Loss on validation dataset immediately before aggregation.
        val_loss_after: Loss on validation dataset immediately after aggregation.
        tolerance: Numerical threshold below which loss increase is labeled harmful.

    Returns:
        Dict with continuous impact, binary harmful indicator, before/after losses.
    """
    impact = float(val_loss_before - val_loss_after)
    is_harmful = bool(impact < -tolerance)

    return {
        "val_loss_before": round(val_loss_before, 6),
        "val_loss_after": round(val_loss_after, 6),
        "impact": round(impact, 6),
        "is_harmful": is_harmful,
        "tolerance": tolerance,
    }
