"""Metric helpers for the centralized baseline."""

from typing import List

import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return top-1 accuracy in [0, 1].

    Args:
        logits: Raw model output of shape (N, C).
        labels: Ground-truth class indices of shape (N,).
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_mean(values: List[float]) -> float:
    """Return arithmetic mean of a non-empty list of floats."""
    if not values:
        raise ValueError("values list must not be empty")
    return sum(values) / len(values)
