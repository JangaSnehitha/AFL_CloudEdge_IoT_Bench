"""Parameter and update divergence metrics for Asynchronous Federated Learning.

Computes parameter-level predictors of aggregation impact:
- Model-version staleness
- Update L2 norm (||delta_i||_2)
- Relative update norm (||delta_i||_2 / (||base_i||_2 + eps))
- Normalized current-model divergence (||client_i - server_t||_2 / (||server_t||_2 + eps))
- Cosine disagreement with previous server update (1 - cos(delta_i, delta_prev))
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn


def flatten_params(
    model_or_params: Union[nn.Module, List[Any], Dict[str, Any], torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """Flatten model parameters into a single 1D float32 tensor from PyTorch models, tensors, or NumPy arrays."""
    if isinstance(model_or_params, torch.Tensor):
        return model_or_params.detach().flatten().float()
    if isinstance(model_or_params, np.ndarray):
        return torch.from_numpy(model_or_params).flatten().float()

    if isinstance(model_or_params, nn.Module):
        tensors = [p.detach().flatten().float() for p in model_or_params.parameters()]
    elif isinstance(model_or_params, dict):
        tensors = [
            torch.from_numpy(v).flatten().float() if isinstance(v, np.ndarray) else v.detach().flatten().float()
            for v in model_or_params.values()
        ]
    elif isinstance(model_or_params, (list, tuple)):
        tensors = [
            torch.from_numpy(t).flatten().float() if isinstance(t, np.ndarray) else t.detach().flatten().float()
            for t in model_or_params
        ]
    else:
        raise TypeError(f"Unsupported type for model_or_params: {type(model_or_params)}")

    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(tensors)


def compute_update_l2_norm(
    client_params: Any,
    base_params: Any,
) -> float:
    """Compute the L2 norm of the client parameter step delta_i = client - base."""
    c_flat = flatten_params(client_params)
    b_flat = flatten_params(base_params)
    delta = c_flat - b_flat
    norm_val = torch.linalg.norm(delta).item()
    return float(norm_val)


def compute_relative_update_norm(
    client_params: Any,
    base_params: Any,
    epsilon: float = 1e-8,
) -> float:
    """Compute ||delta_i||_2 / (||base_i||_2 + eps)."""
    c_flat = flatten_params(client_params)
    b_flat = flatten_params(base_params)
    delta = c_flat - b_flat
    delta_norm = torch.linalg.norm(delta).item()
    base_norm = torch.linalg.norm(b_flat).item()
    return float(delta_norm / (base_norm + epsilon))


def compute_model_divergence(
    client_params: Any,
    server_params: Any,
    epsilon: float = 1e-8,
) -> float:
    """Compute ||client_i - server_t||_2 / (||server_t||_2 + eps)."""
    c_flat = flatten_params(client_params)
    s_flat = flatten_params(server_params)
    diff = c_flat - s_flat
    diff_norm = torch.linalg.norm(diff).item()
    server_norm = torch.linalg.norm(s_flat).item()
    return float(diff_norm / (server_norm + epsilon))


def compute_cosine_disagreement(
    delta: Any,
    prev_server_delta: Optional[Any],
    epsilon: float = 1e-8,
    zero_threshold: float = 1e-12,
) -> Tuple[Optional[float], bool, str]:
    """Compute cosine disagreement 1 - cos(delta, prev_server_delta).

    Returns:
        (cosine_disagreement, is_available, reason_str)
    """
    if prev_server_delta is None:
        return None, False, "first_aggregation_no_previous_update"

    d_flat = flatten_params(delta)
    p_flat = flatten_params(prev_server_delta)

    d_norm = torch.linalg.norm(d_flat).item()
    if d_norm < zero_threshold:
        return None, False, "zero_norm_client_update"

    p_norm = torch.linalg.norm(p_flat).item()
    if p_norm < zero_threshold:
        return None, False, "zero_norm_previous_server_update"

    dot_prod = torch.dot(d_flat, p_flat).item()
    denom = max(d_norm * p_norm, epsilon)
    cos_sim = dot_prod / denom
    # Clamp to [-1.0, 1.0] for numerical stability
    cos_sim = max(-1.0, min(1.0, cos_sim))
    disagreement = 1.0 - cos_sim
    return float(round(disagreement, 6)), True, "ok"


def compute_all_predictors(
    client_params: Any,
    base_params: Any,
    server_params: Any,
    prev_server_delta: Optional[Any] = None,
    client_train_version: int = 0,
    server_version: int = 0,
    local_loss: Optional[float] = None,
    epsilon: float = 1e-8,
) -> Dict[str, Any]:
    """Extract all parameter-based and staleness predictors for an arriving client update."""
    c_flat = flatten_params(client_params)
    b_flat = flatten_params(base_params)
    s_flat = flatten_params(server_params)
    p_flat = flatten_params(prev_server_delta) if prev_server_delta is not None else None

    delta = c_flat - b_flat
    staleness = int(server_version - client_train_version)
    if staleness < 0:
        raise ValueError(f"Negative staleness: server={server_version}, client={client_train_version}")

    l2_norm = compute_update_l2_norm(c_flat, b_flat)
    rel_norm = compute_relative_update_norm(c_flat, b_flat, epsilon=epsilon)
    model_div = compute_model_divergence(c_flat, s_flat, epsilon=epsilon)
    cos_val, cos_avail, cos_reason = compute_cosine_disagreement(delta, p_flat, epsilon=epsilon)

    return {
        "staleness": staleness,
        "update_l2_norm": round(l2_norm, 6),
        "relative_update_norm": round(rel_norm, 6),
        "current_model_divergence": round(model_div, 6),
        "cosine_disagreement": cos_val,
        "cosine_available": cos_avail,
        "cosine_reason": cos_reason,
        "local_loss": round(float(local_loss), 6) if local_loss is not None else None,
        "has_local_loss": local_loss is not None,
    }


def aggregate_buffer_predictors(
    member_predictors: List[Dict[str, Any]],
    joint_delta: Any,
    prev_server_delta: Optional[Any] = None,
    epsilon: float = 1e-8,
) -> Dict[str, Any]:
    """Aggregate individual member predictors across a FedBuff buffer into buffer-level predictors."""
    if not member_predictors:
        raise ValueError("Cannot aggregate empty member predictors list.")

    stales = [m["staleness"] for m in member_predictors]
    l2_norms = [m["update_l2_norm"] for m in member_predictors]
    rel_norms = [m["relative_update_norm"] for m in member_predictors]
    divs = [m["current_model_divergence"] for m in member_predictors]

    p_flat = flatten_params(prev_server_delta) if prev_server_delta is not None else None
    cos_val, cos_avail, cos_reason = compute_cosine_disagreement(
        flatten_params(joint_delta), p_flat, epsilon=epsilon
    )

    return {
        "buffer_size": len(member_predictors),
        "mean_staleness": round(float(sum(stales) / len(stales)), 4),
        "max_staleness": max(stales),
        "min_staleness": min(stales),
        "mean_update_l2_norm": round(float(sum(l2_norms) / len(l2_norms)), 6),
        "max_update_l2_norm": max(l2_norms),
        "mean_relative_norm": round(float(sum(rel_norms) / len(rel_norms)), 6),
        "max_relative_norm": max(rel_norms),
        "mean_divergence": round(float(sum(divs) / len(divs)), 6),
        "max_divergence": max(divs),
        "buffer_cosine_disagreement": cos_val,
        "buffer_cosine_available": cos_avail,
        "buffer_cosine_reason": cos_reason,
    }
