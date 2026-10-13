"""Synchronous FedAvg server strategy with sample-size-weighted aggregation and global evaluation."""

from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.client import set_model_parameters
from src.model import MLPConfig, build_criterion, build_model
from src.trainer import evaluate


def aggregate_weighted_average(
    results: List[Tuple[List[np.ndarray], int]],
) -> List[np.ndarray]:
    """Aggregate model parameters using sample-size-weighted averaging.

    Args:
        results: List of tuples (weights_list, num_examples) from clients.

    Returns:
        Aggregated list of parameter ndarrays.

    Raises:
        ValueError: If results is empty or total sample count is <= 0.
    """
    if not results:
        raise ValueError("Cannot aggregate empty results list.")

    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples <= 0:
        raise ValueError("Total sample count must be positive.")

    # Weighted sum for each parameter array
    first_weights = results[0][0]
    n_layers = len(first_weights)

    aggregated: List[np.ndarray] = []
    for layer_idx in range(n_layers):
        layer_sum = np.zeros_like(first_weights[layer_idx], dtype=np.float64)
        for weights, num_examples in results:
            layer_sum += weights[layer_idx].astype(np.float64) * num_examples
        layer_avg = (layer_sum / total_examples).astype(first_weights[layer_idx].dtype)
        aggregated.append(layer_avg)

    return aggregated


def create_evaluate_fn(
    model_config: MLPConfig,
    val_loader: DataLoader,
    test_loader: DataLoader,
    history: Optional[List[Dict[str, Any]]] = None,
    initial_metrics: Optional[Dict[str, Any]] = None,
) -> Callable[[int, List[np.ndarray], Dict[str, Any]], Tuple[float, Dict[str, Any]]]:
    """Create a server-side evaluation function for global validation and test sets.

    Returns:
        evaluate_fn callback compatible with Flower Strategy.evaluate_fn.
    """
    eval_model = build_model(model_config)
    criterion = build_criterion()

    def evaluate_fn(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        set_model_parameters(eval_model, parameters)

        val_metrics = evaluate(eval_model, val_loader, criterion)
        test_metrics = evaluate(eval_model, test_loader, criterion)

        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["accuracy"])
        test_loss = float(test_metrics["loss"])
        test_acc = float(test_metrics["accuracy"])

        record = {
            "round": server_round,
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_accuracy": round(test_acc, 6),
        }

        if server_round == 0 and initial_metrics is not None:
            initial_metrics.update(record)
        elif server_round > 0 and history is not None:
            history.append(record)

        return val_loss, {
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }

    return evaluate_fn


class FedAvgStrategy(FedAvg):
    """Synchronous FedAvg strategy with sample-size-weighted aggregation."""

    def __init__(
        self,
        initial_parameters: Parameters,
        evaluate_fn: Optional[Callable] = None,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_evaluate_clients: int = 0,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=initial_parameters,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        """Aggregate fit results using sample-size-weighted averaging."""
        if not results:
            return None, {}

        # Do not aggregate if there are failures and we require all clients
        weights_and_counts = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_ndarrays = aggregate_weighted_average(weights_and_counts)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics: Dict[str, Any] = {
            "num_clients": len(results),
            "total_examples": sum(count for _, count in weights_and_counts),
        }
        return aggregated_parameters, metrics
