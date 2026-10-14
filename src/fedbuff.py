"""Buffered Asynchronous Federated Learning (FedBuff) simulation foundation."""

import argparse
import collections
import copy
from dataclasses import asdict, dataclass
import heapq
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure package imports work when executed directly
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.client import (
    get_model_parameters,
    load_partitioned_dataloaders,
    set_model_parameters,
)
from src.dataset import SyntheticDatasetConfig
from src.model import MLPConfig, build_criterion, build_model, build_optimizer
from src.partition import PartitionConfig
from src.staleness import StalenessConfig, StalenessSimulator, save_client_updates_csv
from src.trainer import evaluate, train_one_epoch


@dataclass
class FedBuffConfig:
    """Configuration for FedBuff simulation."""

    n_clients: int = 10
    max_updates: int = 30  # Number of client arrivals to process
    buffer_size: int = 5   # Buffer capacity K before triggering aggregation
    flush_final_buffer: bool = False  # If True, flush incomplete buffer at end
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    partition_method: str = "dirichlet"
    alpha: float = 0.5
    seed: int = 42
    delay_distribution: str = "uniform"  # "uniform", "constant", "poisson", "exponential", "zero"
    max_delay: int = 3
    compute_time: float = 1.0  # Simulated time required per local epoch
    staleness_seed: Optional[int] = None
    min_samples_per_client: int = 5

    def __post_init__(self) -> None:
        if self.n_clients < 1:
            raise ValueError(f"n_clients must be >= 1, got {self.n_clients}")
        if self.max_updates < 1:
            raise ValueError(f"max_updates must be >= 1, got {self.max_updates}")
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {self.buffer_size}")
        if self.local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {self.local_epochs}")
        if self.max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {self.max_delay}")
        if self.compute_time < 0.0:
            raise ValueError(f"compute_time must be >= 0.0, got {self.compute_time}")
        if self.staleness_seed is None:
            self.staleness_seed = self.seed

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FedBuffConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def aggregate_buffered_parameters(
    buffered_updates: List[Dict[str, Any]],
) -> List[np.ndarray]:
    """Aggregate model parameters from buffered updates using sample-count-weighted averaging.

    theta_new = sum_{i in buffer} (n_i / N_buffer) * theta_i

    Args:
        buffered_updates: List of update dicts, each containing:
            - 'trained_params': List[np.ndarray]
            - 'sample_count': int

    Returns:
        Aggregated list of parameter ndarrays.

    Raises:
        ValueError: If buffered_updates is empty or total sample count is <= 0.
    """
    if not buffered_updates:
        raise ValueError("Cannot aggregate empty buffered updates list.")

    total_examples = sum(int(u["sample_count"]) for u in buffered_updates)
    if total_examples <= 0:
        raise ValueError(f"Total sample count must be positive, got {total_examples}")

    first_params = buffered_updates[0]["trained_params"]
    n_layers = len(first_params)

    aggregated: List[np.ndarray] = []
    for layer_idx in range(n_layers):
        layer_sum = np.zeros_like(first_params[layer_idx], dtype=np.float64)
        for u in buffered_updates:
            params = u["trained_params"]
            n_samples = u["sample_count"]
            layer_sum += params[layer_idx].astype(np.float64) * n_samples
        layer_avg = (layer_sum / total_examples).astype(first_params[layer_idx].dtype)
        aggregated.append(layer_avg)

    return aggregated


def train_client_locally(
    model_config: MLPConfig,
    train_loader: DataLoader,
    initial_params: List[np.ndarray],
    local_epochs: int,
    lr: float,
) -> Tuple[List[np.ndarray], float, int]:
    """Perform deterministic local client training starting from initial_params."""
    client_model = build_model(model_config)
    set_model_parameters(client_model, initial_params)
    criterion = build_criterion()
    opt_config = copy.deepcopy(model_config)
    opt_config.lr = lr
    optimizer = build_optimizer(client_model, opt_config)

    last_loss = 0.0
    for _ in range(local_epochs):
        metrics = train_one_epoch(client_model, train_loader, criterion, optimizer)
        last_loss = metrics.get("loss", 0.0)

    num_examples = len(train_loader.dataset)
    return get_model_parameters(client_model), float(last_loss), num_examples


class FedBuffSimulator:
    """Deterministic event-driven buffered asynchronous federated learning (FedBuff) simulator."""

    def __init__(
        self,
        config: FedBuffConfig,
        dataset_config: Optional[SyntheticDatasetConfig] = None,
        model_config: Optional[MLPConfig] = None,
    ) -> None:
        self.config = config

        # Seed global RNGs for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if dataset_config is None:
            ds_path = Path("configs/dataset_config.json")
            if ds_path.exists():
                self.dataset_config = SyntheticDatasetConfig.from_json(ds_path)
            else:
                self.dataset_config = SyntheticDatasetConfig()
        else:
            self.dataset_config = copy.deepcopy(dataset_config)
        self.dataset_config.seed = config.seed

        if model_config is None:
            model_path = Path("configs/model_config.json")
            if model_path.exists():
                self.model_config = MLPConfig.from_json(model_path)
            else:
                self.model_config = MLPConfig()
        else:
            self.model_config = copy.deepcopy(model_config)
        self.model_config.lr = config.lr

        # Partition dataset
        partition_config = PartitionConfig(
            n_clients=config.n_clients,
            method=config.partition_method,
            alpha=config.alpha,
            seed=config.seed,
            min_samples_per_client=config.min_samples_per_client,
        )

        (
            self.client_loaders,
            self.val_loader,
            self.test_loader,
            self.partition_stats,
        ) = load_partitioned_dataloaders(
            dataset_config=self.dataset_config,
            partition_config=partition_config,
            batch_size=config.batch_size,
            return_stats=True,
        )

        # Build server model and evaluation components
        self.server_model = build_model(self.model_config)
        self.server_params = get_model_parameters(self.server_model)
        self.criterion = build_criterion()

        # Simulated delay generator
        staleness_cfg = StalenessConfig(
            distribution=config.delay_distribution,
            max_delay=config.max_delay,
            seed=config.staleness_seed if config.staleness_seed is not None else config.seed,
        )
        self.delay_simulator = StalenessSimulator(staleness_cfg)

        # Simulation state
        self.server_version: int = 0
        self.current_time: float = 0.0
        self.arrival_count: int = 0
        self.applied_update_count: int = 0
        self.aggregation_count: int = 0
        self.event_counter: int = 0

        # Server-side buffer storing accepted update dicts
        self.buffer: List[Dict[str, Any]] = []

        # Event queue: priority queue storing (timestamp, priority, event_id, event_dict)
        # Priority order: DISPATCH (1), TRAIN_COMPLETE (2), UPDATE_ARRIVAL (3)
        self.event_queue: List[Tuple[float, int, int, Dict[str, Any]]] = []

        # Tracking logs
        self.event_log: List[Dict[str, Any]] = []
        self.arrival_log: List[Dict[str, Any]] = []
        self.aggregation_log: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.initial_metrics: Dict[str, Any] = {}

        self.client_arrival_counts: Dict[int, int] = collections.defaultdict(int)
        self.client_applied_counts: Dict[int, int] = collections.defaultdict(int)

    def _push_event(self, timestamp: float, priority: int, event_data: Dict[str, Any]) -> None:
        self.event_counter += 1
        heapq.heappush(self.event_queue, (timestamp, priority, self.event_counter, event_data))

    def _dispatch_client(self, client_id: int, dispatch_time: float) -> None:
        """Dispatch a client with the current server model version."""
        training_version = self.server_version
        training_params = copy.deepcopy(self.server_params)

        dispatch_record = {
            "time": round(float(dispatch_time), 4),
            "event_type": "DISPATCH",
            "client_id": client_id,
            "server_version": training_version,
        }
        self.event_log.append(dispatch_record)

        trained_params, local_loss, num_examples = train_client_locally(
            model_config=self.model_config,
            train_loader=self.client_loaders[client_id],
            initial_params=training_params,
            local_epochs=self.config.local_epochs,
            lr=self.config.lr,
        )

        compute_duration = float(self.config.compute_time * self.config.local_epochs)
        train_complete_time = dispatch_time + compute_duration

        train_complete_event = {
            "event_type": "TRAIN_COMPLETE",
            "client_id": client_id,
            "training_version": training_version,
            "trained_params": trained_params,
            "local_loss": local_loss,
            "num_examples": num_examples,
            "dispatch_time": dispatch_time,
            "train_complete_time": train_complete_time,
        }
        self._push_event(train_complete_time, 2, train_complete_event)

    def _aggregate_buffer(self, is_final_flush: bool = False) -> Dict[str, Any]:
        """Aggregate all client models currently in buffer using sample-count-weighted averaging."""
        if not self.buffer:
            return {}

        items_to_aggregate = list(self.buffer)
        total_samples = sum(u["sample_count"] for u in items_to_aggregate)
        server_version_before = self.server_version

        # Sample-count-weighted aggregation of buffered client updates
        self.server_params = aggregate_buffered_parameters(items_to_aggregate)
        set_model_parameters(self.server_model, self.server_params)

        self.server_version += 1
        self.aggregation_count += 1
        self.applied_update_count += len(items_to_aggregate)

        for u in items_to_aggregate:
            self.client_applied_counts[u["client_id"]] += 1

        # Clear buffer immediately after aggregation
        self.buffer = []

        # Evaluate updated server model
        val_metrics = evaluate(self.server_model, self.val_loader, self.criterion)
        test_metrics = evaluate(self.server_model, self.test_loader, self.criterion)

        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["accuracy"])
        test_loss = float(test_metrics["loss"])
        test_acc = float(test_metrics["accuracy"])

        staleness_list = [u["staleness"] for u in items_to_aggregate]
        mean_stale = float(np.mean(staleness_list)) if staleness_list else 0.0

        sample_weights_details = [
            {
                "client_id": u["client_id"],
                "sample_count": u["sample_count"],
                "weight": round(float(u["sample_count"] / total_samples), 6),
                "staleness": u["staleness"],
            }
            for u in items_to_aggregate
        ]

        agg_record = {
            "aggregation_index": self.aggregation_count,
            "time": round(float(self.current_time), 4),
            "server_version_before": server_version_before,
            "server_version_after": self.server_version,
            "buffer_size": len(items_to_aggregate),
            "client_ids": [u["client_id"] for u in items_to_aggregate],
            "per_update_staleness": staleness_list,
            "mean_staleness": round(mean_stale, 4),
            "sample_weighted_aggregation_details": sample_weights_details,
            "total_samples": total_samples,
            "is_final_flush": is_final_flush,
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_accuracy": round(test_acc, 6),
        }
        self.aggregation_log.append(agg_record)

        self.event_log.append({
            "time": round(float(self.current_time), 4),
            "event_type": "FINAL_BUFFER_FLUSHED" if is_final_flush else "AGGREGATE",
            "aggregation_index": self.aggregation_count,
            "server_version_before": server_version_before,
            "server_version_after": self.server_version,
            "buffer_size": len(items_to_aggregate),
            "client_ids": [u["client_id"] for u in items_to_aggregate],
            "per_update_staleness": staleness_list,
            "is_final_flush": is_final_flush,
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_accuracy": round(test_acc, 6),
        })

        self.history.append({
            "aggregation_index": self.aggregation_count,
            "server_version": self.server_version,
            "time": round(float(self.current_time), 4),
            "buffer_size": len(items_to_aggregate),
            "is_final_flush": is_final_flush,
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_accuracy": round(test_acc, 6),
        })

        return agg_record

    def run_simulation(self) -> Dict[str, Any]:
        """Execute the FedBuff event-driven simulation until max_updates arrivals are processed."""
        # Initial evaluation at round 0 / server_version 0
        init_val = evaluate(self.server_model, self.val_loader, self.criterion)
        init_test = evaluate(self.server_model, self.test_loader, self.criterion)
        self.initial_metrics = {
            "server_version": 0,
            "time": 0.0,
            "val_loss": round(float(init_val["loss"]), 6),
            "val_accuracy": round(float(init_val["accuracy"]), 6),
            "test_loss": round(float(init_test["loss"]), 6),
            "test_accuracy": round(float(init_test["accuracy"]), 6),
        }

        # Initially dispatch all clients at simulated time t=0.0
        for cid in range(self.config.n_clients):
            self._dispatch_client(client_id=cid, dispatch_time=0.0)

        # Event-driven execution loop
        while self.event_queue and self.arrival_count < self.config.max_updates:
            event_time, _, _, event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time
            event_type = event_data["event_type"]

            if event_type == "TRAIN_COMPLETE":
                cid = event_data["client_id"]
                training_version = event_data["training_version"]
                local_loss = event_data["local_loss"]
                num_examples = event_data["num_examples"]

                self.event_log.append({
                    "time": round(float(event_time), 4),
                    "event_type": "TRAIN_COMPLETE",
                    "client_id": cid,
                    "training_version": training_version,
                    "local_loss": round(float(local_loss), 6),
                    "num_examples": num_examples,
                })

                # Sample communication delay using delay simulator
                comm_delay = float(self.delay_simulator.sample_delay(cid, self.server_version))
                arrival_time = event_time + comm_delay

                arrival_event = {
                    "event_type": "UPDATE_ARRIVAL",
                    "client_id": cid,
                    "training_version": training_version,
                    "trained_params": event_data["trained_params"],
                    "local_loss": local_loss,
                    "num_examples": num_examples,
                    "comm_delay": comm_delay,
                    "dispatch_time": event_data["dispatch_time"],
                    "train_complete_time": event_data["train_complete_time"],
                }
                self._push_event(arrival_time, 3, arrival_event)

            elif event_type == "UPDATE_ARRIVAL":
                cid = event_data["client_id"]
                client_training_version = event_data["training_version"]
                trained_params = event_data["trained_params"]
                local_loss = event_data["local_loss"]
                num_examples = event_data["num_examples"]
                comm_delay = event_data["comm_delay"]
                dispatch_time = event_data["dispatch_time"]
                train_complete_time = event_data["train_complete_time"]

                self.arrival_count += 1
                self.client_arrival_counts[cid] += 1

                # True model-version staleness: tau = current_server_version - client_training_version
                server_version_at_arrival = self.server_version
                staleness = int(server_version_at_arrival - client_training_version)
                if staleness < 0:
                    raise RuntimeError(
                        f"Negative staleness encountered: server={server_version_at_arrival}, client={client_training_version}"
                    )

                # Record arrival in arrival_log
                arrival_record = {
                    "arrival_index": self.arrival_count,
                    "client_id": cid,
                    "dispatch_time": round(float(dispatch_time), 4),
                    "train_complete_time": round(float(train_complete_time), 4),
                    "arrival_time": round(float(event_time), 4),
                    "comm_delay": round(float(comm_delay), 4),
                    "client_training_version": client_training_version,
                    "server_version_at_arrival": server_version_at_arrival,
                    "staleness": staleness,
                    "sample_count": num_examples,
                    "local_loss": round(float(local_loss), 6),
                }
                self.arrival_log.append(arrival_record)

                # Store update in server buffer
                buffered_item = {
                    "client_id": cid,
                    "trained_params": trained_params,
                    "sample_count": num_examples,
                    "staleness": staleness,
                    "local_loss": local_loss,
                    "client_training_version": client_training_version,
                    "server_version_at_arrival": server_version_at_arrival,
                    "arrival_time": event_time,
                    "arrival_index": self.arrival_count,
                }
                self.buffer.append(buffered_item)

                self.event_log.append({
                    "time": round(float(event_time), 4),
                    "event_type": "UPDATE_ARRIVAL",
                    "arrival_index": self.arrival_count,
                    "client_id": cid,
                    "client_training_version": client_training_version,
                    "server_version_at_arrival": server_version_at_arrival,
                    "staleness": staleness,
                    "comm_delay": round(float(comm_delay), 4),
                    "buffer_size_after": len(self.buffer),
                })

                # Check if buffer reached capacity K
                if len(self.buffer) >= self.config.buffer_size:
                    self._aggregate_buffer(is_final_flush=False)

                # Immediately re-dispatch this client if more arrivals are needed
                if self.arrival_count < self.config.max_updates:
                    self._dispatch_client(client_id=cid, dispatch_time=event_time)

        # Deterministic end-of-run partial buffer policy
        unflushed_buffer_count = len(self.buffer)
        final_buffer_flushed = False

        if len(self.buffer) > 0:
            if self.config.flush_final_buffer:
                final_buffer_flushed = True
                self._aggregate_buffer(is_final_flush=True)
                unflushed_buffer_count = 0
            else:
                self.event_log.append({
                    "time": round(float(self.current_time), 4),
                    "event_type": "FINAL_BUFFER_DISCARDED",
                    "unflushed_buffer_count": len(self.buffer),
                    "client_ids": [u["client_id"] for u in self.buffer],
                })

        # Final test and validation evaluation
        final_test = evaluate(self.server_model, self.test_loader, self.criterion)
        final_val = evaluate(self.server_model, self.val_loader, self.criterion)
        final_test_loss = float(final_test["loss"])
        final_test_acc = float(final_test["accuracy"])
        final_val_loss = float(final_val["loss"])
        final_val_acc = float(final_val["accuracy"])

        all_staleness = [u["staleness"] for u in self.arrival_log]
        mean_stale = float(np.mean(all_staleness)) if all_staleness else 0.0
        min_stale = int(np.min(all_staleness)) if all_staleness else 0
        max_stale = int(np.max(all_staleness)) if all_staleness else 0

        # Construct comprehensive results payload
        results: Dict[str, Any] = {
            "config": {
                "algorithm": "FedBuff",
                "n_clients": self.config.n_clients,
                "max_updates": self.config.max_updates,
                "buffer_size": self.config.buffer_size,
                "flush_final_buffer": self.config.flush_final_buffer,
                "local_epochs": self.config.local_epochs,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "partition_method": self.config.partition_method,
                "alpha": self.config.alpha,
                "seed": self.config.seed,
                "delay_distribution": self.config.delay_distribution,
                "max_delay": self.config.max_delay,
                "compute_time": self.config.compute_time,
                "staleness_seed": self.config.staleness_seed,
            },
            "client_stats": self.partition_stats,
            "initial": self.initial_metrics,
            "event_log": self.event_log,
            "arrival_log": self.arrival_log,
            "aggregation_log": self.aggregation_log,
            "history": self.history,
            "test": {
                "loss": round(final_test_loss, 6),
                "accuracy": round(final_test_acc, 6),
            },
            "validation": {
                "loss": round(final_val_loss, 6),
                "accuracy": round(final_val_acc, 6),
            },
            "summary": {
                "algorithm": "FedBuff",
                "arrival_count": self.arrival_count,
                "buffered_update_count": self.arrival_count,
                "applied_update_count": self.applied_update_count,
                "server_aggregation_count": self.aggregation_count,
                "final_server_version": self.server_version,
                "buffer_size_k": self.config.buffer_size,
                "unflushed_final_buffer_count": unflushed_buffer_count,
                "discarded_final_buffer_count": unflushed_buffer_count,
                "final_buffer_flushed": final_buffer_flushed,
                "mean_staleness": round(mean_stale, 4),
                "min_staleness": min_stale,
                "max_staleness": max_stale,
                "total_simulated_time": round(float(self.current_time), 4),
                "client_arrival_counts": {str(cid): count for cid, count in sorted(self.client_arrival_counts.items())},
                "client_applied_counts": {str(cid): count for cid, count in sorted(self.client_applied_counts.items())},
                "final_validation_accuracy": round(final_val_acc, 6),
                "final_validation_loss": round(final_val_loss, 6),
                "final_test_accuracy": round(final_test_acc, 6),
                "final_test_loss": round(final_test_loss, 6),
            },
            # Top-level convenience fields
            "arrival_count": self.arrival_count,
            "buffered_update_count": self.arrival_count,
            "applied_update_count": self.applied_update_count,
            "server_aggregation_count": self.aggregation_count,
            "final_server_version": self.server_version,
            "unflushed_buffer_count": unflushed_buffer_count,
            "final_buffer_flushed": final_buffer_flushed,
            "final_test_accuracy": round(final_test_acc, 6),
            "final_test_loss": round(final_test_loss, 6),
            "final_validation_accuracy": round(final_val_acc, 6),
            "final_validation_loss": round(final_val_loss, 6),
            "mean_staleness": round(mean_stale, 4),
            "max_staleness": max_stale,
        }

        return results


def run_fedbuff_simulation(
    fedbuff_config: FedBuffConfig,
    results_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute FedBuff simulation and optionally save results."""
    simulator = FedBuffSimulator(config=fedbuff_config)
    results = simulator.run_simulation()

    if results_path is not None:
        p = Path(results_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if csv_path is not None and results.get("arrival_log"):
        save_client_updates_csv(results["arrival_log"], csv_path)

    return results


def main() -> None:
    """CLI entry point for FedBuff simulation."""
    parser = argparse.ArgumentParser(
        description="Buffered Asynchronous Federated Learning (FedBuff) Event-Driven Simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fedbuff_config.json",
        help="Path to FedBuff configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/fedbuff_metrics.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to save arrival log CSV",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--clients", type=int, default=None, help="Number of clients")
    parser.add_argument(
        "--updates", type=int, default=None, help="Number of total client arrivals to simulate"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=None, help="Buffer capacity K before triggering aggregation"
    )
    parser.add_argument(
        "--flush-final-buffer",
        action="store_true",
        default=None,
        dest="flush_final_buffer",
        help="Flush and aggregate incomplete buffer at end of simulation",
    )
    parser.add_argument(
        "--partition",
        type=str,
        choices=["iid", "dirichlet"],
        default=None,
        help="Partition method",
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Dirichlet concentration parameter alpha"
    )
    parser.add_argument(
        "--delay-dist",
        type=str,
        choices=["uniform", "constant", "poisson", "exponential", "zero"],
        default=None,
        help="Delay distribution",
    )
    parser.add_argument("--max-delay", type=int, default=None, help="Maximum simulated delay")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of local training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for client training"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")

    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        cfg = FedBuffConfig.from_json(config_path)
    else:
        cfg = FedBuffConfig()

    if args.seed is not None:
        cfg.seed = args.seed
        cfg.staleness_seed = args.seed
    if args.clients is not None:
        cfg.n_clients = args.clients
    if args.updates is not None:
        cfg.max_updates = args.updates
    if args.buffer_size is not None:
        cfg.buffer_size = args.buffer_size
    if args.flush_final_buffer is not None:
        cfg.flush_final_buffer = args.flush_final_buffer
    if args.epochs is not None:
        cfg.local_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.partition is not None:
        cfg.partition_method = args.partition
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.delay_dist is not None:
        cfg.delay_distribution = args.delay_dist
    if args.max_delay is not None:
        cfg.max_delay = args.max_delay

    results = run_fedbuff_simulation(
        fedbuff_config=cfg,
        results_path=args.output,
        csv_path=args.csv,
    )

    print(
        f"[FedBuff] Completed {results['arrival_count']} arrivals | "
        f"{results['server_aggregation_count']} aggregations | "
        f"Server Version: {results['final_server_version']} | "
        f"Unflushed buffer: {results['unflushed_buffer_count']} | "
        f"Final Test Acc: {results['final_test_accuracy']:.4f} | "
        f"Saved to: {args.output}"
    )


if __name__ == "__main__":
    main()
