"""Adaptive Staleness-and-Heterogeneity-Aware Asynchronous Federated Learning (Adaptive FedAsync)."""

import argparse
import copy
from dataclasses import asdict, dataclass
import heapq
import json
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
class AdaptiveFedAsyncConfig:
    """Configuration for Adaptive Staleness-and-Heterogeneity-Aware FedAsync."""

    n_clients: int = 10
    max_updates: int = 30
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    partition_method: str = "dirichlet"
    alpha: float = 0.5
    seed: int = 42
    alpha_0: float = 1.0
    lambda_divergence: float = 0.5
    lambda_reliability: float = 0.5
    min_weight: float = 0.01
    delay_distribution: str = "uniform"  # "uniform", "constant", "poisson", "exponential", "zero"
    max_delay: int = 3
    compute_time: float = 1.0  # Simulated time required per local epoch
    staleness_seed: Optional[int] = None
    min_samples_per_client: int = 5

    def __post_init__(self) -> None:
        if self.n_clients < 1:
            raise ValueError("n_clients must be >= 1")
        if self.max_updates < 1:
            raise ValueError("max_updates must be >= 1")
        if self.local_epochs < 1:
            raise ValueError("local_epochs must be >= 1")
        if self.alpha_0 <= 0.0:
            raise ValueError("alpha_0 must be > 0.0")
        if self.lambda_divergence < 0.0:
            raise ValueError("lambda_divergence must be >= 0.0")
        if self.lambda_reliability < 0.0:
            raise ValueError("lambda_reliability must be >= 0.0")
        if not (0.0 < self.min_weight <= 1.0):
            raise ValueError("min_weight must satisfy 0 < min_weight <= 1.0")
        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")
        if self.compute_time < 0.0:
            raise ValueError("compute_time must be >= 0.0")
        if self.staleness_seed is None:
            self.staleness_seed = self.seed

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "AdaptiveFedAsyncConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def compute_total_variation_distance(
    p_k: Union[List[float], np.ndarray],
    p_global: Union[List[float], np.ndarray],
) -> float:
    """Compute total variation distance between client distribution and global distribution:

    d_k = 0.5 * sum_c |p_k(c) - p_global(c)|
    """
    p_k_arr = np.asarray(p_k, dtype=np.float64)
    p_glob_arr = np.asarray(p_global, dtype=np.float64)

    sum_pk = np.sum(p_k_arr)
    sum_glob = np.sum(p_glob_arr)

    if sum_pk > 0:
        p_k_norm = p_k_arr / sum_pk
    else:
        p_k_norm = np.zeros_like(p_k_arr)

    if sum_glob > 0:
        p_glob_norm = p_glob_arr / sum_glob
    else:
        p_glob_norm = np.zeros_like(p_glob_arr)

    tv_dist = 0.5 * float(np.sum(np.abs(p_k_norm - p_glob_norm)))
    return float(np.clip(tv_dist, 0.0, 1.0))


def compute_base_weight(staleness: int, alpha_0: float = 1.0) -> float:
    """Calculate base FedAsync staleness weight: alpha_0 / (1 + staleness)."""
    if staleness < 0:
        raise ValueError(f"Staleness cannot be negative, got {staleness}")
    if alpha_0 <= 0.0:
        raise ValueError(f"alpha_0 must be positive, got {alpha_0}")
    return float(alpha_0 / (1.0 + float(staleness)))


def compute_adaptive_weight(
    base_weight: float,
    heterogeneity_score: float,
    reliability_score: float,
    lambda_divergence: float = 0.5,
    lambda_reliability: float = 0.5,
    min_weight: float = 0.01,
) -> Tuple[float, float]:
    """Calculate raw and clipped adaptive aggregation weights:

    raw_weight = base_weight / (1 + lambda_div * d_k + lambda_rel * r_k)
    applied_weight = clip(raw_weight, min_weight, 1.0)

    Returns:
        (raw_weight, applied_weight)
    """
    if np.isnan(base_weight) or np.isinf(base_weight):
        raise ValueError("base_weight must be finite")
    if np.isnan(heterogeneity_score) or np.isinf(heterogeneity_score) or heterogeneity_score < 0.0:
        raise ValueError("heterogeneity_score must be finite and non-negative")
    if np.isnan(reliability_score) or np.isinf(reliability_score) or not (0.0 <= reliability_score <= 1.0):
        raise ValueError("reliability_score must be finite and in [0, 1]")

    denominator = 1.0 + float(lambda_divergence * heterogeneity_score) + float(lambda_reliability * reliability_score)
    if denominator <= 0.0:
        raise ValueError(f"Invalid denominator in adaptive weight computation: {denominator}")

    raw_weight = float(base_weight / denominator)
    applied_weight = float(np.clip(raw_weight, min_weight, 1.0))
    return raw_weight, applied_weight


def interpolate_parameters(
    server_params: List[np.ndarray],
    client_params: List[np.ndarray],
    weight: float,
) -> List[np.ndarray]:
    """Update server parameters by interpolation: (1 - weight) * server + weight * client."""
    if len(server_params) != len(client_params):
        raise ValueError("Parameter length mismatch")
    return [(1.0 - weight) * s_w + weight * c_w for s_w, c_w in zip(server_params, client_params)]


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


class AdaptiveFedAsyncSimulator:
    """Event-driven simulator for Adaptive Staleness-and-Heterogeneity-Aware FedAsync."""

    def __init__(
        self,
        config: AdaptiveFedAsyncConfig,
        dataset_config: Optional[SyntheticDatasetConfig] = None,
        model_config: Optional[MLPConfig] = None,
    ) -> None:
        self.config = config

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

        # Compute global training class distribution
        total_class_counts = np.zeros(self.dataset_config.n_classes, dtype=np.float64)
        for stat in self.partition_stats:
            total_class_counts += np.array(stat["class_counts"], dtype=np.float64)
        self.global_class_fractions = (
            total_class_counts / np.sum(total_class_counts)
        ).tolist()

        # Compute deterministic heterogeneity score (TV distance) for each client
        self.client_heterogeneity_scores: Dict[int, float] = {}
        for stat in self.partition_stats:
            cid = stat["client_id"]
            d_k = compute_total_variation_distance(
                stat["class_fractions"], self.global_class_fractions
            )
            self.client_heterogeneity_scores[cid] = round(float(d_k), 6)

        # Initialize delay simulator and historical delay tracking
        staleness_cfg = StalenessConfig(
            distribution=config.delay_distribution,
            max_delay=config.max_delay,
            seed=config.staleness_seed if config.staleness_seed is not None else config.seed,
        )
        self.delay_simulator = StalenessSimulator(staleness_cfg)
        self.client_delay_history: Dict[int, List[float]] = {
            cid: [] for cid in range(config.n_clients)
        }

        # Model and criterion
        self.server_model = build_model(self.model_config)
        self.server_params = get_model_parameters(self.server_model)
        self.criterion = build_criterion()

        # State tracking
        self.server_version: int = 0
        self.current_time: float = 0.0
        self.updates_applied: int = 0
        self.event_counter: int = 0

        self.event_queue: List[Tuple[float, int, int, Dict[str, Any]]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.update_log: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.initial_metrics: Dict[str, Any] = {}

    def _push_event(self, timestamp: float, priority: int, event_data: Dict[str, Any]) -> None:
        self.event_counter += 1
        heapq.heappush(self.event_queue, (timestamp, priority, self.event_counter, event_data))

    def _dispatch_client(self, client_id: int, dispatch_time: float) -> None:
        training_version = self.server_version
        training_params = copy.deepcopy(self.server_params)

        self.event_log.append({
            "time": round(float(dispatch_time), 4),
            "event_type": "DISPATCH",
            "client_id": client_id,
            "server_version": training_version,
        })

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

    def run_simulation(self) -> Dict[str, Any]:
        """Execute the adaptive asynchronous simulation until max_updates are applied."""
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

        # Initially dispatch all clients at simulated time 0.0
        for cid in range(self.config.n_clients):
            self._dispatch_client(client_id=cid, dispatch_time=0.0)

        while self.event_queue and self.updates_applied < self.config.max_updates:
            event_time, _, _, event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time
            event_type = event_data["event_type"]

            if event_type == "TRAIN_COMPLETE":
                cid = event_data["client_id"]
                training_version = event_data["training_version"]
                local_loss = event_data["local_loss"]
                num_examples = event_data["num_examples"]
                dispatch_time = event_data["dispatch_time"]
                train_complete_time = event_data["train_complete_time"]

                self.event_log.append({
                    "time": round(float(event_time), 4),
                    "event_type": "TRAIN_COMPLETE",
                    "client_id": cid,
                    "training_version": training_version,
                    "local_loss": round(float(local_loss), 6),
                    "num_examples": num_examples,
                })

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
                    "dispatch_time": dispatch_time,
                    "train_complete_time": train_complete_time,
                    "arrival_time": arrival_time,
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
                arrival_time = event_data["arrival_time"]

                self.event_log.append({
                    "time": round(float(event_time), 4),
                    "event_type": "UPDATE_ARRIVAL",
                    "client_id": cid,
                    "training_version": client_training_version,
                    "comm_delay": comm_delay,
                })

                # --- Adaptive Weighting Calculation ---
                server_version_before = self.server_version
                staleness = int(server_version_before - client_training_version)
                if staleness < 0:
                    raise RuntimeError(
                        f"Negative staleness: server={server_version_before}, client={client_training_version}"
                    )

                # 1. Base staleness weight
                base_weight = compute_base_weight(staleness, self.config.alpha_0)

                # 2. Heterogeneity score (deterministic TV distance)
                heterogeneity_score = self.client_heterogeneity_scores[cid]

                # 3. Reliability score derived strictly from prior history before this update
                prior_delays = self.client_delay_history[cid]
                if not prior_delays:
                    reliability_score = 0.0
                else:
                    mean_prior_delay = float(np.mean(prior_delays))
                    reliability_score = (
                        mean_prior_delay / float(self.config.max_delay)
                        if self.config.max_delay > 0
                        else 0.0
                    )
                    reliability_score = float(np.clip(reliability_score, 0.0, 1.0))

                # 4. Adaptive weight
                raw_weight, applied_weight = compute_adaptive_weight(
                    base_weight=base_weight,
                    heterogeneity_score=heterogeneity_score,
                    reliability_score=reliability_score,
                    lambda_divergence=self.config.lambda_divergence,
                    lambda_reliability=self.config.lambda_reliability,
                    min_weight=self.config.min_weight,
                )

                # Apply update to server model
                self.server_params = interpolate_parameters(
                    self.server_params, trained_params, applied_weight
                )
                set_model_parameters(self.server_model, self.server_params)

                self.server_version += 1
                self.updates_applied += 1

                # Update client's historical delay record AFTER aggregation
                self.client_delay_history[cid].append(float(comm_delay))

                # Evaluate updated server model
                val_metrics = evaluate(self.server_model, self.val_loader, self.criterion)
                test_metrics = evaluate(self.server_model, self.test_loader, self.criterion)

                val_loss = float(val_metrics["loss"])
                val_acc = float(val_metrics["accuracy"])
                test_loss = float(test_metrics["loss"])
                test_acc = float(test_metrics["accuracy"])

                # Log AGGREGATE event
                self.event_log.append({
                    "time": round(float(event_time), 4),
                    "event_type": "AGGREGATE",
                    "client_id": cid,
                    "server_version_before": server_version_before,
                    "client_training_version": client_training_version,
                    "staleness": staleness,
                    "base_weight": round(float(base_weight), 6),
                    "heterogeneity_score": round(float(heterogeneity_score), 6),
                    "reliability_score": round(float(reliability_score), 6),
                    "raw_weight": round(float(raw_weight), 6),
                    "applied_weight": round(float(applied_weight), 6),
                    "server_version_after": self.server_version,
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_acc, 6),
                    "test_loss": round(test_loss, 6),
                    "test_accuracy": round(test_acc, 6),
                })

                # Comprehensive update record matching Section B
                update_record = {
                    "client_id": cid,
                    "dispatch_time": round(float(dispatch_time), 4),
                    "train_complete_time": round(float(train_complete_time), 4),
                    "arrival_time": round(float(arrival_time), 4),
                    "delay": round(float(comm_delay), 4),
                    "client_training_version": client_training_version,
                    "server_version_before": server_version_before,
                    "server_version_after": self.server_version,
                    "staleness": staleness,
                    "sample_count": num_examples,
                    "local_loss": round(float(local_loss), 6),
                    "base_weight": round(float(base_weight), 6),
                    "heterogeneity_score": round(float(heterogeneity_score), 6),
                    "reliability_score_before_update": round(float(reliability_score), 6),
                    "raw_weight": round(float(raw_weight), 6),
                    "applied_weight": round(float(applied_weight), 6),
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_acc, 6),
                    "test_loss": round(test_loss, 6),
                    "test_accuracy": round(test_acc, 6),
                }
                self.update_log.append(update_record)

                self.history.append({
                    "update_index": self.updates_applied,
                    "server_version": self.server_version,
                    "time": round(float(event_time), 4),
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_acc, 6),
                    "test_loss": round(test_loss, 6),
                    "test_accuracy": round(test_acc, 6),
                })

                if self.updates_applied < self.config.max_updates:
                    self._dispatch_client(client_id=cid, dispatch_time=event_time)

        final_test = evaluate(self.server_model, self.test_loader, self.criterion)
        final_test_loss = float(final_test["loss"])
        final_test_acc = float(final_test["accuracy"])

        staleness_vals = [u["staleness"] for u in self.update_log]
        delay_vals = [u["delay"] for u in self.update_log]
        het_vals = [u["heterogeneity_score"] for u in self.update_log]
        rel_vals = [u["reliability_score_before_update"] for u in self.update_log]
        base_w_vals = [u["base_weight"] for u in self.update_log]
        raw_w_vals = [u["raw_weight"] for u in self.update_log]
        applied_w_vals = [u["applied_weight"] for u in self.update_log]

        # Contribution count per client
        client_counts: Dict[str, int] = {str(c): 0 for c in range(self.config.n_clients)}
        for u in self.update_log:
            client_counts[str(u["client_id"])] += 1

        last_update = self.update_log[-1] if self.update_log else {}

        results: Dict[str, Any] = {
            "config": {
                "algorithm": "AdaptiveFedAsync",
                "n_clients": self.config.n_clients,
                "max_updates": self.config.max_updates,
                "local_epochs": self.config.local_epochs,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "partition_method": self.config.partition_method,
                "alpha": self.config.alpha,
                "seed": self.config.seed,
                "alpha_0": self.config.alpha_0,
                "lambda_divergence": self.config.lambda_divergence,
                "lambda_reliability": self.config.lambda_reliability,
                "min_weight": self.config.min_weight,
                "delay_distribution": self.config.delay_distribution,
                "max_delay": self.config.max_delay,
                "compute_time": self.config.compute_time,
                "staleness_seed": self.config.staleness_seed,
            },
            "global_training_distribution": {
                "class_fractions": [round(f, 6) for f in self.global_class_fractions]
            },
            "client_distributions": [
                {
                    "client_id": stat["client_id"],
                    "n_samples": stat["n_samples"],
                    "class_counts": stat["class_counts"],
                    "class_fractions": [round(f, 6) for f in stat["class_fractions"]],
                    "heterogeneity_score": self.client_heterogeneity_scores[stat["client_id"]],
                }
                for stat in self.partition_stats
            ],
            "initial": self.initial_metrics,
            "event_log": self.event_log,
            "update_log": self.update_log,
            "history": self.history,
            "test": {
                "loss": round(final_test_loss, 6),
                "accuracy": round(final_test_acc, 6),
            },
            "summary": {
                "total_updates": self.updates_applied,
                "final_server_version": self.server_version,
                "total_simulated_time": round(float(self.current_time), 4),
                "staleness": {
                    "min": int(np.min(staleness_vals)) if staleness_vals else 0,
                    "max": int(np.max(staleness_vals)) if staleness_vals else 0,
                    "mean": round(float(np.mean(staleness_vals)), 4) if staleness_vals else 0.0,
                },
                "delay": {
                    "min": float(np.min(delay_vals)) if delay_vals else 0.0,
                    "max": float(np.max(delay_vals)) if delay_vals else 0.0,
                    "mean": round(float(np.mean(delay_vals)), 4) if delay_vals else 0.0,
                },
                "heterogeneity_score": {
                    "min": float(np.min(het_vals)) if het_vals else 0.0,
                    "max": float(np.max(het_vals)) if het_vals else 0.0,
                    "mean": round(float(np.mean(het_vals)), 4) if het_vals else 0.0,
                },
                "reliability_score": {
                    "min": float(np.min(rel_vals)) if rel_vals else 0.0,
                    "max": float(np.max(rel_vals)) if rel_vals else 0.0,
                    "mean": round(float(np.mean(rel_vals)), 4) if rel_vals else 0.0,
                },
                "base_weight": {
                    "min": round(float(np.min(base_w_vals)), 6) if base_w_vals else 0.0,
                    "max": round(float(np.max(base_w_vals)), 6) if base_w_vals else 0.0,
                    "mean": round(float(np.mean(base_w_vals)), 6) if base_w_vals else 0.0,
                },
                "raw_weight": {
                    "min": round(float(np.min(raw_w_vals)), 6) if raw_w_vals else 0.0,
                    "max": round(float(np.max(raw_w_vals)), 6) if raw_w_vals else 0.0,
                    "mean": round(float(np.mean(raw_w_vals)), 6) if raw_w_vals else 0.0,
                },
                "applied_weight": {
                    "min": round(float(np.min(applied_w_vals)), 6) if applied_w_vals else 0.0,
                    "max": round(float(np.max(applied_w_vals)), 6) if applied_w_vals else 0.0,
                    "mean": round(float(np.mean(applied_w_vals)), 6) if applied_w_vals else 0.0,
                },
                "client_contribution_counts": client_counts,
                "final_validation_accuracy": last_update.get("val_accuracy"),
                "final_validation_loss": last_update.get("val_loss"),
                "final_test_accuracy": round(final_test_acc, 6),
                "final_test_loss": round(final_test_loss, 6),
            },
            "final_test_accuracy": round(final_test_acc, 6),
            "final_test_loss": round(final_test_loss, 6),
            "mean_staleness": round(float(np.mean(staleness_vals)), 4) if staleness_vals else 0.0,
            "total_updates": self.updates_applied,
            "final_server_version": self.server_version,
        }

        return results


def run_adaptive_fedasync_simulation(
    config: AdaptiveFedAsyncConfig,
    results_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute Adaptive FedAsync simulation and save results."""
    simulator = AdaptiveFedAsyncSimulator(config=config)
    results = simulator.run_simulation()

    if results_path is not None:
        p = Path(results_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if csv_path is not None and results.get("update_log"):
        save_client_updates_csv(results["update_log"], csv_path)

    return results


def main() -> None:
    """CLI entry point for Adaptive FedAsync simulation."""
    parser = argparse.ArgumentParser(
        description="Adaptive Staleness-and-Heterogeneity-Aware Asynchronous Federated Learning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/adaptive_fedasync_config.json",
        help="Path to Adaptive FedAsync config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/adaptive_fedasync_smoke_seed_42.json",
        help="Path to save results JSON",
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to save update log CSV")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--clients", type=int, default=None, help="Number of clients")
    parser.add_argument("--updates", type=int, default=None, help="Number of applied updates")
    parser.add_argument(
        "--partition",
        type=str,
        choices=["iid", "dirichlet"],
        default=None,
        help="Partition method",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Dirichlet concentration alpha")
    parser.add_argument("--alpha-0", type=float, default=None, dest="alpha_0", help="Base alpha_0")
    parser.add_argument(
        "--lambda-divergence",
        type=float,
        default=None,
        dest="lambda_divergence",
        help="Heterogeneity penalty lambda_divergence",
    )
    parser.add_argument(
        "--lambda-reliability",
        type=float,
        default=None,
        dest="lambda_reliability",
        help="Reliability penalty lambda_reliability",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=None,
        dest="min_weight",
        help="Minimum weight floor min_weight",
    )
    parser.add_argument(
        "--delay-dist",
        type=str,
        choices=["uniform", "constant", "poisson", "exponential", "zero"],
        default=None,
        help="Delay distribution",
    )
    parser.add_argument("--max-delay", type=int, default=None, help="Maximum delay")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = AdaptiveFedAsyncConfig.from_json(cfg_path)
    else:
        cfg = AdaptiveFedAsyncConfig()

    if args.seed is not None:
        cfg.seed = args.seed
        cfg.staleness_seed = args.seed
    if args.clients is not None:
        cfg.n_clients = args.clients
    if args.updates is not None:
        cfg.max_updates = args.updates
    if args.partition is not None:
        cfg.partition_method = args.partition
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.alpha_0 is not None:
        cfg.alpha_0 = args.alpha_0
    if args.lambda_divergence is not None:
        cfg.lambda_divergence = args.lambda_divergence
    if args.lambda_reliability is not None:
        cfg.lambda_reliability = args.lambda_reliability
    if args.min_weight is not None:
        cfg.min_weight = args.min_weight
    if args.delay_dist is not None:
        cfg.delay_distribution = args.delay_dist
    if args.max_delay is not None:
        cfg.max_delay = args.max_delay

    start_time = time.time()
    results = run_adaptive_fedasync_simulation(
        config=cfg,
        results_path=args.output,
        csv_path=args.csv,
    )
    duration = time.time() - start_time

    summary = results["summary"]
    print("=" * 70)
    print("Adaptive Staleness-and-Heterogeneity-Aware FedAsync Simulation Complete")
    print(f"Applied updates:           {summary['total_updates']}")
    print(f"Final server version:      {summary['final_server_version']}")
    print(f"Total simulated time:      {summary['total_simulated_time']} time units")
    print(f"Staleness range/mean:      [{summary['staleness']['min']}, {summary['staleness']['max']}] / {summary['staleness']['mean']:.4f}")
    print(f"Delay range/mean:          [{summary['delay']['min']}, {summary['delay']['max']}] / {summary['delay']['mean']:.4f}")
    print(f"Heterogeneity score mean:  {summary['heterogeneity_score']['mean']:.4f} (range [{summary['heterogeneity_score']['min']:.4f}, {summary['heterogeneity_score']['max']:.4f}])")
    print(f"Reliability score mean:    {summary['reliability_score']['mean']:.4f} (range [{summary['reliability_score']['min']:.4f}, {summary['reliability_score']['max']:.4f}])")
    print(f"Base weight mean:          {summary['base_weight']['mean']:.4f}")
    print(f"Raw weight mean:           {summary['raw_weight']['mean']:.4f}")
    print(f"Applied weight mean:       {summary['applied_weight']['mean']:.4f} (range [{summary['applied_weight']['min']:.4f}, {summary['applied_weight']['max']:.4f}])")
    print(f"Final validation accuracy: {summary['final_validation_accuracy']:.4f}")
    print(f"Final test accuracy:       {summary['final_test_accuracy']:.4f}")
    print(f"Final test loss:           {summary['final_test_loss']:.4f}")
    print(f"Wall-clock runtime:        {duration:.2f}s")
    print(f"Results saved to:          {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
