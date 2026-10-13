"""Flower simulation entry point and configuration for synchronous FedAvg."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import flwr as fl
from flwr.common import ndarrays_to_parameters

from src.client import (
    create_client_fn,
    get_model_parameters,
    load_partitioned_dataloaders,
)
from src.dataset import SyntheticDatasetConfig
from src.model import MLPConfig, build_model
from src.partition import PartitionConfig
from src.strategy import FedAvgStrategy, create_evaluate_fn
from src.trainer import save_results, set_seed


@dataclass
class FedAvgConfig:
    """Hyperparameters for synchronous FedAvg federated training."""

    n_clients: int = 5
    local_epochs: int = 1
    num_rounds: int = 5
    partition_method: str = "iid"
    alpha: float = 0.5
    seed: int = 42
    batch_size: int = 32
    lr: float = 0.01
    min_samples_per_client: int = 10

    def __post_init__(self) -> None:
        if self.n_clients < 1:
            raise ValueError("n_clients must be >= 1")
        if self.local_epochs < 1:
            raise ValueError("local_epochs must be >= 1")
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if self.partition_method not in ("iid", "dirichlet"):
            raise ValueError("partition_method must be 'iid' or 'dirichlet'")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.lr <= 0.0:
            raise ValueError("lr must be > 0")
        if self.min_samples_per_client < 1:
            raise ValueError("min_samples_per_client must be >= 1")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FedAvgConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def run_fedavg_simulation(
    fedavg_config: FedAvgConfig,
    dataset_config: Optional[SyntheticDatasetConfig] = None,
    model_config: Optional[MLPConfig] = None,
    results_path: Optional[Union[str, Path]] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute synchronous FedAvg using Flower Simulation.

    Args:
        fedavg_config: Hyperparameters for federated training.
        dataset_config: Dataset generation config (optional, loaded from configs/ if None).
        model_config: Model architecture config (optional, loaded from configs/ if None).
        results_path: Optional path to save JSON metrics.
        ray_init_args: Optional arguments passed to Ray initialization.

    Returns:
        Structured dictionary containing run configuration, per-round history, and final test metrics.
    """
    set_seed(fedavg_config.seed)

    if dataset_config is None:
        ds_path = Path("configs/dataset_config.json")
        if ds_path.exists():
            dataset_config = SyntheticDatasetConfig.from_json(ds_path)
        else:
            dataset_config = SyntheticDatasetConfig()
    dataset_config.seed = fedavg_config.seed

    if model_config is None:
        model_path = Path("configs/model_config.json")
        if model_path.exists():
            model_config = MLPConfig.from_json(model_path)
        else:
            model_config = MLPConfig()
    model_config.lr = fedavg_config.lr

    partition_config = PartitionConfig(
        n_clients=fedavg_config.n_clients,
        method=fedavg_config.partition_method,
        alpha=fedavg_config.alpha,
        seed=fedavg_config.seed,
        min_samples_per_client=fedavg_config.min_samples_per_client,
    )

    client_loaders, val_loader, test_loader = load_partitioned_dataloaders(
        dataset_config=dataset_config,
        partition_config=partition_config,
        batch_size=fedavg_config.batch_size,
    )

    init_model = build_model(model_config)
    initial_params = ndarrays_to_parameters(get_model_parameters(init_model))

    history: List[Dict[str, Any]] = []
    initial_metrics: Dict[str, Any] = {}
    evaluate_fn = create_evaluate_fn(
        model_config=model_config,
        val_loader=val_loader,
        test_loader=test_loader,
        history=history,
        initial_metrics=initial_metrics,
    )

    strategy = FedAvgStrategy(
        initial_parameters=initial_params,
        evaluate_fn=evaluate_fn,
        min_fit_clients=fedavg_config.n_clients,
        min_available_clients=fedavg_config.n_clients,
        fraction_fit=1.0,
    )

    client_fn = create_client_fn(
        client_loaders=client_loaders,
        model_config=model_config,
        local_epochs=fedavg_config.local_epochs,
        lr=fedavg_config.lr,
    )

    if ray_init_args is None:
        ray_init_args = {"include_dashboard": False, "ignore_reinit_error": True}

    server_config = fl.server.ServerConfig(num_rounds=fedavg_config.num_rounds)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=fedavg_config.n_clients,
        config=server_config,
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args=ray_init_args,
    )

    final_test = history[-1] if history else {}
    results: Dict[str, Any] = {
        "config": {
            "algorithm": "FedAvg",
            "n_clients": fedavg_config.n_clients,
            "local_epochs": fedavg_config.local_epochs,
            "num_rounds": fedavg_config.num_rounds,
            "partition_method": fedavg_config.partition_method,
            "alpha": fedavg_config.alpha,
            "seed": fedavg_config.seed,
            "batch_size": fedavg_config.batch_size,
            "lr": fedavg_config.lr,
        },
        "initial": initial_metrics,
        "history": history,
        "test": {
            "loss": final_test.get("test_loss"),
            "accuracy": final_test.get("test_accuracy"),
        },
    }

    if results_path is not None:
        save_results(results, results_path)

    return results


def main() -> None:
    """CLI entry point for running FedAvg experiments."""
    parser = argparse.ArgumentParser(description="Synchronous FedAvg Simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fedavg_config.json",
        help="Path to FedAvg configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/fedavg_metrics.json",
        help="Path to save results JSON",
    )
    parser.add_argument("--clients", type=int, default=None, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=None, help="Communication rounds")
    parser.add_argument("--epochs", type=int, default=None, help="Local epochs per round")
    parser.add_argument(
        "--partition",
        type=str,
        choices=["iid", "dirichlet"],
        default=None,
        help="Partitioning strategy",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Dirichlet concentration alpha")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = FedAvgConfig.from_json(cfg_path)
    else:
        cfg = FedAvgConfig()

    if args.clients is not None:
        cfg.n_clients = args.clients
    if args.rounds is not None:
        cfg.num_rounds = args.rounds
    if args.epochs is not None:
        cfg.local_epochs = args.epochs
    if args.partition is not None:
        cfg.partition_method = args.partition
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.seed is not None:
        cfg.seed = args.seed
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    results = run_fedavg_simulation(fedavg_config=cfg, results_path=args.output)
    print("FedAvg simulation completed.")
    if results.get("test"):
        print(f"Final Test Accuracy: {results['test']['accuracy']}")
        print(f"Final Test Loss:     {results['test']['loss']}")


if __name__ == "__main__":
    main()
