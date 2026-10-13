"""Flower client implementation and data loading for federated learning."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from flwr.app import Context
from flwr.client import Client, NumPyClient

from src.dataset import (
    SyntheticDatasetConfig,
    generate_synthetic_classification_data,
    split_data,
)
from src.model import MLPConfig, build_criterion, build_model, build_optimizer
from src.partition import PartitionConfig, compute_partition_stats, partition_data
from src.trainer import evaluate, train_one_epoch


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.detach().cpu().numpy().copy() for val in model.state_dict().values()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Load NumPy parameters into a PyTorch model."""
    state_dict = {
        key: torch.from_numpy(val.copy())
        for key, val in zip(model.state_dict().keys(), parameters)
    }
    model.load_state_dict(state_dict, strict=True)


class FlowerClient(NumPyClient):
    """Flower client for local model training and parameter exchange."""

    def __init__(
        self,
        cid: str,
        train_loader: DataLoader,
        model_config: MLPConfig,
        local_epochs: int = 1,
        lr: Optional[float] = None,
    ) -> None:
        self.cid = str(cid)
        self.train_loader = train_loader
        self.model_config = model_config
        self.local_epochs = local_epochs
        self.lr = lr if lr is not None else model_config.lr

        # Client-side model copy
        self.model = build_model(model_config)
        self.criterion = build_criterion()

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """Return current local model parameters."""
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update local model parameters from incoming array list."""
        set_model_parameters(self.model, parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train locally using received global parameters."""
        self.set_parameters(parameters)

        epochs = int(config.get("local_epochs", self.local_epochs))
        client_lr = float(config.get("lr", self.lr))

        # Re-build optimizer with current client parameters and learning rate
        opt_config = MLPConfig(
            input_dim=self.model_config.input_dim,
            hidden_dims=self.model_config.hidden_dims,
            output_dim=self.model_config.output_dim,
            dropout=self.model_config.dropout,
            optimizer=self.model_config.optimizer,
            lr=client_lr,
            momentum=self.model_config.momentum,
        )
        optimizer = build_optimizer(self.model, opt_config)

        last_loss = 0.0
        for _ in range(epochs):
            metrics = train_one_epoch(
                self.model, self.train_loader, self.criterion, optimizer
            )
            last_loss = metrics.get("loss", 0.0)

        num_examples = len(self.train_loader.dataset)
        return (
            self.get_parameters(),
            num_examples,
            {"cid": self.cid, "loss": float(last_loss)},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate local model on local training dataset."""
        self.set_parameters(parameters)
        metrics = evaluate(self.model, self.train_loader, self.criterion)
        num_examples = len(self.train_loader.dataset)
        return (
            float(metrics["loss"]),
            num_examples,
            {"accuracy": float(metrics["accuracy"])},
        )


def load_partitioned_dataloaders(
    dataset_config: SyntheticDatasetConfig,
    partition_config: PartitionConfig,
    batch_size: int = 32,
    return_stats: bool = False,
) -> Union[
    Tuple[List[DataLoader], DataLoader, DataLoader],
    Tuple[List[DataLoader], DataLoader, DataLoader, List[Dict[str, Any]]],
]:
    """Generate synthetic dataset, normalize, partition training split, and return DataLoaders.

    Returns:
        (client_loaders, val_loader, test_loader) if return_stats=False,
        else (client_loaders, val_loader, test_loader, partition_stats)
    """
    x, y = generate_synthetic_classification_data(dataset_config)
    splits = split_data(x, y, dataset_config)

    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]

    # Compute normalization statistics strictly from the training split
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)

    x_train_norm = ((x_train - mean) / std).astype(np.float32)
    x_val_norm = ((x_val - mean) / std).astype(np.float32)
    x_test_norm = ((x_test - mean) / std).astype(np.float32)

    # Partition the normalized training split
    partitions = partition_data(x_train_norm, y_train, partition_config)
    stats = compute_partition_stats(partitions, dataset_config.n_classes)

    client_loaders: List[DataLoader] = []
    for x_c, y_c in partitions:
        ds_c = TensorDataset(torch.from_numpy(x_c), torch.from_numpy(y_c).long())
        client_loaders.append(DataLoader(ds_c, batch_size=batch_size, shuffle=True))

    val_dataset = TensorDataset(
        torch.from_numpy(x_val_norm), torch.from_numpy(y_val).long()
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(
        torch.from_numpy(x_test_norm), torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if return_stats:
        return client_loaders, val_loader, test_loader, stats
    return client_loaders, val_loader, test_loader


def create_client_fn(
    client_loaders: List[DataLoader],
    model_config: MLPConfig,
    local_epochs: int = 1,
    lr: Optional[float] = None,
):
    """Create a client_fn compatible with Flower Simulation."""

    def client_fn(context: Union[Context, str, int]) -> Client:
        if isinstance(context, (str, int)):
            cid = str(context)
        elif hasattr(context, "node_config"):
            cid = str(context.node_config.get("partition-id", context.node_id))
        else:
            cid = str(context)

        cid_int = int(cid)
        loader = client_loaders[cid_int]
        client = FlowerClient(
            cid=cid,
            train_loader=loader,
            model_config=model_config,
            local_epochs=local_epochs,
            lr=lr,
        )
        return client.to_client()

    return client_fn
