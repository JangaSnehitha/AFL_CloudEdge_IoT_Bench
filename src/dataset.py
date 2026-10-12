"""Reproducible synthetic classification dataset using NumPy and PyTorch."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SyntheticDatasetConfig:
    seed: int = 42
    n_samples: int = 1200
    n_features: int = 20
    n_classes: int = 4
    class_sep: float = 1.5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self):
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(ratio_sum, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if self.n_features < 2:
            raise ValueError("n_features must be at least 2")
        if self.n_samples < self.n_classes:
            raise ValueError("n_samples must be at least equal to n_classes")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SyntheticDatasetConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def generate_synthetic_classification_data(
    config: SyntheticDatasetConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data using multi-cluster Gaussian blobs.

    Args:
        config: Configuration containing sample size, dimensions, classes, and seed.

    Returns:
        X: Feature array of shape (n_samples, n_features) with dtype float32.
        y: Target label array of shape (n_samples,) with dtype int64.
    """
    rng = np.random.default_rng(config.seed)

    # Base class centers separated in feature space
    centers = rng.normal(
        loc=0.0,
        scale=config.class_sep,
        size=(config.n_classes, config.n_features),
    )

    # Distribute samples evenly across classes
    samples_per_class = [config.n_samples // config.n_classes] * config.n_classes
    remainder = config.n_samples % config.n_classes
    for i in range(remainder):
        samples_per_class[i] += 1

    x_list = []
    y_list = []

    for class_idx, n_c in enumerate(samples_per_class):
        # Generate samples around the class center
        cluster_samples = rng.normal(
            loc=centers[class_idx],
            scale=1.0,
            size=(n_c, config.n_features),
        )
        x_list.append(cluster_samples)
        y_list.append(np.full(n_c, class_idx, dtype=np.int64))

    x_all = np.vstack(x_list).astype(np.float32)
    y_all = np.concatenate(y_list).astype(np.int64)

    # Reproducibly shuffle the generated data
    shuffle_indices = rng.permutation(config.n_samples)
    x_shuffled = x_all[shuffle_indices]
    y_shuffled = y_all[shuffle_indices]

    return x_shuffled, y_shuffled


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    config: SyntheticDatasetConfig,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split data into train, validation, and test subsets."""
    n_samples = len(x)
    n_train = int(n_samples * config.train_ratio)
    n_val = int(n_samples * config.val_ratio)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    x_test, y_test = x[n_train + n_val :], y[n_train + n_val :]

    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }


def get_synthetic_datasets(
    config: Optional[SyntheticDatasetConfig] = None,
) -> Dict[str, TensorDataset]:
    """
    Generate synthetic data and return PyTorch TensorDatasets for each split.

    Features are normalized using training set statistics.
    """
    if config is None:
        config = SyntheticDatasetConfig()

    x, y = generate_synthetic_classification_data(config)
    splits = split_data(x, y, config)

    # Standardize features using train split statistics to avoid data leakage
    x_train = splits["train"][0]
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)

    datasets = {}
    for split_name, (x_split, y_split) in splits.items():
        x_norm = ((x_split - mean) / std).astype(np.float32)
        tensor_x = torch.from_numpy(x_norm)
        tensor_y = torch.from_numpy(y_split).long()
        datasets[split_name] = TensorDataset(tensor_x, tensor_y)

    return datasets


def get_synthetic_dataloaders(
    config: Optional[SyntheticDatasetConfig] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train, val, and test splits."""
    datasets = get_synthetic_datasets(config)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
