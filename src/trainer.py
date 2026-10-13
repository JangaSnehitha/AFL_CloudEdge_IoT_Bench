"""Training and evaluation functions for the centralized MLP baseline."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

from src.dataset import SyntheticDatasetConfig, get_synthetic_dataloaders
from src.metrics import compute_accuracy, compute_mean
from src.model import MLPConfig, build_criterion, build_model, build_optimizer


@dataclass
class TrainConfig:
    """Training hyperparameters loaded from configs/train_config.json."""

    n_epochs: int = 10
    batch_size: int = 32
    seed: int = 42

    def __post_init__(self):
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def set_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """Run one training epoch.

    Returns:
        {"loss": average_batch_loss}
    """
    model.train()
    losses: List[float] = []
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return {"loss": compute_mean(losses)}


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
) -> Dict[str, float]:
    """Evaluate model on a data loader.

    Returns:
        {"loss": average_loss, "accuracy": average_top1_accuracy}
    """
    model.eval()
    losses: List[float] = []
    accs: List[float] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            losses.append(criterion(logits, y).item())
            accs.append(compute_accuracy(logits, y))
    return {"loss": compute_mean(losses), "accuracy": compute_mean(accs)}


def run(
    train_cfg: TrainConfig,
    dataset_cfg: SyntheticDatasetConfig,
    model_cfg: MLPConfig,
) -> Dict:
    """Full training loop.

    Returns a structured metrics dict with config, per-epoch history,
    and final test results.
    """
    set_seed(train_cfg.seed)

    loaders = get_synthetic_dataloaders(dataset_cfg, batch_size=train_cfg.batch_size)
    model = build_model(model_cfg)
    criterion = build_criterion()
    optimizer = build_optimizer(model, model_cfg)

    history: List[Dict] = []
    for epoch in range(1, train_cfg.n_epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer)
        val_metrics = evaluate(model, loaders["val"], criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
            }
        )

    test_metrics = evaluate(model, loaders["test"], criterion)

    return {
        "config": {
            "seed": train_cfg.seed,
            "n_epochs": train_cfg.n_epochs,
            "batch_size": train_cfg.batch_size,
            "optimizer": model_cfg.optimizer,
            "lr": model_cfg.lr,
            "hidden_dims": model_cfg.hidden_dims,
        },
        "history": history,
        "test": {
            "loss": round(test_metrics["loss"], 6),
            "accuracy": round(test_metrics["accuracy"], 6),
        },
    }


def save_results(metrics: Dict, path: Union[str, Path]) -> None:
    """Write metrics dict to a JSON file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
