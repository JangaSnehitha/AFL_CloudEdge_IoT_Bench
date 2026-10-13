"""Small CPU-friendly MLP classifier for the centralized baseline."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    input_dim: int = 20
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 4
    dropout: float = 0.0
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9

    def __post_init__(self):
        if self.input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if self.output_dim < 2:
            raise ValueError("output_dim must be >= 2")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be 'sgd' or 'adam'")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "MLPConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class MLP(nn.Module):
    """Multi-layer perceptron classifier.

    Args:
        config: MLPConfig specifying architecture and dropout rate.
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_features = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, config.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch_size, output_dim)."""
        return self.net(x)


def build_model(config: MLPConfig) -> MLP:
    """Instantiate an MLP on CPU."""
    return MLP(config)


def build_optimizer(model: MLP, config: MLPConfig) -> torch.optim.Optimizer:
    """Return an SGD or Adam optimizer for the given model and config."""
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    return torch.optim.Adam(model.parameters(), lr=config.lr)


def build_criterion() -> nn.CrossEntropyLoss:
    """Return the CrossEntropyLoss used for all classification tasks."""
    return nn.CrossEntropyLoss()
