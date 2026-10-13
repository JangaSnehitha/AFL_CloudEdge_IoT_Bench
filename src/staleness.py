"""Simulated client update delay and staleness instrumentation for federated learning."""

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class StalenessConfig:
    """Configuration for client update delay and staleness simulation."""

    distribution: str = "uniform"  # "uniform", "constant", "poisson", "exponential", "zero"
    max_delay: int = 4
    staleness_threshold: Optional[int] = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.distribution not in ("uniform", "constant", "poisson", "exponential", "zero"):
            raise ValueError(f"Unsupported delay distribution: {self.distribution}")
        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")
        if self.staleness_threshold is not None and self.staleness_threshold < 0:
            raise ValueError("staleness_threshold must be >= 0")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "StalenessConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class StalenessSimulator:
    """Generates simulated update delays and evaluates staleness acceptance."""

    def __init__(self, config: StalenessConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def sample_delay(self, client_id: int, server_round: int) -> int:
        """Sample simulated update delay for a client in a given communication round.

        Returns:
            Integer delay >= 0 and <= max_delay.
        """
        if self.config.max_delay == 0 or self.config.distribution == "zero":
            return 0

        if self.config.distribution == "constant":
            return self.config.max_delay

        if self.config.distribution == "uniform":
            return int(self.rng.integers(0, self.config.max_delay + 1))

        if self.config.distribution == "poisson":
            lam = max(0.5, self.config.max_delay / 2.0)
            sample = int(self.rng.poisson(lam))
            return min(self.config.max_delay, max(0, sample))

        if self.config.distribution == "exponential":
            scale = max(0.5, self.config.max_delay / 2.0)
            sample = int(np.round(self.rng.exponential(scale)))
            return min(self.config.max_delay, max(0, sample))

        return 0

    def calculate_staleness(self, delay: int) -> int:
        """Return staleness at aggregation time corresponding to the simulated delay."""
        return max(0, int(delay))

    def is_accepted(self, staleness: int) -> bool:
        """Evaluate if an update with the given staleness is accepted under the threshold."""
        if self.config.staleness_threshold is None:
            return True
        return staleness <= self.config.staleness_threshold


def save_client_updates_csv(client_updates: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Write client updates list to a CSV file."""
    if not client_updates:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(client_updates[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(client_updates)
