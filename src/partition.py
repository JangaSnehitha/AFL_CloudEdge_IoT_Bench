"""Federated client data partitioning using IID and non-IID (Dirichlet) strategies.

Only the training split is partitioned.  Validation and test sets remain global.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PartitionConfig:
    """Partitioning hyperparameters.

    Attributes:
        n_clients:              Number of federated clients.
        method:                 "iid" or "dirichlet".
        alpha:                  Dirichlet concentration parameter.  Smaller
                                values produce more skewed class distributions.
                                Ignored for IID partitioning.
        seed:                   RNG seed for reproducibility.
        min_samples_per_client: Every client must receive at least this many
                                training samples.
    """

    n_clients: int = 5
    method: str = "iid"
    alpha: float = 0.5
    seed: int = 42
    min_samples_per_client: int = 1

    def __post_init__(self) -> None:
        if self.n_clients < 1:
            raise ValueError("n_clients must be >= 1")
        if self.method not in ("iid", "dirichlet"):
            raise ValueError("method must be 'iid' or 'dirichlet'")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if self.min_samples_per_client < 1:
            raise ValueError("min_samples_per_client must be >= 1")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PartitionConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_feasibility(n_total: int, config: PartitionConfig) -> None:
    """Raise ValueError when the min-samples constraint cannot be satisfied."""
    required = config.n_clients * config.min_samples_per_client
    if n_total < required:
        raise ValueError(
            f"Cannot satisfy min_samples_per_client={config.min_samples_per_client} "
            f"with n_clients={config.n_clients}: need at least {required} samples "
            f"but only {n_total} are available."
        )


def _enforce_min_samples(
    client_lists: List[List[int]],
    min_samples: int,
) -> List[np.ndarray]:
    """Redistribute indices so every client has >= min_samples.

    Samples are stolen from the currently richest client and appended to the
    poorest client.  Assumes feasibility has already been checked.

    Args:
        client_lists: Per-client lists of integer sample indices.
        min_samples:  Minimum number of samples each client must hold.

    Returns:
        Sorted numpy index arrays, one per client.
    """
    n_clients = len(client_lists)
    # Work with mutable Python lists for efficient appending
    while True:
        sizes = [len(cl) for cl in client_lists]
        poor = int(np.argmin(sizes))
        if sizes[poor] >= min_samples:
            break
        rich = int(np.argmax(sizes))
        need = min_samples - sizes[poor]
        # Move the last `need` indices from the richest client
        stolen = client_lists[rich][-need:]
        client_lists[rich] = client_lists[rich][:-need]
        client_lists[poor].extend(stolen)

    return [np.sort(np.array(cl, dtype=np.intp)) for cl in client_lists]


def _iid_indices(y: np.ndarray, config: PartitionConfig) -> List[np.ndarray]:
    """Shuffle all indices and split into n_clients equal-ish chunks."""
    rng = np.random.default_rng(config.seed)
    all_idx = np.arange(len(y), dtype=np.intp)
    rng.shuffle(all_idx)
    chunks = np.array_split(all_idx, config.n_clients)
    # Enforce minimum — IID split can only fail if n_clients is huge
    client_lists = [ch.tolist() for ch in chunks]
    return _enforce_min_samples(client_lists, config.min_samples_per_client)


def _dirichlet_indices(y: np.ndarray, config: PartitionConfig) -> List[np.ndarray]:
    """Dirichlet-based non-IID partitioning.

    For each class a Dirichlet draw determines what fraction of that class's
    samples each client receives.  The approach preserves all samples and
    guarantees exact coverage.
    """
    rng = np.random.default_rng(config.seed)
    n_clients = config.n_clients
    unique_classes = np.unique(y)

    client_lists: List[List[int]] = [[] for _ in range(n_clients)]

    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)

        # Draw proportions and convert to integer counts
        props = rng.dirichlet(np.ones(n_clients) * config.alpha)
        counts = np.floor(props * n_cls).astype(int)

        # Distribute any remainder to clients with the largest fractional parts
        remainder = n_cls - int(counts.sum())
        if remainder > 0:
            frac_parts = props * n_cls - counts.astype(float)
            top = np.argsort(frac_parts)[::-1][:remainder]
            counts[top] += 1

        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            client_lists[client_id].extend(cls_idx[start:end].tolist())
            start = end

    return _enforce_min_samples(client_lists, config.min_samples_per_client)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def partition_data(
    x: np.ndarray,
    y: np.ndarray,
    config: PartitionConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition training data into per-client subsets.

    Args:
        x:      Feature array of shape (n_samples, n_features).
        y:      Label array of shape (n_samples,).
        config: Partitioning configuration.

    Returns:
        A list of (x_client, y_client) tuples, one per client.
        Each tuple contains numpy arrays with the client's samples.

    Raises:
        ValueError: If the min-samples constraint cannot be satisfied.
    """
    _check_feasibility(len(y), config)

    if config.method == "iid":
        indices = _iid_indices(y, config)
    else:
        indices = _dirichlet_indices(y, config)

    return [(x[idx], y[idx]) for idx in indices]


def compute_partition_stats(
    partitions: List[Tuple[np.ndarray, np.ndarray]],
    n_classes: int,
) -> List[Dict]:
    """Compute per-client sample counts and label distributions.

    Args:
        partitions: Output of :func:`partition_data`.
        n_classes:  Total number of classes (for consistent bincount length).

    Returns:
        A list of dicts, one per client, with keys:
        ``client_id``, ``n_samples``, ``class_counts``, ``class_fractions``.
    """
    stats = []
    for client_id, (_, y_c) in enumerate(partitions):
        counts = np.bincount(y_c, minlength=n_classes)
        fractions = (counts / len(y_c)).tolist() if len(y_c) > 0 else []
        stats.append(
            {
                "client_id": client_id,
                "n_samples": int(len(y_c)),
                "class_counts": counts.tolist(),
                "class_fractions": fractions,
            }
        )
    return stats
