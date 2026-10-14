"""Persistent client capability profiles and dimensional timing calculations."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class ClientProfile:
    """Persistent hardware and network capability profile for a federated client."""

    client_id: int
    speed_tier: str  # "fast", "medium", "slow"
    compute_factor: float  # Multiplier on local compute time (> 0.0)
    download_bandwidth: float  # Bytes per second (> 0.0)
    upload_bandwidth: float  # Bytes per second (> 0.0)
    base_latency: float  # Seconds (>= 0.0)
    data_partition_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.client_id < 0:
            raise ValueError(f"client_id must be >= 0, got {self.client_id}")
        if self.speed_tier not in {"fast", "medium", "slow"}:
            raise ValueError(f"speed_tier must be 'fast', 'medium', or 'slow', got {self.speed_tier}")
        if self.compute_factor <= 0.0:
            raise ValueError(f"compute_factor must be > 0.0, got {self.compute_factor}")
        if self.download_bandwidth <= 0.0:
            raise ValueError(f"download_bandwidth must be > 0.0, got {self.download_bandwidth}")
        if self.upload_bandwidth <= 0.0:
            raise ValueError(f"upload_bandwidth must be > 0.0, got {self.upload_bandwidth}")
        if self.base_latency < 0.0:
            raise ValueError(f"base_latency must be >= 0.0, got {self.base_latency}")
        if self.data_partition_id is None:
            self.data_partition_id = self.client_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_default_tier_specs() -> Dict[str, Dict[str, Any]]:
    """Return default hardware and connectivity specifications for each speed tier.

    All units are standardized:
    - compute_factor: unitless multiplier
    - download_bandwidth: Bytes/second (1 MB/s = 1,048,576 B/s)
    - upload_bandwidth: Bytes/second
    - base_latency: Seconds (0.010 s = 10 ms)
    """
    return {
        "fast": {
            "compute_factor": 0.5,
            "download_bandwidth": 2.0 * 1024.0 * 1024.0,  # 2 MB/s = 2,097,152 B/s
            "upload_bandwidth": 1.0 * 1024.0 * 1024.0,    # 1 MB/s = 1,048,576 B/s
            "base_latency": 0.010,                         # 10 ms
        },
        "medium": {
            "compute_factor": 1.0,
            "download_bandwidth": 512.0 * 1024.0,         # 512 KB/s = 524,288 B/s
            "upload_bandwidth": 256.0 * 1024.0,           # 256 KB/s = 262,144 B/s
            "base_latency": 0.050,                         # 50 ms
        },
        "slow": {
            "compute_factor": 2.5,
            "download_bandwidth": 64.0 * 1024.0,          # 64 KB/s = 65,536 B/s
            "upload_bandwidth": 32.0 * 1024.0,            # 32 KB/s = 32,768 B/s
            "base_latency": 0.200,                         # 200 ms
        },
    }


def create_client_profile(
    client_id: int,
    speed_tier: str,
    data_partition_id: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ClientProfile:
    """Create a validated ClientProfile based on standard tier defaults with optional overrides."""
    tier_specs = get_default_tier_specs()
    if speed_tier not in tier_specs:
        raise ValueError(f"Unknown speed_tier '{speed_tier}', choose from {list(tier_specs.keys())}")

    specs = dict(tier_specs[speed_tier])
    if overrides:
        specs.update(overrides)

    return ClientProfile(
        client_id=client_id,
        speed_tier=speed_tier,
        compute_factor=float(specs["compute_factor"]),
        download_bandwidth=float(specs["download_bandwidth"]),
        upload_bandwidth=float(specs["upload_bandwidth"]),
        base_latency=float(specs["base_latency"]),
        data_partition_id=data_partition_id if data_partition_id is not None else client_id,
    )


def compute_model_bytes(model: torch.nn.Module) -> int:
    """Calculate the exact total byte count of a PyTorch model's parameters."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return int(total_bytes)


def compute_training_time(
    n_samples: int,
    batch_size: int,
    local_epochs: int,
    compute_factor: float,
    base_step_time: float = 0.02,
) -> float:
    """Calculate deterministic local training duration in virtual seconds.

    Formula:
        steps_per_epoch = ceil(n_samples / batch_size)
        total_steps = local_epochs * steps_per_epoch
        T_train = total_steps * compute_factor * base_step_time

    Args:
        n_samples: Number of training samples on the client.
        batch_size: Mini-batch size.
        local_epochs: Number of local training epochs.
        compute_factor: Client hardware speed factor (>0).
        base_step_time: Calibrated virtual time per mini-batch gradient step (seconds).

    Returns:
        Virtual computation duration in seconds.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if local_epochs <= 0:
        raise ValueError(f"local_epochs must be > 0, got {local_epochs}")
    if compute_factor <= 0.0:
        raise ValueError(f"compute_factor must be > 0.0, got {compute_factor}")
    if base_step_time <= 0.0:
        raise ValueError(f"base_step_time must be > 0.0, got {base_step_time}")

    steps_per_epoch = math.ceil(float(n_samples) / float(batch_size))
    total_steps = local_epochs * steps_per_epoch
    duration = float(total_steps * compute_factor * base_step_time)
    return round(duration, 6)


def compute_transmission_time(
    model_bytes: int,
    bandwidth: float,
    latency: float,
) -> float:
    """Calculate deterministic network transmission time for downloading or uploading a model.

    Formula:
        T_comm = latency + (model_bytes / bandwidth)

    Args:
        model_bytes: Number of bytes to transfer.
        bandwidth: Transfer bandwidth in Bytes/second (>0).
        latency: Base propagation latency in seconds (>=0).

    Returns:
        Virtual transmission duration in seconds.
    """
    if model_bytes <= 0:
        raise ValueError(f"model_bytes must be > 0, got {model_bytes}")
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be > 0.0, got {bandwidth}")
    if latency < 0.0:
        raise ValueError(f"latency must be >= 0.0, got {latency}")

    duration = float(latency + (float(model_bytes) / float(bandwidth)))
    return round(duration, 6)


def couple_profiles_with_data_partitions(
    partition_stats: Dict[str, Any],
    profiles: List[ClientProfile],
    strategy: str = "most_skewed_to_slow",
) -> Tuple[List[ClientProfile], Dict[str, Any]]:
    """Deterministically map client capability profiles to data partitions.

    When strategy == 'most_skewed_to_slow':
    1. Computes total variation distance from uniform distribution for each client's labels.
    2. Sorts data partitions by skew descending (most skewed first).
    3. Sorts client profiles by capability ascending (slowest first: slow, medium, fast).
    4. Couples the most skewed data partition to the slowest device profile.

    Crucially, this changes ONLY the partition assignment pointer (`data_partition_id`),
    leaving the actual client data partitions, validation set, and test set untouched.

    Returns:
        coupled_profiles: List of ClientProfile with updated data_partition_id.
        coupling_audit: Detailed audit payload for reproducibility.
    """
    # Extract client class distributions
    skews: List[Tuple[int, float, List[int]]] = []
    for cid_str, stats in partition_stats.items():
        cid = int(cid_str)
        counts = stats.get("class_counts", [])
        total = sum(counts)
        if total > 0 and len(counts) > 0:
            fractions = np.array(counts, dtype=np.float64) / total
            uniform = np.ones_like(fractions) / len(fractions)
            tv_dist = 0.5 * float(np.sum(np.abs(fractions - uniform)))
        else:
            tv_dist = 0.0
        skews.append((cid, tv_dist, counts))

    # Sort partitions: highest skew first
    skews_sorted = sorted(skews, key=lambda x: x[1], reverse=True)
    partition_ids_ranked = [x[0] for x in skews_sorted]

    # Rank profiles: slowest first (highest compute_factor and latency, lowest bandwidth)
    def profile_slowness_score(p: ClientProfile) -> float:
        tier_weights = {"slow": 3.0, "medium": 2.0, "fast": 1.0}
        return tier_weights.get(p.speed_tier, 2.0) * p.compute_factor

    profiles_sorted_by_slowness = sorted(profiles, key=profile_slowness_score, reverse=True)

    coupled_profiles: List[ClientProfile] = []
    mapping: Dict[int, int] = {}
    audit_records: List[Dict[str, Any]] = []

    for rank, prof in enumerate(profiles_sorted_by_slowness):
        assigned_part_id = partition_ids_ranked[rank % len(partition_ids_ranked)]
        part_skew_info = next(x for x in skews if x[0] == assigned_part_id)

        coupled = ClientProfile(
            client_id=prof.client_id,
            speed_tier=prof.speed_tier,
            compute_factor=prof.compute_factor,
            download_bandwidth=prof.download_bandwidth,
            upload_bandwidth=prof.upload_bandwidth,
            base_latency=prof.base_latency,
            data_partition_id=assigned_part_id,
        )
        coupled_profiles.append(coupled)
        mapping[prof.client_id] = assigned_part_id

        audit_records.append({
            "client_id": prof.client_id,
            "speed_tier": prof.speed_tier,
            "assigned_partition_id": assigned_part_id,
            "partition_skew_tv_distance": round(part_skew_info[1], 4),
            "partition_class_counts": part_skew_info[2],
        })

    # Sort back by client_id for predictable indexing
    coupled_profiles.sort(key=lambda p: p.client_id)

    coupling_audit = {
        "strategy": strategy,
        "client_to_partition_mapping": {str(k): v for k, v in sorted(mapping.items())},
        "audit_records": audit_records,
    }

    return coupled_profiles, coupling_audit
