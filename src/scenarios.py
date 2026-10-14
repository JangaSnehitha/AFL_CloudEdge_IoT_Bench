"""Dynamic Cloud-Edge IoT scenario templates, network perturbation models, and outage events."""

import copy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.client_profiles import ClientProfile, create_client_profile


@dataclass
class NetworkConditionEvent:
    """Piecewise virtual-time network perturbation (latency spike, bandwidth throttle, etc.)."""

    event_type: str  # "latency_spike", "bandwidth_throttle", "recovery", "permanent_improvement"
    start_time: float  # Virtual time when perturbation activates
    end_time: Optional[float] = None  # None indicates permanent step change
    client_ids: Optional[List[int]] = None  # None applies to all clients
    latency_delta: float = 0.0  # Additive latency disturbance in seconds
    bandwidth_multiplier: float = 1.0  # Multiplier on bandwidth (e.g. 0.25 = 75% throttle)

    def __post_init__(self) -> None:
        if self.start_time < 0.0:
            raise ValueError(f"start_time must be >= 0.0, got {self.start_time}")
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError(f"end_time ({self.end_time}) cannot be less than start_time ({self.start_time})")
        if self.bandwidth_multiplier <= 0.0:
            raise ValueError(f"bandwidth_multiplier must be > 0.0, got {self.bandwidth_multiplier}")


@dataclass
class OutageEvent:
    """Scheduled client outage interval where the device is offline/unreachable."""

    client_ids: List[int]
    start_time: float
    end_time: float
    failure_stage: str = "BEFORE_UPLOAD"  # "BEFORE_DOWNLOAD", "DURING_TRAINING", "BEFORE_UPLOAD"

    def __post_init__(self) -> None:
        if not self.client_ids:
            raise ValueError("client_ids cannot be empty")
        if self.start_time < 0.0:
            raise ValueError(f"start_time must be >= 0.0, got {self.start_time}")
        if self.end_time <= self.start_time:
            raise ValueError(f"end_time ({self.end_time}) must be strictly greater than start_time ({self.start_time})")
        valid_stages = {"BEFORE_DOWNLOAD", "DURING_TRAINING", "BEFORE_UPLOAD"}
        if self.failure_stage not in valid_stages:
            raise ValueError(f"failure_stage must be one of {valid_stages}, got {self.failure_stage}")


@dataclass
class ScenarioConfig:
    """Complete definition of a Cloud-Edge IoT dynamic benchmark scenario."""

    scenario_name: str
    description: str
    n_clients: int = 5
    target_applied_updates: int = 15
    max_attempted_jobs: int = 100
    max_virtual_time: float = 1000.0
    rejoin_policy: str = "fresh_restart"  # "fresh_restart" or "retained_update"
    job_failure_probability: float = 0.0
    network_events: List[NetworkConditionEvent] = field(default_factory=list)
    outage_events: List[OutageEvent] = field(default_factory=list)
    tier_distribution: Dict[str, float] = field(
        default_factory=lambda: {"fast": 0.2, "medium": 0.6, "slow": 0.2}
    )
    coupling_enabled: bool = False
    partition_method: str = "dirichlet"
    alpha: float = 0.5
    data_seed: int = 42
    profile_seed: int = 42
    network_seed: int = 42
    dropout_seed: int = 42

    def __post_init__(self) -> None:
        if self.n_clients <= 0:
            raise ValueError(f"n_clients must be > 0, got {self.n_clients}")
        if self.target_applied_updates <= 0:
            raise ValueError(f"target_applied_updates must be > 0, got {self.target_applied_updates}")
        if self.max_attempted_jobs <= 0:
            raise ValueError(f"max_attempted_jobs must be > 0, got {self.max_attempted_jobs}")
        if self.max_virtual_time <= 0.0:
            raise ValueError(f"max_virtual_time must be > 0.0, got {self.max_virtual_time}")
        if self.rejoin_policy not in {"fresh_restart", "retained_update"}:
            raise ValueError(f"rejoin_policy must be 'fresh_restart' or 'retained_update', got {self.rejoin_policy}")
        if not (0.0 <= self.job_failure_probability <= 1.0):
            raise ValueError(f"job_failure_probability must be in [0, 1], got {self.job_failure_probability}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class DynamicNetworkManager:
    """Evaluates dynamic network conditions and outage states across virtual time."""

    def __init__(
        self,
        network_events: List[NetworkConditionEvent],
        outage_events: List[OutageEvent],
    ) -> None:
        self.network_events = list(network_events)
        self.outage_events = list(outage_events)

    def get_effective_network(
        self,
        client_id: int,
        virtual_time: float,
        base_profile: ClientProfile,
    ) -> Tuple[float, float, float]:
        """Sample network conditions at virtual_time.

        Rule: Sampled at transmission start (t_start).
        Returns:
            (effective_download_bw, effective_upload_bw, effective_latency)
        """
        dl_bw = base_profile.download_bandwidth
        ul_bw = base_profile.upload_bandwidth
        lat = base_profile.base_latency

        for ev in self.network_events:
            if ev.client_ids is not None and client_id not in ev.client_ids:
                continue
            if ev.start_time <= virtual_time:
                if ev.end_time is None or virtual_time < ev.end_time:
                    lat += ev.latency_delta
                    dl_bw *= ev.bandwidth_multiplier
                    ul_bw *= ev.bandwidth_multiplier

        # Guard against non-positive bandwidth or negative latency
        dl_bw = max(1024.0, float(dl_bw))
        ul_bw = max(1024.0, float(ul_bw))
        lat = max(0.0, float(lat))

        return dl_bw, ul_bw, lat

    def is_in_outage(
        self,
        client_id: int,
        virtual_time: float,
        stage: Optional[str] = None,
    ) -> Tuple[bool, Optional[OutageEvent]]:
        """Check if client_id is currently experiencing a scheduled outage at virtual_time."""
        for ev in self.outage_events:
            if client_id in ev.client_ids:
                if stage is None or ev.failure_stage == stage:
                    if ev.start_time <= virtual_time < ev.end_time:
                        return True, ev
        return False, None

    def get_intersecting_outage(
        self,
        client_id: int,
        interval_start: float,
        interval_end: float,
        stage: Optional[str] = None,
    ) -> Optional[OutageEvent]:
        """Detect if client experiences an outage intersecting [interval_start, interval_end]."""
        for ev in self.outage_events:
            if client_id in ev.client_ids:
                if stage is None or ev.failure_stage == stage:
                    if max(interval_start, ev.start_time) < min(interval_end, ev.end_time):
                        return ev
        return None


# -------------------------------------------------------------------------
# Standard Scenario Templates (S1 - S5)
# -------------------------------------------------------------------------

def get_scenario_s1(n_clients: int = 5, target_updates: int = 15) -> ScenarioConfig:
    """S1: Stable Edge - Near-IID data, mild compute variation, stable network, always online."""
    return ScenarioConfig(
        scenario_name="S1_stable_edge",
        description="Near-IID data with mild compute variation, stable network bandwidth, and zero outages.",
        n_clients=n_clients,
        target_applied_updates=target_updates,
        partition_method="iid",
        alpha=10.0,
        rejoin_policy="fresh_restart",
        job_failure_probability=0.0,
        network_events=[],
        outage_events=[],
        tier_distribution={"fast": 0.4, "medium": 0.6, "slow": 0.0},
        coupling_enabled=False,
    )


def get_scenario_s2(n_clients: int = 5, target_updates: int = 15) -> ScenarioConfig:
    """S2: Statistical Skew - Severe non-IID data with identical stable edge system conditions."""
    return ScenarioConfig(
        scenario_name="S2_statistical_skew",
        description="Severe non-IID Dirichlet partition with identical stable edge physical conditions.",
        n_clients=n_clients,
        target_applied_updates=target_updates,
        partition_method="dirichlet",
        alpha=0.1,
        rejoin_policy="fresh_restart",
        job_failure_probability=0.0,
        network_events=[],
        outage_events=[],
        tier_distribution={"fast": 0.4, "medium": 0.6, "slow": 0.0},
        coupling_enabled=False,
    )


def get_scenario_s3(n_clients: int = 5, target_updates: int = 15) -> ScenarioConfig:
    """S3: Resource Imbalance - Moderate non-IID data with heterogeneous fast/med/slow profiles and network jitter."""
    jitter_event = NetworkConditionEvent(
        event_type="latency_spike",
        start_time=0.15,
        end_time=0.80,
        client_ids=None,
        latency_delta=0.100,  # +100ms latency spike
        bandwidth_multiplier=0.5,  # 50% bandwidth throttle
    )
    return ScenarioConfig(
        scenario_name="S3_resource_imbalance",
        description="Moderate non-IID data with heterogeneous device tiers and a temporary network disturbance window.",
        n_clients=n_clients,
        target_applied_updates=target_updates,
        partition_method="dirichlet",
        alpha=0.5,
        rejoin_policy="fresh_restart",
        job_failure_probability=0.0,
        network_events=[jitter_event],
        outage_events=[],
        tier_distribution={"fast": 0.2, "medium": 0.4, "slow": 0.4},
        coupling_enabled=False,
    )


def get_scenario_s4(n_clients: int = 5, target_updates: int = 15) -> ScenarioConfig:
    """S4: Unreliable IoT - Moderate non-IID data with heterogeneous compute, network disruption, outage and rejoin."""
    outage_client_0 = OutageEvent(
        client_ids=[0],
        start_time=0.20,
        end_time=0.40,
        failure_stage="BEFORE_UPLOAD",
    )
    # Bandwidth throttle on slow tier
    throttle_event = NetworkConditionEvent(
        event_type="bandwidth_throttle",
        start_time=0.10,
        end_time=0.80,
        client_ids=[1, 3],
        latency_delta=0.050,
        bandwidth_multiplier=0.25,  # 75% reduction
    )
    return ScenarioConfig(
        scenario_name="S4_unreliable_iot",
        description="Unreliable IoT system with network degradation, scheduled client outage, and retained update rejoin.",
        n_clients=n_clients,
        target_applied_updates=target_updates,
        partition_method="dirichlet",
        alpha=0.5,
        rejoin_policy="retained_update",
        job_failure_probability=0.0,
        network_events=[throttle_event],
        outage_events=[outage_client_0],
        tier_distribution={"fast": 0.2, "medium": 0.4, "slow": 0.4},
        coupling_enabled=False,
    )


def get_scenario_s5(n_clients: int = 5, target_updates: int = 15) -> ScenarioConfig:
    """S5: Coupled Stress - Strong non-IID data where skewed clients are coupled to slow tiers and disrupted."""
    coupled_outage = OutageEvent(
        client_ids=[0],
        start_time=0.15,
        end_time=0.45,
        failure_stage="BEFORE_UPLOAD",
    )
    coupled_throttle = NetworkConditionEvent(
        event_type="bandwidth_throttle",
        start_time=0.05,
        end_time=0.80,
        client_ids=[0],
        latency_delta=0.150,
        bandwidth_multiplier=0.2,
    )
    return ScenarioConfig(
        scenario_name="S5_coupled_representation_stress",
        description="Coupled representation stress: highly skewed data partitions mapped to slow disrupted clients.",
        n_clients=n_clients,
        target_applied_updates=target_updates,
        partition_method="dirichlet",
        alpha=0.2,
        rejoin_policy="retained_update",
        job_failure_probability=0.0,
        network_events=[coupled_throttle],
        outage_events=[coupled_outage],
        tier_distribution={"fast": 0.2, "medium": 0.4, "slow": 0.4},
        coupling_enabled=True,
    )


def get_named_scenario(name: str, n_clients: int = 5, target_updates: int = 15) -> ScenarioConfig:
    """Factory helper to obtain a named scenario configuration."""
    name_clean = name.strip().lower()
    if name_clean in {"s1", "s1_stable_edge", "stable_edge"}:
        return get_scenario_s1(n_clients, target_updates)
    elif name_clean in {"s2", "s2_statistical_skew", "statistical_skew"}:
        return get_scenario_s2(n_clients, target_updates)
    elif name_clean in {"s3", "s3_resource_imbalance", "resource_imbalance"}:
        return get_scenario_s3(n_clients, target_updates)
    elif name_clean in {"s4", "s4_unreliable_iot", "unreliable_iot"}:
        return get_scenario_s4(n_clients, target_updates)
    elif name_clean in {"s5", "s5_coupled_stress", "coupled_stress", "s5_coupled_representation_stress"}:
        return get_scenario_s5(n_clients, target_updates)
    else:
        raise ValueError(f"Unknown scenario name '{name}'. Choose from S1, S2, S3, S4, S5.")
