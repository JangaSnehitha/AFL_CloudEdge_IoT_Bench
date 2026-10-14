"""Dynamic Cloud-Edge IoT scenario simulation engine for FedAsync and FedBuff."""

import argparse
import collections
import copy
from dataclasses import asdict, dataclass
import heapq
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.client import (
    get_model_parameters,
    load_partitioned_dataloaders,
    set_model_parameters,
)
from src.client_profiles import (
    ClientProfile,
    compute_model_bytes,
    compute_training_time,
    compute_transmission_time,
    couple_profiles_with_data_partitions,
    create_client_profile,
    get_default_tier_specs,
)
from src.dataset import SyntheticDatasetConfig
from src.fedasync import compute_fedasync_weight, fedasync_aggregate_parameters, train_client_locally
from src.fedbuff import aggregate_buffered_parameters
from src.model import MLPConfig, build_criterion, build_model
from src.partition import PartitionConfig
from src.scenarios import (
    DynamicNetworkManager,
    NetworkConditionEvent,
    OutageEvent,
    ScenarioConfig,
    get_named_scenario,
)
from src.trainer import evaluate
from src.divergence_metrics import (
    aggregate_buffer_predictors,
    compute_all_predictors,
    flatten_params,
)
from src.fairness_metrics import (
    compile_fairness_summary,
    evaluate_per_class_held_out,
)
from src.impact_metrics import (
    compute_action_impact,
    evaluate_model_loss,
)
from src.predictor_analysis import (
    analyze_action_observations,
    export_actions_to_csv,
)


def get_git_commit() -> str:
    """Retrieve the current Git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class ScenarioSimulator:
    """Event-driven simulation engine modeling dynamic Cloud-Edge IoT environments."""

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        algorithm: str = "FedAsync",  # "FedAsync" or "FedBuff"
        buffer_size: int = 5,
        flush_final_buffer: bool = False,
        alpha_0: float = 1.0,
        local_epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.01,
        dataset_config: Optional[SyntheticDatasetConfig] = None,
        model_config: Optional[MLPConfig] = None,
        enable_analysis: bool = False,
        impact_tolerance: float = 1e-5,
    ) -> None:
        self.scenario_config = scenario_config
        self.algorithm = algorithm
        self.buffer_size = buffer_size
        self.flush_final_buffer = flush_final_buffer
        self.alpha_0 = alpha_0
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr

        if self.algorithm not in {"FedAsync", "FedBuff"}:
            raise ValueError(f"Unsupported algorithm '{self.algorithm}', choose 'FedAsync' or 'FedBuff'")

        # Seed dedicated RNG streams
        self.data_rng = np.random.RandomState(scenario_config.data_seed)
        self.profile_rng = np.random.RandomState(scenario_config.profile_seed)
        self.network_rng = np.random.RandomState(scenario_config.network_seed)
        self.dropout_rng = np.random.RandomState(scenario_config.dropout_seed)

        torch.manual_seed(scenario_config.data_seed)
        np.random.seed(scenario_config.data_seed)

        # Dataset configuration
        if dataset_config is None:
            ds_path = Path("configs/dataset_config.json")
            if ds_path.exists():
                self.dataset_config = SyntheticDatasetConfig.from_json(ds_path)
            else:
                self.dataset_config = SyntheticDatasetConfig()
        else:
            self.dataset_config = copy.deepcopy(dataset_config)
        self.dataset_config.seed = scenario_config.data_seed

        # Model configuration
        if model_config is None:
            model_path = Path("configs/model_config.json")
            if model_path.exists():
                self.model_config = MLPConfig.from_json(model_path)
            else:
                self.model_config = MLPConfig()
        else:
            self.model_config = copy.deepcopy(model_config)
        self.model_config.lr = lr

        # Load data partitions
        partition_config = PartitionConfig(
            n_clients=scenario_config.n_clients,
            method=scenario_config.partition_method,
            alpha=scenario_config.alpha,
            seed=scenario_config.data_seed,
            min_samples_per_client=5,
        )
        (
            self.client_loaders,
            self.val_loader,
            self.test_loader,
            self.partition_stats,
        ) = load_partitioned_dataloaders(
            dataset_config=self.dataset_config,
            partition_config=partition_config,
            batch_size=batch_size,
            return_stats=True,
        )

        # Build server model
        self.server_model = build_model(self.model_config)
        self.server_params = get_model_parameters(self.server_model)
        self.criterion = build_criterion()
        self.model_bytes = compute_model_bytes(self.server_model)

        # Assign client capability profiles
        self.profiles: List[ClientProfile] = self._assign_profiles()
        self.coupling_audit: Dict[str, Any] = {}

        if scenario_config.coupling_enabled:
            self.profiles, self.coupling_audit = couple_profiles_with_data_partitions(
                partition_stats=self.partition_stats,
                profiles=self.profiles,
                strategy="most_skewed_to_slow",
            )
        else:
            self.coupling_audit = {
                "strategy": "uncoupled",
                "client_to_partition_mapping": {str(p.client_id): p.data_partition_id for p in self.profiles},
            }

        # Network and outage manager
        self.network_mgr = DynamicNetworkManager(
            network_events=scenario_config.network_events,
            outage_events=scenario_config.outage_events,
        )

        # Simulation state
        self.server_version: int = 0
        self.current_time: float = 0.0
        self.event_counter: int = 0
        self.job_counter: int = 0

        # Accounting counters
        self.jobs_dispatched: int = 0
        self.training_starts: int = 0
        self.training_completions: int = 0
        self.failed_jobs: int = 0
        self.failures_by_stage: Dict[str, int] = collections.defaultdict(int)
        self.retained_updates_count: int = 0
        self.arrival_count: int = 0
        self.applied_updates: int = 0
        self.server_aggregation_count: int = 0
        self.total_bytes_transferred: int = 0

        # Client active jobs tracking (at most 1 active or retained job per client)
        self.active_jobs: Dict[int, Dict[str, Any]] = {}

        # FedBuff buffer
        self.buffer: List[Dict[str, Any]] = []

        # Event queue: priority queue storing (timestamp, priority, event_counter, event_dict)
        # Priority: 1: DISPATCH / REJOIN, 2: DOWNLOAD_COMPLETE, 3: TRAIN_COMPLETE, 4: UPDATE_ARRIVAL
        self.event_queue: List[Tuple[float, int, int, Dict[str, Any]]] = []

        # Logs
        self.event_log: List[Dict[str, Any]] = []
        self.arrival_log: List[Dict[str, Any]] = []
        self.aggregation_log: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.initial_metrics: Dict[str, Any] = {}

        self.per_client_arrivals: Dict[int, int] = collections.defaultdict(int)
        self.per_client_applied: Dict[int, int] = collections.defaultdict(int)
        self.per_tier_applied: Dict[str, int] = collections.defaultdict(int)
        self.client_samples_uploaded: Dict[int, int] = collections.defaultdict(int)

        # Phase 15 Divergence-vs-Staleness Analysis
        self.enable_analysis: bool = enable_analysis
        self.impact_tolerance: float = impact_tolerance
        self.prev_server_delta: Optional[torch.Tensor] = None
        self.action_observations: List[Dict[str, Any]] = []

    def _assign_profiles(self) -> List[ClientProfile]:
        """Assign fast/medium/slow profiles to clients based on configured tier distribution."""
        n = self.scenario_config.n_clients
        tier_dist = self.scenario_config.tier_distribution
        tiers = ["fast", "medium", "slow"]
        probs = [tier_dist.get(t, 0.0) for t in tiers]
        total_p = sum(probs)
        if total_p <= 0.0:
            probs = [0.2, 0.6, 0.2]
            total_p = 1.0
        normalized_probs = [p / total_p for p in probs]

        profiles = []
        for cid in range(n):
            chosen_tier = str(self.profile_rng.choice(tiers, p=normalized_probs))
            prof = create_client_profile(client_id=cid, speed_tier=chosen_tier, data_partition_id=cid)
            profiles.append(prof)
        return profiles

    def _push_event(self, timestamp: float, priority: int, event_data: Dict[str, Any]) -> None:
        self.event_counter += 1
        heapq.heappush(self.event_queue, (timestamp, priority, self.event_counter, event_data))

    def _dispatch_client(self, client_id: int, dispatch_time: float) -> Optional[str]:
        """Dispatch a client with a model download task."""
        # Enforce hard constraints: at most one active/retained job per client
        if client_id in self.active_jobs:
            return None

        # Check stopping budget
        if self.applied_updates >= self.scenario_config.target_applied_updates:
            return None
        if self.jobs_dispatched >= self.scenario_config.max_attempted_jobs:
            return None

        self.job_counter += 1
        job_id = f"job_c{client_id}_s{self.job_counter}"
        self.jobs_dispatched += 1

        profile = self.profiles[client_id]

        # Check if client is currently in outage before download
        in_outage, outage_ev = self.network_mgr.is_in_outage(client_id, dispatch_time, stage="BEFORE_DOWNLOAD")
        if in_outage and outage_ev is not None:
            self.failed_jobs += 1
            self.failures_by_stage["BEFORE_DOWNLOAD"] += 1
            self.event_log.append({
                "time": round(float(dispatch_time), 4),
                "event_type": "JOB_FAILED",
                "stage": "BEFORE_DOWNLOAD",
                "client_id": client_id,
                "job_id": job_id,
                "reason": f"Client in outage until t={outage_ev.end_time:.2f}",
            })
            # Schedule rejoin dispatch after outage ends
            self._push_event(outage_ev.end_time, 1, {
                "event_type": "REJOIN_DISPATCH",
                "client_id": client_id,
            })
            return None

        # Stochastic job failure check
        if self.dropout_rng.uniform(0.0, 1.0) < self.scenario_config.job_failure_probability:
            self.failed_jobs += 1
            self.failures_by_stage["BEFORE_DOWNLOAD"] += 1
            self.event_log.append({
                "time": round(float(dispatch_time), 4),
                "event_type": "JOB_FAILED",
                "stage": "BEFORE_DOWNLOAD",
                "client_id": client_id,
                "job_id": job_id,
                "reason": "Stochastic transmission failure before download",
            })
            # Retry after backoff
            self._push_event(dispatch_time + 1.0, 1, {
                "event_type": "REJOIN_DISPATCH",
                "client_id": client_id,
            })
            return None

        # Sample download network condition at dispatch_time
        dl_bw, _, lat = self.network_mgr.get_effective_network(client_id, dispatch_time, profile)
        download_duration = compute_transmission_time(self.model_bytes, dl_bw, lat)
        download_complete_time = dispatch_time + download_duration

        job_info = {
            "job_id": job_id,
            "client_id": client_id,
            "status": "DOWNLOADING",
            "dispatch_time": dispatch_time,
            "download_complete_time": download_complete_time,
            "download_bandwidth": dl_bw,
            "download_latency": lat,
            "download_duration": download_duration,
        }
        self.active_jobs[client_id] = job_info

        self.event_log.append({
            "time": round(float(dispatch_time), 4),
            "event_type": "DISPATCH",
            "client_id": client_id,
            "job_id": job_id,
            "speed_tier": profile.speed_tier,
            "download_bandwidth_bps": round(dl_bw, 1),
            "latency_s": round(lat, 4),
            "expected_download_time_s": round(download_duration, 4),
        })

        self._push_event(download_complete_time, 2, {
            "event_type": "DOWNLOAD_COMPLETE",
            "client_id": client_id,
            "job_id": job_id,
            "dispatch_time": dispatch_time,
        })
        return job_id

    def run_simulation(self) -> Dict[str, Any]:
        """Execute the event-driven dynamic scenario simulation."""
        # Initial evaluation at t=0.0
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

        # Initially dispatch all clients at t=0.0
        for cid in range(self.scenario_config.n_clients):
            self._dispatch_client(client_id=cid, dispatch_time=0.0)

        stopping_reason = "queue_exhausted"

        while self.event_queue:
            # Check stopping budgets
            if self.applied_updates >= self.scenario_config.target_applied_updates:
                stopping_reason = "target_applied_updates_reached"
                break
            if self.current_time >= self.scenario_config.max_virtual_time:
                stopping_reason = "max_virtual_time_reached"
                break
            if self.jobs_dispatched >= self.scenario_config.max_attempted_jobs:
                if not any(j["status"] in {"DOWNLOADING", "TRAINING", "UPLOADING", "RETAINED"} for j in self.active_jobs.values()):
                    stopping_reason = "max_attempted_jobs_reached"
                    break

            event_time, _, _, event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time
            event_type = event_data["event_type"]

            if event_type == "REJOIN_DISPATCH":
                cid = event_data["client_id"]
                if cid not in self.active_jobs:
                    self._dispatch_client(client_id=cid, dispatch_time=event_time)

            elif event_type == "DOWNLOAD_COMPLETE":
                cid = event_data["client_id"]
                job_id = event_data["job_id"]
                dispatch_time = event_data["dispatch_time"]

                job = self.active_jobs.get(cid)
                if not job or job["job_id"] != job_id:
                    continue

                # Check if an outage hit during model download
                outage = self.network_mgr.get_intersecting_outage(cid, dispatch_time, event_time, stage="BEFORE_DOWNLOAD")
                if outage is not None:
                    self.failed_jobs += 1
                    self.failures_by_stage["BEFORE_DOWNLOAD"] += 1
                    self.active_jobs.pop(cid, None)
                    self.event_log.append({
                        "time": round(float(event_time), 4),
                        "event_type": "JOB_FAILED",
                        "stage": "BEFORE_DOWNLOAD",
                        "client_id": cid,
                        "job_id": job_id,
                        "reason": f"Outage interrupted download between t={dispatch_time:.2f} and t={event_time:.2f}",
                    })
                    # Schedule fresh retry after outage
                    self._push_event(outage.end_time, 1, {
                        "event_type": "REJOIN_DISPATCH",
                        "client_id": cid,
                    })
                    continue

                # Model download completed successfully
                self.training_starts += 1
                self.total_bytes_transferred += self.model_bytes
                profile = self.profiles[cid]

                # Client obtains current global model snapshot
                training_version = self.server_version
                training_params = copy.deepcopy(self.server_params)

                data_part_id = profile.data_partition_id
                part_loader = self.client_loaders[data_part_id]
                n_samples = len(part_loader.dataset)

                compute_duration = compute_training_time(
                    n_samples=n_samples,
                    batch_size=self.batch_size,
                    local_epochs=self.local_epochs,
                    compute_factor=profile.compute_factor,
                )
                train_complete_time = event_time + compute_duration

                job["status"] = "TRAINING"
                job["training_version"] = training_version
                job["train_start_time"] = event_time
                job["train_complete_time"] = train_complete_time
                job["compute_duration"] = compute_duration
                job["n_samples"] = n_samples
                job["base_params"] = copy.deepcopy(training_params)

                self.event_log.append({
                    "time": round(float(event_time), 4),
                    "event_type": "DOWNLOAD_COMPLETE",
                    "client_id": cid,
                    "job_id": job_id,
                    "training_version": training_version,
                    "n_samples": n_samples,
                    "compute_factor": profile.compute_factor,
                    "expected_compute_time_s": round(compute_duration, 4),
                })

                # Perform deterministic local client training
                trained_params, local_loss, _ = train_client_locally(
                    model_config=self.model_config,
                    train_loader=part_loader,
                    initial_params=training_params,
                    local_epochs=self.local_epochs,
                    lr=self.lr,
                )
                job["trained_params"] = trained_params
                job["local_loss"] = local_loss

                self._push_event(train_complete_time, 3, {
                    "event_type": "TRAIN_COMPLETE",
                    "client_id": cid,
                    "job_id": job_id,
                    "train_start_time": event_time,
                })

            elif event_type == "TRAIN_COMPLETE":
                cid = event_data["client_id"]
                job_id = event_data["job_id"]
                train_start_time = event_data["train_start_time"]

                job = self.active_jobs.get(cid)
                if not job or job["job_id"] != job_id:
                    continue

                # Check if an outage occurred during local training
                outage = self.network_mgr.get_intersecting_outage(cid, train_start_time, event_time)
                if outage is not None and outage.failure_stage == "DURING_TRAINING":
                    self.failed_jobs += 1
                    self.failures_by_stage["DURING_TRAINING"] += 1
                    self.active_jobs.pop(cid, None)
                    self.event_log.append({
                        "time": round(float(event_time), 4),
                        "event_type": "JOB_FAILED",
                        "stage": "DURING_TRAINING",
                        "client_id": cid,
                        "job_id": job_id,
                        "reason": f"Device outage during training: [{outage.start_time:.2f}, {outage.end_time:.2f}]",
                    })
                    self._push_event(outage.end_time, 1, {
                        "event_type": "REJOIN_DISPATCH",
                        "client_id": cid,
                    })
                    continue

                # Training successfully completed
                self.training_completions += 1
                profile = self.profiles[cid]

                self.event_log.append({
                    "time": round(float(event_time), 4),
                    "event_type": "TRAIN_COMPLETE",
                    "client_id": cid,
                    "job_id": job_id,
                    "local_loss": round(float(job["local_loss"]), 6),
                })

                # Check connectivity for upload
                in_outage, outage_ev = self.network_mgr.is_in_outage(cid, event_time, stage="BEFORE_UPLOAD")
                if in_outage and outage_ev is not None:
                    # Client completed training but cannot upload due to active outage
                    if self.scenario_config.rejoin_policy == "retained_update":
                        self.retained_updates_count += 1
                        job["status"] = "RETAINED"
                        self.event_log.append({
                            "time": round(float(event_time), 4),
                            "event_type": "JOB_RETAINED",
                            "client_id": cid,
                            "job_id": job_id,
                            "training_version": job["training_version"],
                            "rejoin_time": outage_ev.end_time,
                            "reason": "Training completed before outage; retaining update until rejoin",
                        })
                        self._push_event(outage_ev.end_time, 1, {
                            "event_type": "REJOIN_UPLOAD",
                            "client_id": cid,
                            "job_id": job_id,
                        })
                    else:
                        # Fresh restart policy: discard incomplete work
                        self.failed_jobs += 1
                        self.failures_by_stage["BEFORE_UPLOAD"] += 1
                        self.active_jobs.pop(cid, None)
                        self.event_log.append({
                            "time": round(float(event_time), 4),
                            "event_type": "JOB_FAILED",
                            "stage": "BEFORE_UPLOAD",
                            "client_id": cid,
                            "job_id": job_id,
                            "reason": "Outage before upload; discarding update under fresh_restart policy",
                        })
                        self._push_event(outage_ev.end_time, 1, {
                            "event_type": "REJOIN_DISPATCH",
                            "client_id": cid,
                        })
                    continue

                # Normal upload scheduling
                _, ul_bw, lat = self.network_mgr.get_effective_network(cid, event_time, profile)
                upload_duration = compute_transmission_time(self.model_bytes, ul_bw, lat)
                arrival_time = event_time + upload_duration

                job["status"] = "UPLOADING"
                job["upload_start_time"] = event_time
                job["arrival_time"] = arrival_time
                job["upload_bandwidth"] = ul_bw
                job["upload_latency"] = lat
                job["upload_duration"] = upload_duration

                self._push_event(arrival_time, 4, {
                    "event_type": "UPDATE_ARRIVAL",
                    "client_id": cid,
                    "job_id": job_id,
                    "upload_start_time": event_time,
                })

            elif event_type == "REJOIN_UPLOAD":
                cid = event_data["client_id"]
                job_id = event_data["job_id"]

                job = self.active_jobs.get(cid)
                if not job or job["job_id"] != job_id or job["status"] != "RETAINED":
                    continue

                profile = self.profiles[cid]
                _, ul_bw, lat = self.network_mgr.get_effective_network(cid, event_time, profile)
                upload_duration = compute_transmission_time(self.model_bytes, ul_bw, lat)
                arrival_time = event_time + upload_duration

                job["status"] = "UPLOADING"
                job["upload_start_time"] = event_time
                job["arrival_time"] = arrival_time
                job["upload_bandwidth"] = ul_bw
                job["upload_latency"] = lat
                job["upload_duration"] = upload_duration

                self.event_log.append({
                    "time": round(float(event_time), 4),
                    "event_type": "REJOIN_UPLOAD_STARTED",
                    "client_id": cid,
                    "job_id": job_id,
                    "retained_training_version": job["training_version"],
                    "upload_bandwidth_bps": round(ul_bw, 1),
                    "expected_upload_time_s": round(upload_duration, 4),
                })

                self._push_event(arrival_time, 4, {
                    "event_type": "UPDATE_ARRIVAL",
                    "client_id": cid,
                    "job_id": job_id,
                    "upload_start_time": event_time,
                })

            elif event_type == "UPDATE_ARRIVAL":
                cid = event_data["client_id"]
                job_id = event_data["job_id"]
                upload_start_time = event_data["upload_start_time"]

                job = self.active_jobs.get(cid)
                if not job or job["job_id"] != job_id:
                    continue

                # Check if upload was disrupted by an outage
                outage = self.network_mgr.get_intersecting_outage(cid, upload_start_time, event_time, stage="BEFORE_UPLOAD")
                if outage is not None:
                    if self.scenario_config.rejoin_policy == "retained_update":
                        self.retained_updates_count += 1
                        job["status"] = "RETAINED"
                        self.event_log.append({
                            "time": round(float(event_time), 4),
                            "event_type": "JOB_RETAINED",
                            "client_id": cid,
                            "job_id": job_id,
                            "training_version": job["training_version"],
                            "rejoin_time": outage.end_time,
                            "reason": "Upload disrupted by outage; retaining update until rejoin",
                        })
                        self._push_event(outage.end_time, 1, {
                            "event_type": "REJOIN_UPLOAD",
                            "client_id": cid,
                            "job_id": job_id,
                        })
                    else:
                        self.failed_jobs += 1
                        self.failures_by_stage["BEFORE_UPLOAD"] += 1
                        self.active_jobs.pop(cid, None)
                        self.event_log.append({
                            "time": round(float(event_time), 4),
                            "event_type": "JOB_FAILED",
                            "stage": "BEFORE_UPLOAD",
                            "client_id": cid,
                            "job_id": job_id,
                            "reason": "Upload disrupted by outage under fresh_restart policy",
                        })
                        self._push_event(outage.end_time, 1, {
                            "event_type": "REJOIN_DISPATCH",
                            "client_id": cid,
                        })
                    continue

                # Arrival successfully processed
                self.arrival_count += 1
                self.total_bytes_transferred += self.model_bytes
                self.per_client_arrivals[cid] += 1
                self.client_samples_uploaded[cid] += job["n_samples"]

                client_training_version = job["training_version"]
                server_version_at_arrival = self.server_version
                staleness = int(server_version_at_arrival - client_training_version)
                if staleness < 0:
                    raise RuntimeError(f"Negative staleness: server={server_version_at_arrival}, client={client_training_version}")

                profile = self.profiles[cid]
                trained_params = job["trained_params"]
                local_loss = job["local_loss"]
                sample_count = job["n_samples"]

                arrival_record = {
                    "arrival_index": self.arrival_count,
                    "client_id": cid,
                    "job_id": job_id,
                    "speed_tier": profile.speed_tier,
                    "dispatch_time": round(float(job["dispatch_time"]), 4),
                    "download_complete_time": round(float(job["download_complete_time"]), 4),
                    "train_start_time": round(float(job["train_start_time"]), 4),
                    "train_complete_time": round(float(job["train_complete_time"]), 4),
                    "arrival_time": round(float(event_time), 4),
                    "client_training_version": client_training_version,
                    "server_version_at_arrival": server_version_at_arrival,
                    "staleness": staleness,
                    "sample_count": sample_count,
                    "local_loss": round(float(local_loss), 6),
                    "total_turnaround_time": round(float(event_time - job["dispatch_time"]), 4),
                }
                self.arrival_log.append(arrival_record)

                # Free active job slot
                self.active_jobs.pop(cid, None)

                # --- AGGREGATION LOGIC ---
                if self.algorithm == "FedAsync":
                    weight = compute_fedasync_weight(staleness, self.alpha_0)
                    server_version_before = self.server_version

                    if self.enable_analysis:
                        val_loss_before = evaluate_model_loss(self.server_model, self.val_loader)
                        server_params_before = copy.deepcopy(self.server_params)
                        base_params_used = job.get("base_params", server_params_before)
                        pred_dict = compute_all_predictors(
                            client_params=trained_params,
                            base_params=base_params_used,
                            server_params=server_params_before,
                            prev_server_delta=self.prev_server_delta,
                            client_train_version=client_training_version,
                            server_version=server_version_before,
                            local_loss=local_loss,
                        )

                    self.server_params = fedasync_aggregate_parameters(
                        self.server_params, trained_params, weight
                    )
                    set_model_parameters(self.server_model, self.server_params)

                    self.server_version += 1
                    self.applied_updates += 1
                    self.server_aggregation_count += 1
                    self.per_client_applied[cid] += 1
                    self.per_tier_applied[profile.speed_tier] += 1

                    val_m = evaluate(self.server_model, self.val_loader, self.criterion)
                    test_m = evaluate(self.server_model, self.test_loader, self.criterion)

                    if self.enable_analysis:
                        val_loss_after = evaluate_model_loss(self.server_model, self.val_loader)
                        impact_res = compute_action_impact(
                            val_loss_before=val_loss_before,
                            val_loss_after=val_loss_after,
                            tolerance=self.impact_tolerance,
                        )
                        self.prev_server_delta = flatten_params(self.server_params) - flatten_params(server_params_before)
                        self.action_observations.append({
                            "action_id": self.server_aggregation_count,
                            "scenario": self.scenario_config.scenario_name,
                            "algorithm": self.algorithm,
                            "seed": self.scenario_config.data_seed,
                            "virtual_time": round(float(event_time), 4),
                            "client_id": cid,
                            "staleness": pred_dict["staleness"],
                            "update_l2_norm": pred_dict["update_l2_norm"],
                            "relative_update_norm": pred_dict["relative_update_norm"],
                            "current_model_divergence": pred_dict["current_model_divergence"],
                            "cosine_disagreement": pred_dict["cosine_disagreement"],
                            "cosine_available": pred_dict["cosine_available"],
                            "local_loss": pred_dict["local_loss"],
                            "val_loss_before": impact_res["val_loss_before"],
                            "val_loss_after": impact_res["val_loss_after"],
                            "impact": impact_res["impact"],
                            "is_harmful": impact_res["is_harmful"],
                        })

                    agg_record = {
                        "aggregation_index": self.server_aggregation_count,
                        "time": round(float(event_time), 4),
                        "server_version_before": server_version_before,
                        "server_version_after": self.server_version,
                        "client_id": cid,
                        "job_id": job_id,
                        "staleness": staleness,
                        "aggregation_weight": round(float(weight), 6),
                        "val_loss": round(float(val_m["loss"]), 6),
                        "val_accuracy": round(float(val_m["accuracy"]), 6),
                        "test_loss": round(float(test_m["loss"]), 6),
                        "test_accuracy": round(float(test_m["accuracy"]), 6),
                    }
                    self.aggregation_log.append(agg_record)
                    self.history.append(agg_record)

                    self.event_log.append({
                        "time": round(float(event_time), 4),
                        "event_type": "AGGREGATE",
                        "client_id": cid,
                        "job_id": job_id,
                        "staleness": staleness,
                        "aggregation_weight": round(float(weight), 6),
                        "server_version_after": self.server_version,
                        "applied_updates": self.applied_updates,
                    })

                    # Redispatch client
                    if self.applied_updates < self.scenario_config.target_applied_updates:
                        self._dispatch_client(client_id=cid, dispatch_time=event_time)

                elif self.algorithm == "FedBuff":
                    buffered_item = {
                        "client_id": cid,
                        "job_id": job_id,
                        "trained_params": trained_params,
                        "sample_count": sample_count,
                        "staleness_at_arrival": staleness,
                        "client_training_version": client_training_version,
                        "speed_tier": profile.speed_tier,
                    }
                    if self.enable_analysis:
                        base_params_used = job.get("base_params", self.server_params)
                        arr_preds = compute_all_predictors(
                            client_params=trained_params,
                            base_params=base_params_used,
                            server_params=self.server_params,
                            prev_server_delta=self.prev_server_delta,
                            client_train_version=client_training_version,
                            server_version=server_version_at_arrival,
                            local_loss=local_loss,
                        )
                        buffered_item["arrival_predictors"] = arr_preds
                        buffered_item["base_params"] = base_params_used
                        buffered_item["local_loss"] = local_loss
                    self.buffer.append(buffered_item)

                    self.event_log.append({
                        "time": round(float(event_time), 4),
                        "event_type": "UPDATE_BUFFERED",
                        "client_id": cid,
                        "job_id": job_id,
                        "buffer_size_after": len(self.buffer),
                    })

                    # Check if buffer reached K
                    if len(self.buffer) >= self.buffer_size:
                        self._aggregate_fedbuff_buffer(is_final_flush=False, event_time=event_time)

                    # Redispatch client
                    if self.applied_updates < self.scenario_config.target_applied_updates:
                        self._dispatch_client(client_id=cid, dispatch_time=event_time)

        # FedBuff partial buffer policy at simulation termination
        unflushed_buffer_count = len(self.buffer)
        final_buffer_flushed = False

        if self.algorithm == "FedBuff" and len(self.buffer) > 0:
            if self.flush_final_buffer:
                final_buffer_flushed = True
                self._aggregate_fedbuff_buffer(is_final_flush=True, event_time=self.current_time)
                unflushed_buffer_count = 0
            else:
                self.event_log.append({
                    "time": round(float(self.current_time), 4),
                    "event_type": "FINAL_BUFFER_DISCARDED",
                    "unflushed_buffer_count": len(self.buffer),
                    "client_ids": [u["client_id"] for u in self.buffer],
                })

        # Final evaluation
        final_test = evaluate(self.server_model, self.test_loader, self.criterion)
        final_val = evaluate(self.server_model, self.val_loader, self.criterion)
        final_test_loss = float(final_test["loss"])
        final_test_acc = float(final_test["accuracy"])
        final_val_loss = float(final_val["loss"])
        final_val_acc = float(final_val["accuracy"])

        staleness_vals = [u["staleness"] for u in self.arrival_log]
        mean_stale = float(np.mean(staleness_vals)) if staleness_vals else 0.0
        max_stale = int(np.max(staleness_vals)) if staleness_vals else 0
        min_stale = int(np.min(staleness_vals)) if staleness_vals else 0

        pending_jobs = len([j for j in self.active_jobs.values() if j["status"] in {"DOWNLOADING", "TRAINING", "UPLOADING", "RETAINED"}])

        # Phase 15 Fairness and Predictor Analysis compilation
        analysis_data: Optional[Dict[str, Any]] = None
        if self.enable_analysis:
            all_client_ids = list(range(self.scenario_config.n_clients))
            client_tiers = {p.client_id: p.speed_tier for p in self.profiles}
            client_label_counts = {}
            for cid in all_client_ids:
                part_id = self.profiles[cid].data_partition_id
                counts_list = self.partition_stats[part_id]["class_counts"]
                client_label_counts[cid] = {c: count for c, count in enumerate(counts_list)}
            held_out_metrics = evaluate_per_class_held_out(
                model=self.server_model,
                data_loader=self.val_loader,
                num_classes=self.dataset_config.n_classes,
            )
            fairness_summary = compile_fairness_summary(
                all_client_ids=all_client_ids,
                client_arrivals=self.per_client_arrivals,
                client_applied=self.per_client_applied,
                client_samples_uploaded=self.client_samples_uploaded,
                client_tiers=client_tiers,
                client_label_counts=client_label_counts,
                held_out_metrics=held_out_metrics,
            )
            stat_analysis = analyze_action_observations(self.action_observations)
            analysis_data = {
                "action_observations": self.action_observations,
                "action_count": len(self.action_observations),
                "statistical_evaluation": stat_analysis,
                "fairness_and_representation": fairness_summary,
            }

        results: Dict[str, Any] = {
            "config": {
                "scenario_name": self.scenario_config.scenario_name,
                "description": self.scenario_config.description,
                "algorithm": self.algorithm,
                "n_clients": self.scenario_config.n_clients,
                "target_applied_updates": self.scenario_config.target_applied_updates,
                "max_attempted_jobs": self.scenario_config.max_attempted_jobs,
                "max_virtual_time": self.scenario_config.max_virtual_time,
                "rejoin_policy": self.scenario_config.rejoin_policy,
                "buffer_size": self.buffer_size if self.algorithm == "FedBuff" else 1,
                "flush_final_buffer": self.flush_final_buffer,
                "alpha_0": self.alpha_0,
                "local_epochs": self.local_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "partition_method": self.scenario_config.partition_method,
                "alpha": self.scenario_config.alpha,
                "coupling_enabled": self.scenario_config.coupling_enabled,
                "data_seed": self.scenario_config.data_seed,
                "profile_seed": self.scenario_config.profile_seed,
                "network_seed": self.scenario_config.network_seed,
                "dropout_seed": self.scenario_config.dropout_seed,
            },
            "git_commit": get_git_commit(),
            "client_profiles": [p.to_dict() for p in self.profiles],
            "client_stats": self.partition_stats,
            "coupling_audit": self.coupling_audit,
            "network_schedule": [asdict(ev) for ev in self.scenario_config.network_events],
            "outage_schedule": [asdict(ev) for ev in self.scenario_config.outage_events],
            "initial": self.initial_metrics,
            "event_log": self.event_log,
            "arrival_log": self.arrival_log,
            "aggregation_log": self.aggregation_log,
            "history": self.history,
            "test": {
                "loss": round(final_test_loss, 6),
                "accuracy": round(final_test_acc, 6),
            },
            "validation": {
                "loss": round(final_val_loss, 6),
                "accuracy": round(final_val_acc, 6),
            },
            "summary": {
                "scenario_name": self.scenario_config.scenario_name,
                "algorithm": self.algorithm,
                "stopping_reason": stopping_reason,
                "target_applied_updates": self.scenario_config.target_applied_updates,
                "jobs_dispatched": self.jobs_dispatched,
                "training_starts": self.training_starts,
                "training_completions": self.training_completions,
                "failed_jobs": self.failed_jobs,
                "failures_by_stage": dict(self.failures_by_stage),
                "retained_updates_count": self.retained_updates_count,
                "pending_jobs_at_termination": pending_jobs,
                "arrival_count": self.arrival_count,
                "applied_updates": self.applied_updates,
                "server_aggregation_count": self.server_aggregation_count,
                "final_server_version": self.server_version,
                "unflushed_buffer_count": unflushed_buffer_count,
                "final_buffer_flushed": final_buffer_flushed,
                "mean_staleness": round(mean_stale, 4),
                "min_staleness": min_stale,
                "max_staleness": max_stale,
                "total_virtual_time_s": round(float(self.current_time), 4),
                "total_bytes_transferred": self.total_bytes_transferred,
                "per_client_arrivals": {str(cid): count for cid, count in sorted(self.per_client_arrivals.items())},
                "per_client_applied": {str(cid): count for cid, count in sorted(self.per_client_applied.items())},
                "per_tier_applied": dict(self.per_tier_applied),
                "final_validation_accuracy": round(final_val_acc, 6),
                "final_validation_loss": round(final_val_loss, 6),
                "final_test_accuracy": round(final_test_acc, 6),
                "final_test_loss": round(final_test_loss, 6),
            },
            # Top-level convenience fields
            "stopping_reason": stopping_reason,
            "jobs_dispatched": self.jobs_dispatched,
            "arrival_count": self.arrival_count,
            "applied_updates": self.applied_updates,
            "server_aggregation_count": self.server_aggregation_count,
            "final_server_version": self.server_version,
            "total_virtual_time_s": round(float(self.current_time), 4),
            "total_bytes_transferred": self.total_bytes_transferred,
            "final_test_accuracy": round(final_test_acc, 6),
            "final_test_loss": round(final_test_loss, 6),
            "final_validation_accuracy": round(final_val_acc, 6),
            "final_validation_loss": round(final_val_loss, 6),
            "mean_staleness": round(mean_stale, 4),
            "max_staleness": max_stale,
        }

        if self.enable_analysis and analysis_data is not None:
            results["analysis"] = analysis_data

        return results

    def _aggregate_fedbuff_buffer(self, is_final_flush: bool, event_time: float) -> None:
        """Aggregate buffered client models in FedBuff using sample-count-weighted averaging."""
        if not self.buffer:
            return

        items = list(self.buffer)
        total_samples = sum(u["sample_count"] for u in items)
        server_version_before = self.server_version

        if self.enable_analysis:
            val_loss_before = evaluate_model_loss(self.server_model, self.val_loader)
            server_params_before = copy.deepcopy(self.server_params)
            member_app_preds = []
            member_deltas = []
            member_weights = []
            for u in items:
                m_base = u.get("base_params", server_params_before)
                m_pred = compute_all_predictors(
                    client_params=u["trained_params"],
                    base_params=m_base,
                    server_params=server_params_before,
                    prev_server_delta=self.prev_server_delta,
                    client_train_version=u["client_training_version"],
                    server_version=server_version_before,
                    local_loss=u.get("local_loss"),
                )
                member_app_preds.append(m_pred)
                w = float(u["sample_count"] / total_samples) if total_samples > 0 else 0.0
                member_weights.append(w)
                delta_u = flatten_params(u["trained_params"]) - flatten_params(m_base)
                member_deltas.append(w * delta_u)
            joint_delta = torch.stack(member_deltas).sum(dim=0)
            buf_preds = aggregate_buffer_predictors(
                member_predictors=member_app_preds,
                joint_delta=joint_delta,
                prev_server_delta=self.prev_server_delta,
            )

        self.server_params = aggregate_buffered_parameters(items)
        set_model_parameters(self.server_model, self.server_params)

        self.server_version += 1
        self.applied_updates += len(items)
        self.server_aggregation_count += 1

        for u in items:
            self.per_client_applied[u["client_id"]] += 1
            self.per_tier_applied[u["speed_tier"]] += 1

        # Clear buffer immediately
        self.buffer = []

        val_m = evaluate(self.server_model, self.val_loader, self.criterion)
        test_m = evaluate(self.server_model, self.test_loader, self.criterion)

        if self.enable_analysis:
            val_loss_after = evaluate_model_loss(self.server_model, self.val_loader)
            impact_res = compute_action_impact(
                val_loss_before=val_loss_before,
                val_loss_after=val_loss_after,
                tolerance=self.impact_tolerance,
            )
            self.prev_server_delta = flatten_params(self.server_params) - flatten_params(server_params_before)
            self.action_observations.append({
                "action_id": self.server_aggregation_count,
                "scenario": self.scenario_config.scenario_name,
                "algorithm": self.algorithm,
                "seed": self.scenario_config.data_seed,
                "virtual_time": round(float(event_time), 4),
                "client_id": "buffer_" + "_".join(str(u["client_id"]) for u in items),
                "client_ids": [u["client_id"] for u in items],
                "buffer_size": len(items),
                "member_weights": [round(float(w), 6) for w in member_weights],
                "staleness": buf_preds["mean_staleness"],
                "max_staleness": buf_preds["max_staleness"],
                "update_l2_norm": buf_preds["mean_update_l2_norm"],
                "relative_update_norm": buf_preds["mean_relative_norm"],
                "current_model_divergence": buf_preds["mean_divergence"],
                "cosine_disagreement": buf_preds["buffer_cosine_disagreement"],
                "cosine_available": buf_preds["buffer_cosine_available"],
                "local_loss": round(float(np.mean([u.get("local_loss", 0.0) for u in items])), 6),
                "val_loss_before": impact_res["val_loss_before"],
                "val_loss_after": impact_res["val_loss_after"],
                "impact": impact_res["impact"],
                "is_harmful": impact_res["is_harmful"],
                "member_arrival_predictors": [u.get("arrival_predictors") for u in items],
                "member_application_predictors": member_app_preds,
            })

        # Log staleness at both arrival and application
        application_staleness = [int(server_version_before - u["client_training_version"]) for u in items]

        agg_record = {
            "aggregation_index": self.server_aggregation_count,
            "time": round(float(event_time), 4),
            "server_version_before": server_version_before,
            "server_version_after": self.server_version,
            "buffer_size": len(items),
            "client_ids": [u["client_id"] for u in items],
            "staleness_at_arrival": [u["staleness_at_arrival"] for u in items],
            "staleness_at_application": application_staleness,
            "mean_application_staleness": round(float(np.mean(application_staleness)), 4),
            "is_final_flush": is_final_flush,
            "val_loss": round(float(val_m["loss"]), 6),
            "val_accuracy": round(float(val_m["accuracy"]), 6),
            "test_loss": round(float(test_m["loss"]), 6),
            "test_accuracy": round(float(test_m["accuracy"]), 6),
        }
        self.aggregation_log.append(agg_record)
        self.history.append(agg_record)

        self.event_log.append({
            "time": round(float(event_time), 4),
            "event_type": "FINAL_BUFFER_FLUSHED" if is_final_flush else "AGGREGATE",
            "aggregation_index": self.server_aggregation_count,
            "server_version_after": self.server_version,
            "buffer_size": len(items),
            "applied_updates": self.applied_updates,
        })


def run_scenario_simulation(
    scenario_config: ScenarioConfig,
    algorithm: str = "FedAsync",
    buffer_size: int = 5,
    flush_final_buffer: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    enable_analysis: bool = False,
    impact_tolerance: float = 1e-5,
    export_actions_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute scenario simulation and optionally save output JSON and action CSV."""
    sim = ScenarioSimulator(
        scenario_config=scenario_config,
        algorithm=algorithm,
        buffer_size=buffer_size,
        flush_final_buffer=flush_final_buffer,
        enable_analysis=enable_analysis,
        impact_tolerance=impact_tolerance,
    )
    results = sim.run_simulation()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Auto-export matching action CSV if analysis enabled and not explicitly specified
        if enable_analysis and export_actions_csv is None and "analysis" in results:
            csv_path = p.parent / f"{p.stem}_actions.csv"
            export_actions_to_csv(results["analysis"]["action_observations"], csv_path)

    if export_actions_csv is not None and "analysis" in results:
        export_actions_to_csv(results["analysis"]["action_observations"], export_actions_csv)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="AFL Dynamic Scenario Engine")
    parser.add_argument("--scenario", type=str, default="S1", help="Named scenario (S1..S5) or config path")
    parser.add_argument("--algorithm", type=str, default="FedAsync", choices=["FedAsync", "FedBuff"], help="AFL Algorithm")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--updates", type=int, default=15, help="Target applied updates")
    parser.add_argument("--buffer-size", type=int, default=5, help="Buffer size K for FedBuff")
    parser.add_argument("--flush-final-buffer", action="store_true", default=False, help="Flush incomplete final buffer in FedBuff")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--analysis", action="store_true", default=False, help="Enable Phase 15 divergence and impact analysis")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Harmful impact loss tolerance")
    parser.add_argument("--export-actions-csv", type=str, default=None, help="Path to export action-level CSV")
    parser.add_argument("--max-attempted-jobs", type=int, default=None, help="Explicit attempted-job cap")
    parser.add_argument("--max-virtual-time", type=float, default=None, help="Explicit virtual-time cap")

    args = parser.parse_args()

    sc_name = args.scenario.upper()
    if sc_name.startswith("S"):
        cfg = get_named_scenario(sc_name, n_clients=args.clients, target_updates=args.updates)
    else:
        # Load from file if JSON
        p = Path(args.scenario)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = ScenarioConfig(**data)
        else:
            cfg = get_named_scenario("S1", n_clients=args.clients, target_updates=args.updates)

    cfg.data_seed = args.seed
    cfg.profile_seed = args.seed
    cfg.network_seed = args.seed
    cfg.dropout_seed = args.seed
    if args.max_attempted_jobs is not None:
        cfg.max_attempted_jobs = args.max_attempted_jobs
    if args.max_virtual_time is not None:
        cfg.max_virtual_time = args.max_virtual_time

    results = run_scenario_simulation(
        scenario_config=cfg,
        algorithm=args.algorithm,
        buffer_size=args.buffer_size,
        flush_final_buffer=args.flush_final_buffer,
        output_path=args.output,
        enable_analysis=args.analysis,
        impact_tolerance=args.tolerance,
        export_actions_csv=args.export_actions_csv,
    )

    print(
        f"[{results['config']['algorithm']} on {results['config']['scenario_name']}] "
        f"Reason: {results['stopping_reason']} | "
        f"Applied: {results['applied_updates']}/{cfg.target_applied_updates} | "
        f"Virtual Time: {results['total_virtual_time_s']:.2f}s | "
        f"Bytes: {results['total_bytes_transferred']} | "
        f"Test Acc: {results['final_test_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
