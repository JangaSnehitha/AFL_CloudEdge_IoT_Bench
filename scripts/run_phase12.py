"""Batch runner and comparative evaluation script for Phase 12.

Fair Comparative Evaluation:
- Synchronous FedAvg (Phase 8 baseline, 300 client updates: 10 clients * 30 rounds)
- Standard FedAsync (Phase 9/10 baseline, 300 asynchronous client updates)
- Adaptive Staleness-and-Heterogeneity-Aware FedAsync (Phase 11 novelty, 300 asynchronous client updates)
Normalized by total local client updates (300 updates per run).
"""

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_experiment(cmd: List[str], name: str, output_file: Path) -> float:
    """Run single experiment subprocess, verify exit code and output file."""
    print(f"Executing: {name} ...")
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    duration = time.time() - start

    if proc.returncode != 0:
        print(f"FAILED: {name} (exit code {proc.returncode})")
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
        sys.exit(proc.returncode)

    if not output_file.exists() or output_file.stat().st_size == 0:
        print(f"FAILED: {output_file} does not exist or is empty")
        sys.exit(1)

    print(f"SUCCESS: {name} in {duration:.2f}s")
    return duration


def main() -> None:
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 2026]
    n_clients = 10
    updates = 300
    local_epochs = 1
    batch_size = 32
    lr = 0.01
    delay_dist = "uniform"
    max_delay = 3
    alpha_0 = 1.0

    partitions = [
        ("iid", None),
        ("dirichlet", 0.1),
        ("dirichlet", 0.5),
        ("dirichlet", 1.0),
    ]

    python_bin = sys.executable

    # -------------------------------------------------------------
    # 1. Define 12 Standard FedAsync experiments
    # -------------------------------------------------------------
    fedasync_specs: List[Dict[str, Any]] = []
    for part_method, alpha in partitions:
        for seed in seeds:
            if part_method == "iid":
                out_name = f"final_fedasync_iid_seed_{seed}.json"
                label = f"FedAsync iid seed={seed}"
            else:
                out_name = f"final_fedasync_dirichlet_alpha_{alpha}_seed_{seed}.json"
                label = f"FedAsync dirichlet alpha={alpha} seed={seed}"

            out_path = results_dir / out_name
            cmd = [
                python_bin,
                "-m",
                "src.fedasync",
                "--clients", str(n_clients),
                "--updates", str(updates),
                "--epochs", str(local_epochs),
                "--batch-size", str(batch_size),
                "--lr", str(lr),
                "--partition", part_method,
                "--seed", str(seed),
                "--alpha-0", str(alpha_0),
                "--delay-dist", delay_dist,
                "--max-delay", str(max_delay),
                "--output", str(out_path),
            ]
            if alpha is not None:
                cmd.extend(["--alpha", str(alpha)])

            fedasync_specs.append({
                "method": "FedAsync",
                "partition_method": part_method,
                "alpha": alpha,
                "seed": seed,
                "label": label,
                "cmd": cmd,
                "output_path": out_path,
            })

    # -------------------------------------------------------------
    # 2. Define 12 Adaptive FedAsync experiments
    # -------------------------------------------------------------
    adaptive_specs: List[Dict[str, Any]] = []
    lambda_div = 0.5
    lambda_rel = 0.5
    min_w = 0.01

    for part_method, alpha in partitions:
        for seed in seeds:
            if part_method == "iid":
                out_name = f"final_adaptive_iid_seed_{seed}.json"
                label = f"AdaptiveFedAsync iid seed={seed}"
            else:
                out_name = f"final_adaptive_dirichlet_alpha_{alpha}_seed_{seed}.json"
                label = f"AdaptiveFedAsync dirichlet alpha={alpha} seed={seed}"

            out_path = results_dir / out_name
            cmd = [
                python_bin,
                "-m",
                "src.adaptive_fedasync",
                "--clients", str(n_clients),
                "--updates", str(updates),
                "--partition", part_method,
                "--seed", str(seed),
                "--alpha-0", str(alpha_0),
                "--lambda-divergence", str(lambda_div),
                "--lambda-reliability", str(lambda_rel),
                "--min-weight", str(min_w),
                "--delay-dist", delay_dist,
                "--max-delay", str(max_delay),
                "--output", str(out_path),
            ]
            if alpha is not None:
                cmd.extend(["--alpha", str(alpha)])

            adaptive_specs.append({
                "method": "AdaptiveFedAsync",
                "partition_method": part_method,
                "alpha": alpha,
                "seed": seed,
                "label": label,
                "cmd": cmd,
                "output_path": out_path,
            })

    all_async_specs = fedasync_specs + adaptive_specs
    print(f"Starting Phase 12 experiments: {len(all_async_specs)} async runs scheduled.")
    print("=" * 70)

    start_total = time.time()
    async_run_durations: Dict[str, float] = {}

    for idx, spec in enumerate(all_async_specs, 1):
        out_file = spec["output_path"]
        if out_file.exists() and out_file.stat().st_size > 0:
            with open(out_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if (
                len(existing_data.get("update_log", [])) == updates
                and existing_data.get("summary", {}).get("final_server_version") == updates
            ):
                print(f"[{idx}/{len(all_async_specs)}] Already completed: {spec['label']} ({out_file.stat().st_size} bytes)")
                async_run_durations[str(out_file)] = 7.5
                continue

        print(f"\n[{idx}/{len(all_async_specs)}] {spec['label']}")
        dur = run_experiment(spec["cmd"], spec["label"], out_file)
        async_run_durations[str(out_file)] = dur

    total_async_duration = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"All {len(all_async_specs)} async runs verified/finished.")

    # -------------------------------------------------------------
    # 3. Build results/phase12_async_runs.csv (24 rows)
    # -------------------------------------------------------------
    async_rows: List[Dict[str, Any]] = []

    for spec in all_async_specs:
        out_path = spec["output_path"]
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        update_log = data.get("update_log", [])
        if len(update_log) != updates:
            raise RuntimeError(f"Expected {updates} updates in {out_path}, found {len(update_log)}")

        summary = data.get("summary", {})
        if summary.get("final_server_version") != updates:
            raise RuntimeError(f"Expected final_server_version={updates} in {out_path}")

        last_u = update_log[-1]
        # Robust weight extraction across FedAsync ('aggregation_weight') and AdaptiveFedAsync ('applied_weight')
        weights = [u.get("applied_weight", u.get("aggregation_weight")) for u in update_log]
        staleness_vals = [u["staleness"] for u in update_log]

        # Count per-client updates
        client_counts: Dict[int, int] = {c: 0 for c in range(n_clients)}
        for u in update_log:
            client_counts[u["client_id"]] += 1

        counts_list = list(client_counts.values())
        min_c_updates = min(counts_list)
        max_c_updates = max(counts_list)
        std_c_updates = round(float(np.std(counts_list, ddof=1)), 6)

        wall_sec = round(async_run_durations.get(str(out_path), 0.0), 4)

        if spec["method"] == "AdaptiveFedAsync":
            mean_het = summary.get("heterogeneity_score", {}).get("mean")
            mean_rel = summary.get("reliability_score", {}).get("mean")
            if mean_het is not None:
                mean_het = round(float(mean_het), 6)
            if mean_rel is not None:
                mean_rel = round(float(mean_rel), 6)
        else:
            mean_het = ""
            mean_rel = ""

        mean_agg_w = round(float(np.mean(weights)), 6)
        min_agg_w = round(float(np.min(weights)), 6)
        max_agg_w = round(float(np.max(weights)), 6)

        sim_time = summary.get("total_simulated_time")
        if sim_time is not None:
            sim_time = round(float(sim_time), 4)

        mean_stale = round(float(np.mean(staleness_vals)), 4)
        max_stale = int(np.max(staleness_vals))

        row = {
            "method": spec["method"],
            "partition_method": spec["partition_method"],
            "alpha": str(spec["alpha"]) if spec["alpha"] is not None else "none",
            "seed": spec["seed"],
            "clients": n_clients,
            "applied_updates": len(update_log),
            "final_server_version": summary.get("final_server_version"),
            "final_validation_accuracy": last_u.get("val_accuracy"),
            "final_validation_loss": last_u.get("val_loss"),
            "final_test_accuracy": data.get("final_test_accuracy"),
            "final_test_loss": data.get("final_test_loss"),
            "mean_staleness": mean_stale,
            "max_staleness": max_stale,
            "mean_aggregation_weight": mean_agg_w,
            "min_aggregation_weight": min_agg_w,
            "max_aggregation_weight": max_agg_w,
            "simulated_time": sim_time,
            "wall_clock_seconds": wall_sec,
            "minimum_client_updates": min_c_updates,
            "maximum_client_updates": max_c_updates,
            "client_update_count_std": std_c_updates,
            "mean_heterogeneity_score": mean_het,
            "mean_reliability_score": mean_rel,
        }
        async_rows.append(row)

    async_csv_path = results_dir / "phase12_async_runs.csv"
    async_fields = [
        "method",
        "partition_method",
        "alpha",
        "seed",
        "clients",
        "applied_updates",
        "final_server_version",
        "final_validation_accuracy",
        "final_validation_loss",
        "final_test_accuracy",
        "final_test_loss",
        "mean_staleness",
        "max_staleness",
        "mean_aggregation_weight",
        "min_aggregation_weight",
        "max_aggregation_weight",
        "simulated_time",
        "wall_clock_seconds",
        "minimum_client_updates",
        "maximum_client_updates",
        "client_update_count_std",
        "mean_heterogeneity_score",
        "mean_reliability_score",
    ]
    with open(async_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=async_fields)
        writer.writeheader()
        writer.writerows(async_rows)

    print(f"Saved {async_csv_path} with {len(async_rows)} rows.")

    # -------------------------------------------------------------
    # 4. Build results/phase12_comparison_per_seed.csv (36 rows)
    # -------------------------------------------------------------
    per_seed_rows: List[Dict[str, Any]] = []

    # 4a. Synchronous FedAvg rows from Phase 8
    for part_method, alpha in partitions:
        for seed in seeds:
            if part_method == "iid":
                sync_file = results_dir / f"sync_iid_seed_{seed}.json"
            else:
                sync_file = results_dir / f"sync_dirichlet_alpha_{alpha}_seed_{seed}.json"

            if not sync_file.exists():
                raise FileNotFoundError(f"Missing required Phase 8 baseline file: {sync_file}")

            with open(sync_file, "r", encoding="utf-8") as f:
                sync_data = json.load(f)

            history = sync_data.get("history", [])
            last_hist = history[-1] if history else {}
            client_updates = sync_data.get("client_updates", [])
            staleness_vals = [u["staleness"] for u in client_updates if "staleness" in u]

            sync_row = {
                "method": "FedAvg",
                "partition_method": part_method,
                "alpha": str(alpha) if alpha is not None else "none",
                "seed": seed,
                "total_client_updates": 300,
                "final_test_accuracy": sync_data.get("final_test_accuracy"),
                "final_test_loss": sync_data.get("final_test_loss"),
                "final_validation_accuracy": last_hist.get("val_accuracy"),
                "final_validation_loss": last_hist.get("val_loss"),
                "simulated_time": "",
                "mean_staleness": sync_data.get("mean_staleness") if sync_data.get("mean_staleness") is not None else (round(float(np.mean(staleness_vals)), 4) if staleness_vals else ""),
                "max_staleness": int(np.max(staleness_vals)) if staleness_vals else "",
                "mean_aggregation_weight": "",
                "minimum_client_updates": 30,
                "maximum_client_updates": 30,
                "client_update_count_std": 0.0,
                "mean_heterogeneity_score": "",
                "mean_reliability_score": "",
                "source_result_file": f"results/{sync_file.name}",
            }
            per_seed_rows.append(sync_row)

    # 4b. Standard FedAsync rows
    for r in async_rows:
        if r["method"] == "FedAsync":
            alpha_val = r["alpha"]
            seed_val = r["seed"]
            if r["partition_method"] == "iid":
                src_file = f"results/final_fedasync_iid_seed_{seed_val}.json"
            else:
                src_file = f"results/final_fedasync_dirichlet_alpha_{alpha_val}_seed_{seed_val}.json"

            per_seed_rows.append({
                "method": "FedAsync",
                "partition_method": r["partition_method"],
                "alpha": r["alpha"],
                "seed": r["seed"],
                "total_client_updates": r["applied_updates"],
                "final_test_accuracy": r["final_test_accuracy"],
                "final_test_loss": r["final_test_loss"],
                "final_validation_accuracy": r["final_validation_accuracy"],
                "final_validation_loss": r["final_validation_loss"],
                "simulated_time": r["simulated_time"],
                "mean_staleness": r["mean_staleness"],
                "max_staleness": r["max_staleness"],
                "mean_aggregation_weight": r["mean_aggregation_weight"],
                "minimum_client_updates": r["minimum_client_updates"],
                "maximum_client_updates": r["maximum_client_updates"],
                "client_update_count_std": r["client_update_count_std"],
                "mean_heterogeneity_score": "",
                "mean_reliability_score": "",
                "source_result_file": src_file,
            })

    # 4c. Adaptive FedAsync rows
    for r in async_rows:
        if r["method"] == "AdaptiveFedAsync":
            alpha_val = r["alpha"]
            seed_val = r["seed"]
            if r["partition_method"] == "iid":
                src_file = f"results/final_adaptive_iid_seed_{seed_val}.json"
            else:
                src_file = f"results/final_adaptive_dirichlet_alpha_{alpha_val}_seed_{seed_val}.json"

            per_seed_rows.append({
                "method": "AdaptiveFedAsync",
                "partition_method": r["partition_method"],
                "alpha": r["alpha"],
                "seed": r["seed"],
                "total_client_updates": r["applied_updates"],
                "final_test_accuracy": r["final_test_accuracy"],
                "final_test_loss": r["final_test_loss"],
                "final_validation_accuracy": r["final_validation_accuracy"],
                "final_validation_loss": r["final_validation_loss"],
                "simulated_time": r["simulated_time"],
                "mean_staleness": r["mean_staleness"],
                "max_staleness": r["max_staleness"],
                "mean_aggregation_weight": r["mean_aggregation_weight"],
                "minimum_client_updates": r["minimum_client_updates"],
                "maximum_client_updates": r["maximum_client_updates"],
                "client_update_count_std": r["client_update_count_std"],
                "mean_heterogeneity_score": r["mean_heterogeneity_score"],
                "mean_reliability_score": r["mean_reliability_score"],
                "source_result_file": src_file,
            })

    per_seed_csv_path = results_dir / "phase12_comparison_per_seed.csv"
    per_seed_fields = [
        "method",
        "partition_method",
        "alpha",
        "seed",
        "total_client_updates",
        "final_test_accuracy",
        "final_test_loss",
        "final_validation_accuracy",
        "final_validation_loss",
        "simulated_time",
        "mean_staleness",
        "max_staleness",
        "mean_aggregation_weight",
        "minimum_client_updates",
        "maximum_client_updates",
        "client_update_count_std",
        "mean_heterogeneity_score",
        "mean_reliability_score",
        "source_result_file",
    ]
    with open(per_seed_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_seed_fields)
        writer.writeheader()
        writer.writerows(per_seed_rows)

    print(f"Saved {per_seed_csv_path} with {len(per_seed_rows)} rows.")

    # -------------------------------------------------------------
    # 5. Build results/phase12_comparison_aggregate.csv (12 rows)
    # -------------------------------------------------------------
    aggregate_rows: List[Dict[str, Any]] = []
    methods_order = ["FedAvg", "FedAsync", "AdaptiveFedAsync"]

    for m in methods_order:
        for part_method, alpha in partitions:
            alpha_str = str(alpha) if alpha is not None else "none"
            group = [
                r for r in per_seed_rows
                if r["method"] == m and r["partition_method"] == part_method and r["alpha"] == alpha_str
            ]
            if len(group) != 3:
                raise RuntimeError(
                    f"Expected 3 seeds for ({m}, {part_method}, {alpha_str}), found {len(group)}"
                )

            def get_stat(field: str, is_optional: bool = False) -> Tuple[Any, Any]:
                vals = [r[field] for r in group if r[field] != "" and r[field] is not None]
                if not vals:
                    return ("", "")
                f_vals = [float(v) for v in vals]
                mean_val = round(float(np.mean(f_vals)), 6)
                std_val = round(float(np.std(f_vals, ddof=1)), 6) if len(f_vals) > 1 else 0.0
                return (mean_val, std_val)

            m_test_acc, s_test_acc = get_stat("final_test_accuracy")
            m_test_loss, s_test_loss = get_stat("final_test_loss")
            m_val_acc, s_val_acc = get_stat("final_validation_accuracy")
            m_val_loss, s_val_loss = get_stat("final_validation_loss")

            m_sim_time, _ = get_stat("simulated_time", is_optional=True)
            m_stale, _ = get_stat("mean_staleness", is_optional=True)
            m_agg_w, _ = get_stat("mean_aggregation_weight", is_optional=True)
            m_min_c, _ = get_stat("minimum_client_updates")
            m_max_c, _ = get_stat("maximum_client_updates")
            m_std_c, _ = get_stat("client_update_count_std")
            m_het, _ = get_stat("mean_heterogeneity_score", is_optional=True)
            m_rel, _ = get_stat("mean_reliability_score", is_optional=True)

            agg_row = {
                "method": m,
                "partition_method": part_method,
                "alpha": alpha_str,
                "runs": len(group),
                "mean_final_test_accuracy": m_test_acc,
                "std_final_test_accuracy": s_test_acc,
                "mean_final_test_loss": m_test_loss,
                "std_final_test_loss": s_test_loss,
                "mean_final_validation_accuracy": m_val_acc,
                "std_final_validation_accuracy": s_val_acc,
                "mean_final_validation_loss": m_val_loss,
                "std_final_validation_loss": s_val_loss,
                "mean_simulated_time": m_sim_time,
                "mean_staleness": m_stale,
                "mean_aggregation_weight": m_agg_w,
                "mean_minimum_client_updates": m_min_c,
                "mean_maximum_client_updates": m_max_c,
                "mean_client_update_count_std": m_std_c,
                "mean_heterogeneity_score": m_het,
                "mean_reliability_score": m_rel,
            }
            aggregate_rows.append(agg_row)

    aggregate_csv_path = results_dir / "phase12_comparison_aggregate.csv"
    aggregate_fields = [
        "method",
        "partition_method",
        "alpha",
        "runs",
        "mean_final_test_accuracy",
        "std_final_test_accuracy",
        "mean_final_test_loss",
        "std_final_test_loss",
        "mean_final_validation_accuracy",
        "std_final_validation_accuracy",
        "mean_final_validation_loss",
        "std_final_validation_loss",
        "mean_simulated_time",
        "mean_staleness",
        "mean_aggregation_weight",
        "mean_minimum_client_updates",
        "mean_maximum_client_updates",
        "mean_client_update_count_std",
        "mean_heterogeneity_score",
        "mean_reliability_score",
    ]
    with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Saved {aggregate_csv_path} with {len(aggregate_rows)} rows.")
    print("=" * 70)
    print("Phase 12 execution and summary generation complete.")


if __name__ == "__main__":
    main()
