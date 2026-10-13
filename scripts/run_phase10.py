"""Batch runner script for Phase 10: Controlled FedAsync Baseline Experiments."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main() -> None:
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 2026]
    n_clients = 10
    updates = 30
    local_epochs = 1
    batch_size = 32
    lr = 0.01
    alpha_0 = 1.0
    delay_dist = "uniform"
    max_delay = 3

    experiment_specs: List[Dict[str, Any]] = []

    # 1. IID experiments
    for seed in seeds:
        experiment_specs.append({
            "name": f"IID (seed={seed})",
            "partition_method": "iid",
            "alpha": None,
            "seed": seed,
            "output_file": results_dir / f"fedasync_iid_seed_{seed}.json",
        })

    # 2. Non-IID Dirichlet experiments (alphas: 0.1, 0.5, 1.0)
    for alpha in [0.1, 0.5, 1.0]:
        for seed in seeds:
            experiment_specs.append({
                "name": f"Dirichlet (alpha={alpha}, seed={seed})",
                "partition_method": "dirichlet",
                "alpha": alpha,
                "seed": seed,
                "output_file": results_dir / f"fedasync_dirichlet_alpha_{alpha}_seed_{seed}.json",
            })

    total_experiments = len(experiment_specs)
    print(f"Starting Phase 10 FedAsync experiments: {total_experiments} total runs scheduled.")
    print("=" * 70)

    summary_rows: List[Dict[str, Any]] = []
    start_total_time = time.time()
    completed_count = 0

    python_executable = sys.executable

    for idx, spec in enumerate(experiment_specs, 1):
        print(f"\n[{idx}/{total_experiments}] Executing: {spec['name']}")
        cmd = [
            python_executable,
            "-m",
            "src.fedasync",
            "--clients", str(n_clients),
            "--updates", str(updates),
            "--epochs", str(local_epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--partition", spec["partition_method"],
            "--seed", str(spec["seed"]),
            "--alpha-0", str(alpha_0),
            "--delay-dist", delay_dist,
            "--max-delay", str(max_delay),
            "--output", str(spec["output_file"]),
        ]
        if spec["alpha"] is not None:
            cmd.extend(["--alpha", str(spec["alpha"])])

        run_start = time.time()
        # Subprocess execution with explicit returncode check
        proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
        run_duration = time.time() - run_start

        if proc.returncode != 0:
            print(f"ERROR: Experiment {spec['name']} failed with returncode {proc.returncode}!")
            print("STDOUT:", proc.stdout)
            print("STDERR:", proc.stderr)
            sys.exit(proc.returncode)

        if not spec["output_file"].exists() or spec["output_file"].stat().st_size == 0:
            print(f"ERROR: Output file {spec['output_file']} was not created or is empty!")
            sys.exit(1)

        # Parse output JSON to ensure completeness
        with open(spec["output_file"], "r", encoding="utf-8") as f:
            data = json.load(f)

        update_log = data.get("update_log", [])
        if len(update_log) != updates:
            print(f"ERROR: Expected {updates} updates, got {len(update_log)} in {spec['output_file']}!")
            sys.exit(1)

        summary = data.get("summary", {})
        if summary.get("final_server_version") != updates:
            print(f"ERROR: Expected final_server_version={updates}, got {summary.get('final_server_version')}!")
            sys.exit(1)

        last_update = update_log[-1]
        weights = [u["aggregation_weight"] for u in update_log]

        summary_row = {
            "partition_method": spec["partition_method"],
            "alpha": spec["alpha"] if spec["alpha"] is not None else "none",
            "seed": spec["seed"],
            "clients": n_clients,
            "applied_updates": len(update_log),
            "final_server_version": summary.get("final_server_version"),
            "final_validation_accuracy": last_update.get("val_accuracy"),
            "final_validation_loss": last_update.get("val_loss"),
            "final_test_accuracy": data.get("final_test_accuracy"),
            "final_test_loss": data.get("final_test_loss"),
            "mean_staleness": summary.get("mean_staleness"),
            "max_staleness": summary.get("max_staleness"),
            "mean_aggregation_weight": round(float(np.mean(weights)), 6),
            "min_aggregation_weight": round(float(np.min(weights)), 6),
            "max_aggregation_weight": round(float(np.max(weights)), 6),
            "simulated_time": summary.get("total_simulated_time"),
        }
        summary_rows.append(summary_row)
        completed_count += 1

        print(
            f"Finished {spec['name']} in {run_duration:.2f}s | "
            f"Test Acc: {summary_row['final_test_accuracy']:.4f} | "
            f"Test Loss: {summary_row['final_test_loss']:.4f} | "
            f"Mean Staleness: {summary_row['mean_staleness']:.2f} | "
            f"Simulated Time: {summary_row['simulated_time']} ticks"
        )

    total_duration = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"All {completed_count}/{total_experiments} runs finished successfully.")
    print(f"Total wall-clock duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")

    # Save summary CSV
    summary_csv_path = results_dir / "fedasync_summary.csv"
    fieldnames = [
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
    ]

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary CSV saved to {summary_csv_path} with {len(summary_rows)} rows.")


if __name__ == "__main__":
    main()
