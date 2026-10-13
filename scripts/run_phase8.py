"""Execution script for Phase 8: Controlled Synchronous FedAvg Baseline Experiments."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from src.fedavg import FedAvgConfig, run_fedavg_simulation


def run_all_experiments() -> None:
    """Run all 12 controlled synchronous FedAvg experiments and generate summary CSV."""
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 2026]
    n_clients = 10
    num_rounds = 30
    local_epochs = 1
    batch_size = 32
    lr = 0.01
    delay_dist = "uniform"
    max_delay = 3
    staleness_threshold = 2

    experiment_specs: List[Dict[str, Any]] = []

    # 1. IID experiments
    for seed in seeds:
        spec = {
            "name": f"IID (seed={seed})",
            "partition_method": "iid",
            "alpha": None,
            "seed": seed,
            "output_file": results_dir / f"sync_iid_seed_{seed}.json",
        }
        experiment_specs.append(spec)

    # 2. Non-IID Dirichlet experiments (alphas: 0.1, 0.5, 1.0)
    alphas = [0.1, 0.5, 1.0]
    for alpha in alphas:
        for seed in seeds:
            spec = {
                "name": f"Dirichlet (alpha={alpha}, seed={seed})",
                "partition_method": "dirichlet",
                "alpha": alpha,
                "seed": seed,
                "output_file": results_dir / f"sync_dirichlet_alpha_{alpha}_seed_{seed}.json",
            }
            experiment_specs.append(spec)

    total_experiments = len(experiment_specs)
    print(f"Starting Phase 8 experiments: {total_experiments} total runs scheduled.")
    print("=" * 70)

    summary_rows: List[Dict[str, Any]] = []
    start_total_time = time.time()
    completed_count = 0

    for idx, spec in enumerate(experiment_specs, 1):
        print(f"\n[{idx}/{total_experiments}] Running: {spec['name']}")
        run_start = time.time()

        cfg = FedAvgConfig(
            n_clients=n_clients,
            local_epochs=local_epochs,
            num_rounds=num_rounds,
            partition_method=spec["partition_method"],
            alpha=spec["alpha"] if spec["alpha"] is not None else 0.5,
            seed=spec["seed"],
            batch_size=batch_size,
            lr=lr,
            delay_distribution=delay_dist,
            max_delay=max_delay,
            staleness_threshold=staleness_threshold,
            staleness_seed=spec["seed"],
        )

        try:
            results = run_fedavg_simulation(
                fedavg_config=cfg,
                results_path=spec["output_file"],
            )
        except Exception as e:
            print(f"FAILED run {spec['name']} with error: {e}")
            raise e

        run_duration = time.time() - run_start
        completed_count += 1

        history = results.get("history", [])
        client_updates = results.get("client_updates", [])
        final_test = results.get("test", {})

        final_test_acc = final_test.get("accuracy")
        final_test_loss = final_test.get("loss")
        best_val_acc = max((h["val_accuracy"] for h in history), default=None)

        staleness_vals = [u["staleness"] for u in client_updates]
        mean_staleness = round(float(np.mean(staleness_vals)), 4) if staleness_vals else 0.0
        accepted_count = sum(1 for u in client_updates if u.get("accepted", True))
        rejected_count = sum(1 for u in client_updates if not u.get("accepted", True))

        summary_row = {
            "partition_method": spec["partition_method"],
            "alpha": spec["alpha"] if spec["alpha"] is not None else "none",
            "seed": spec["seed"],
            "rounds": num_rounds,
            "clients": n_clients,
            "final_test_accuracy": final_test_acc,
            "final_test_loss": final_test_loss,
            "best_validation_accuracy": best_val_acc,
            "mean_staleness": mean_staleness,
            "accepted_updates": accepted_count,
            "rejected_updates": rejected_count,
        }
        summary_rows.append(summary_row)

        print(f"Completed {spec['name']} in {run_duration:.2f}s | "
              f"Test Acc: {final_test_acc:.4f} | "
              f"Test Loss: {final_test_loss:.4f} | "
              f"Best Val Acc: {best_val_acc:.4f} | "
              f"Mean Staleness: {mean_staleness:.2f}")

    total_duration = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"All {completed_count}/{total_experiments} runs finished successfully.")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")

    # Save summary CSV
    summary_csv_path = results_dir / "sync_fedavg_summary.csv"
    fieldnames = [
        "partition_method",
        "alpha",
        "seed",
        "rounds",
        "clients",
        "final_test_accuracy",
        "final_test_loss",
        "best_validation_accuracy",
        "mean_staleness",
        "accepted_updates",
        "rejected_updates",
    ]

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary CSV to: {summary_csv_path}")


if __name__ == "__main__":
    run_all_experiments()
