"""Unit tests for Buffered Asynchronous Federated Learning (FedBuff)."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.fedbuff import (
    FedBuffConfig,
    FedBuffSimulator,
    aggregate_buffered_parameters,
    run_fedbuff_simulation,
)


class TestFedBuff(unittest.TestCase):
    """Test suite for FedBuff algorithm, buffer mechanics, and simulation lifecycle."""

    def setUp(self) -> None:
        self.base_config = FedBuffConfig(
            n_clients=5,
            max_updates=10,
            buffer_size=4,
            flush_final_buffer=False,
            local_epochs=1,
            batch_size=32,
            lr=0.01,
            partition_method="dirichlet",
            alpha=0.5,
            seed=42,
            delay_distribution="uniform",
            max_delay=3,
            compute_time=1.0,
            staleness_seed=42,
            min_samples_per_client=5,
        )

    def test_config_validation(self) -> None:
        """Verify parameter bounds and validation in FedBuffConfig."""
        with self.assertRaises(ValueError):
            FedBuffConfig(n_clients=0)
        with self.assertRaises(ValueError):
            FedBuffConfig(max_updates=0)
        with self.assertRaises(ValueError):
            FedBuffConfig(buffer_size=0)
        with self.assertRaises(ValueError):
            FedBuffConfig(local_epochs=0)
        with self.assertRaises(ValueError):
            FedBuffConfig(max_delay=-1)
        with self.assertRaises(ValueError):
            FedBuffConfig(compute_time=-0.1)

    def test_sample_count_weighted_aggregation_math(self) -> None:
        """Verify sample-count-weighted parameter averaging is mathematically exact."""
        # Client 0: 100 samples, weights = [[1.0, 2.0], [3.0]]
        # Client 1: 300 samples, weights = [[5.0, 6.0], [7.0]]
        # Expected:
        # Layer 0: (100 * [1, 2] + 300 * [5, 6]) / 400 = [1600/400, 2000/400] = [4.0, 5.0]
        # Layer 1: (100 * 3 + 300 * 7) / 400 = (300 + 2100) / 400 = 2400 / 400 = [6.0]
        updates = [
            {
                "trained_params": [np.array([1.0, 2.0]), np.array([3.0])],
                "sample_count": 100,
            },
            {
                "trained_params": [np.array([5.0, 6.0]), np.array([7.0])],
                "sample_count": 300,
            },
        ]
        aggregated = aggregate_buffered_parameters(updates)
        self.assertEqual(len(aggregated), 2)
        np.testing.assert_allclose(aggregated[0], np.array([4.0, 5.0]))
        np.testing.assert_allclose(aggregated[1], np.array([6.0]))

        # Error cases
        with self.assertRaises(ValueError):
            aggregate_buffered_parameters([])
        with self.assertRaises(ValueError):
            aggregate_buffered_parameters([
                {"trained_params": [np.array([1.0])], "sample_count": 0}
            ])

    def test_no_aggregation_before_k_arrivals(self) -> None:
        """Verify buffer does not aggregate before K arrivals arrive."""
        cfg = copy.deepcopy(self.base_config)
        cfg.buffer_size = 4
        cfg.max_updates = 3  # Stop after 3 arrivals (K=4)
        cfg.flush_final_buffer = False

        sim = FedBuffSimulator(cfg)
        res = sim.run_simulation()

        self.assertEqual(res["arrival_count"], 3)
        self.assertEqual(res["server_aggregation_count"], 0)
        self.assertEqual(res["final_server_version"], 0)
        self.assertEqual(res["unflushed_buffer_count"], 3)
        self.assertEqual(res["applied_update_count"], 0)
        self.assertEqual(len(res["aggregation_log"]), 0)

    def test_exactly_k_arrivals_trigger_one_aggregation(self) -> None:
        """Verify that exactly K arrivals trigger exactly one aggregation."""
        cfg = copy.deepcopy(self.base_config)
        cfg.buffer_size = 4
        cfg.max_updates = 4  # Exactly K arrivals
        cfg.flush_final_buffer = False

        sim = FedBuffSimulator(cfg)
        res = sim.run_simulation()

        self.assertEqual(res["arrival_count"], 4)
        self.assertEqual(res["server_aggregation_count"], 1)
        self.assertEqual(res["final_server_version"], 1)
        self.assertEqual(res["unflushed_buffer_count"], 0)
        self.assertEqual(res["applied_update_count"], 4)
        self.assertEqual(len(res["aggregation_log"]), 1)
        self.assertEqual(res["aggregation_log"][0]["buffer_size"], 4)
        self.assertEqual(res["aggregation_log"][0]["server_version_before"], 0)
        self.assertEqual(res["aggregation_log"][0]["server_version_after"], 1)

    def test_server_version_increments_per_buffer_not_per_arrival(self) -> None:
        """Verify server version increments once per buffer aggregation, not per arrival."""
        cfg = copy.deepcopy(self.base_config)
        cfg.buffer_size = 3
        cfg.max_updates = 9  # 9 arrivals with K=3 -> exactly 3 aggregations
        cfg.flush_final_buffer = False

        sim = FedBuffSimulator(cfg)
        res = sim.run_simulation()

        self.assertEqual(res["arrival_count"], 9)
        self.assertEqual(res["server_aggregation_count"], 3)
        # Server version must be 3, NOT 9!
        self.assertEqual(res["final_server_version"], 3)
        self.assertEqual(res["unflushed_buffer_count"], 0)
        self.assertEqual(res["applied_update_count"], 9)

        # Check versions in aggregation log
        self.assertEqual(len(res["aggregation_log"]), 3)
        for i, agg in enumerate(res["aggregation_log"]):
            self.assertEqual(agg["aggregation_index"], i + 1)
            self.assertEqual(agg["server_version_before"], i)
            self.assertEqual(agg["server_version_after"], i + 1)
            self.assertEqual(agg["buffer_size"], 3)

    def test_staleness_uses_model_version_not_delay(self) -> None:
        """Verify arrival staleness is true model-version lag, not sampled delay."""
        cfg = copy.deepcopy(self.base_config)
        cfg.buffer_size = 3
        cfg.max_updates = 9
        cfg.delay_distribution = "uniform"
        cfg.max_delay = 3

        sim = FedBuffSimulator(cfg)
        res = sim.run_simulation()

        # In every arrival record: staleness = server_version_at_arrival - client_training_version
        saw_delay_different_from_staleness = False
        for arr in res["arrival_log"]:
            expected_staleness = arr["server_version_at_arrival"] - arr["client_training_version"]
            self.assertEqual(arr["staleness"], expected_staleness)
            if arr["comm_delay"] != arr["staleness"]:
                saw_delay_different_from_staleness = True

        self.assertTrue(
            saw_delay_different_from_staleness,
            "Expected staleness (version difference) to differ from raw communication delay in at least one update.",
        )

    def test_partial_final_buffer_default_discard(self) -> None:
        """Verify incomplete final buffer is discarded by default when flush_final_buffer is False."""
        cfg = copy.deepcopy(self.base_config)
        cfg.buffer_size = 4
        cfg.max_updates = 6  # 6 arrivals: first 4 aggregated, last 2 remaining
        cfg.flush_final_buffer = False

        sim = FedBuffSimulator(cfg)
        res = sim.run_simulation()

        self.assertEqual(res["arrival_count"], 6)
        self.assertEqual(res["server_aggregation_count"], 1)
        self.assertEqual(res["final_server_version"], 1)
        self.assertEqual(res["applied_update_count"], 4)
        self.assertEqual(res["unflushed_buffer_count"], 2)
        self.assertFalse(res["final_buffer_flushed"])

        # Check event log records FINAL_BUFFER_DISCARDED
        discarded_events = [e for e in res["event_log"] if e.get("event_type") == "FINAL_BUFFER_DISCARDED"]
        self.assertEqual(len(discarded_events), 1)
        self.assertEqual(discarded_events[0]["unflushed_buffer_count"], 2)

    def test_partial_final_buffer_flush_enabled(self) -> None:
        """Verify incomplete final buffer is aggregated when flush_final_buffer is True."""
        cfg = copy.deepcopy(self.base_config)
        cfg.buffer_size = 4
        cfg.max_updates = 6  # 6 arrivals: first 4 aggregated (K=4), final 2 flushed
        cfg.flush_final_buffer = True

        sim = FedBuffSimulator(cfg)
        res = sim.run_simulation()

        self.assertEqual(res["arrival_count"], 6)
        self.assertEqual(res["server_aggregation_count"], 2)
        self.assertEqual(res["final_server_version"], 2)
        self.assertEqual(res["applied_update_count"], 6)
        self.assertEqual(res["unflushed_buffer_count"], 0)
        self.assertTrue(res["final_buffer_flushed"])

        # Check aggregation log has 2 entries: regular then flush
        self.assertEqual(len(res["aggregation_log"]), 2)
        self.assertEqual(res["aggregation_log"][0]["buffer_size"], 4)
        self.assertFalse(res["aggregation_log"][0]["is_final_flush"])
        self.assertEqual(res["aggregation_log"][1]["buffer_size"], 2)
        self.assertTrue(res["aggregation_log"][1]["is_final_flush"])

        # Check event log records FINAL_BUFFER_FLUSHED
        flush_events = [e for e in res["event_log"] if e.get("event_type") == "FINAL_BUFFER_FLUSHED"]
        self.assertEqual(len(flush_events), 1)
        self.assertEqual(flush_events[0]["buffer_size"], 2)

    def test_deterministic_fixed_seed(self) -> None:
        """Verify identical seed produces strictly identical simulation execution."""
        cfg1 = copy.deepcopy(self.base_config)
        cfg2 = copy.deepcopy(self.base_config)

        res1 = FedBuffSimulator(cfg1).run_simulation()
        res2 = FedBuffSimulator(cfg2).run_simulation()

        self.assertEqual(res1["arrival_count"], res2["arrival_count"])
        self.assertEqual(res1["server_aggregation_count"], res2["server_aggregation_count"])
        self.assertEqual(res1["final_server_version"], res2["final_server_version"])
        self.assertEqual(res1["final_test_accuracy"], res2["final_test_accuracy"])
        self.assertEqual(res1["final_test_loss"], res2["final_test_loss"])
        self.assertEqual(res1["arrival_log"], res2["arrival_log"])
        self.assertEqual(res1["aggregation_log"], res2["aggregation_log"])

    def test_run_simulation_saves_json_and_csv(self) -> None:
        """Verify file output functionality for JSON metrics and CSV arrival log."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "metrics.json"
            csv_path = Path(tmp_dir) / "arrivals.csv"

            cfg = copy.deepcopy(self.base_config)
            cfg.max_updates = 5
            cfg.buffer_size = 5

            res = run_fedbuff_simulation(cfg, results_path=json_path, csv_path=csv_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

            with open(json_path, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
            self.assertEqual(saved_json["arrival_count"], 5)
            self.assertEqual(saved_json["final_server_version"], 1)

            # Check CSV rows
            lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
            # 1 header + 5 arrivals = 6 lines
            self.assertEqual(len(lines), 6)


if __name__ == "__main__":
    unittest.main()
