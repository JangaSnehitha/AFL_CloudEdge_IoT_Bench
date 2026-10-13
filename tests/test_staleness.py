"""Focused unit and integration tests for staleness instrumentation in synchronous FedAvg."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from src.dataset import SyntheticDatasetConfig
from src.fedavg import FedAvgConfig, run_fedavg_simulation
from src.model import MLPConfig
from src.staleness import (
    StalenessConfig,
    StalenessSimulator,
    save_client_updates_csv,
)
from src.strategy import FedAvgStrategy


class TestDelayGenerationAndReproducibility(unittest.TestCase):
    """Tests for simulated update delay distributions and RNG reproducibility."""

    def test_constant_delay(self):
        cfg = StalenessConfig(distribution="constant", max_delay=3, seed=42)
        sim = StalenessSimulator(cfg)
        for i in range(10):
            self.assertEqual(sim.sample_delay(client_id=i, server_round=1), 3)

    def test_zero_delay(self):
        cfg = StalenessConfig(distribution="zero", max_delay=5, seed=42)
        sim = StalenessSimulator(cfg)
        for i in range(10):
            self.assertEqual(sim.sample_delay(client_id=i, server_round=1), 0)

    def test_uniform_delay_bounds(self):
        cfg = StalenessConfig(distribution="uniform", max_delay=4, seed=42)
        sim = StalenessSimulator(cfg)
        samples = [sim.sample_delay(client_id=i % 5, server_round=i // 5 + 1) for i in range(100)]
        self.assertTrue(all(0 <= s <= 4 for s in samples))
        # Ensure it is not trivial (multiple distinct values)
        self.assertGreater(len(set(samples)), 1)

    def test_poisson_and_exponential_bounds(self):
        for dist in ("poisson", "exponential"):
            cfg = StalenessConfig(distribution=dist, max_delay=5, seed=42)
            sim = StalenessSimulator(cfg)
            samples = [sim.sample_delay(client_id=i % 5, server_round=1) for i in range(50)]
            self.assertTrue(all(0 <= s <= 5 for s in samples))

    def test_reproducibility_same_seed(self):
        cfg1 = StalenessConfig(distribution="uniform", max_delay=5, seed=123)
        cfg2 = StalenessConfig(distribution="uniform", max_delay=5, seed=123)
        sim1 = StalenessSimulator(cfg1)
        sim2 = StalenessSimulator(cfg2)

        delays1 = [sim1.sample_delay(client_id=i % 4, server_round=i // 4 + 1) for i in range(40)]
        delays2 = [sim2.sample_delay(client_id=i % 4, server_round=i // 4 + 1) for i in range(40)]
        self.assertEqual(delays1, delays2)

    def test_different_seeds_differ(self):
        cfg1 = StalenessConfig(distribution="uniform", max_delay=10, seed=123)
        cfg2 = StalenessConfig(distribution="uniform", max_delay=10, seed=999)
        sim1 = StalenessSimulator(cfg1)
        sim2 = StalenessSimulator(cfg2)

        delays1 = [sim1.sample_delay(client_id=i, server_round=1) for i in range(30)]
        delays2 = [sim2.sample_delay(client_id=i, server_round=1) for i in range(30)]
        self.assertNotEqual(delays1, delays2)


class TestStalenessCalculation(unittest.TestCase):
    """Tests for staleness calculation at aggregation time."""

    def test_calculate_staleness(self):
        cfg = StalenessConfig(distribution="uniform", max_delay=5, seed=42)
        sim = StalenessSimulator(cfg)
        self.assertEqual(sim.calculate_staleness(0), 0)
        self.assertEqual(sim.calculate_staleness(3), 3)
        self.assertEqual(sim.calculate_staleness(5), 5)
        self.assertEqual(sim.calculate_staleness(-2), 0)


class TestStalenessThresholdAndAcceptance(unittest.TestCase):
    """Tests for update acceptance logic based on staleness threshold."""

    def test_threshold_none_accepts_all(self):
        cfg = StalenessConfig(staleness_threshold=None)
        sim = StalenessSimulator(cfg)
        for s in (0, 1, 5, 20, 100):
            self.assertTrue(sim.is_accepted(s))

    def test_threshold_enforced(self):
        cfg = StalenessConfig(staleness_threshold=2)
        sim = StalenessSimulator(cfg)
        self.assertTrue(sim.is_accepted(0))
        self.assertTrue(sim.is_accepted(1))
        self.assertTrue(sim.is_accepted(2))   # Boundary: equal to threshold is accepted
        self.assertFalse(sim.is_accepted(3))  # Exceeds threshold: rejected
        self.assertFalse(sim.is_accepted(10))


class TestMetricLoggingStructure(unittest.TestCase):
    """Tests verifying client update log fields, types, and CSV export."""

    def setUp(self):
        self.client_updates = []
        self.staleness_cfg = StalenessConfig(
            distribution="constant",
            max_delay=2,
            staleness_threshold=2,
            seed=42,
        )
        self.simulator = StalenessSimulator(self.staleness_cfg)

        init_params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
        self.strategy = FedAvgStrategy(
            initial_parameters=init_params,
            min_fit_clients=2,
            min_available_clients=2,
            staleness_simulator=self.simulator,
            partition_method="dirichlet",
            alpha=0.5,
            local_epochs=2,
            client_updates=self.client_updates,
        )

    def test_logged_fields_and_types(self):
        status = Status(code=Code.OK, message="Success")
        res1 = FitRes(
            status=status,
            parameters=ndarrays_to_parameters([np.ones((2, 2), dtype=np.float32)]),
            num_examples=120,
            metrics={"cid": "0", "loss": 0.854321},
        )
        res2 = FitRes(
            status=status,
            parameters=ndarrays_to_parameters([np.ones((2, 2), dtype=np.float32)]),
            num_examples=80,
            metrics={"cid": "1", "loss": 1.123456},
        )

        p1, p2 = MagicMock(spec=ClientProxy), MagicMock(spec=ClientProxy)
        self.strategy.aggregate_fit(server_round=1, results=[(p1, res1), (p2, res2)], failures=[])

        self.assertEqual(len(self.client_updates), 2)

        required_keys = {
            "client_id",
            "round",
            "partition_method",
            "alpha",
            "n_samples",
            "local_epochs",
            "simulated_delay",
            "staleness",
            "accepted",
            "client_loss",
            "num_examples",
        }

        for rec in self.client_updates:
            self.assertEqual(set(rec.keys()), required_keys)
            self.assertIsInstance(rec["client_id"], int)
            self.assertIsInstance(rec["round"], int)
            self.assertIsInstance(rec["partition_method"], str)
            self.assertIsInstance(rec["alpha"], float)
            self.assertIsInstance(rec["n_samples"], int)
            self.assertIsInstance(rec["local_epochs"], int)
            self.assertIsInstance(rec["simulated_delay"], int)
            self.assertIsInstance(rec["staleness"], int)
            self.assertIsInstance(rec["accepted"], bool)
            self.assertIsInstance(rec["client_loss"], float)
            self.assertIsInstance(rec["num_examples"], int)

        rec0 = self.client_updates[0]
        self.assertEqual(rec0["client_id"], 0)
        self.assertEqual(rec0["round"], 1)
        self.assertEqual(rec0["partition_method"], "dirichlet")
        self.assertEqual(rec0["alpha"], 0.5)
        self.assertEqual(rec0["n_samples"], 120)
        self.assertEqual(rec0["local_epochs"], 2)
        self.assertEqual(rec0["simulated_delay"], 2)
        self.assertEqual(rec0["staleness"], 2)
        self.assertTrue(rec0["accepted"])
        self.assertAlmostEqual(rec0["client_loss"], 0.854321, places=5)
        self.assertEqual(rec0["num_examples"], 120)

    def test_save_client_updates_csv(self):
        sample_updates = [
            {
                "client_id": 0,
                "round": 1,
                "partition_method": "iid",
                "alpha": 0.5,
                "n_samples": 50,
                "local_epochs": 1,
                "simulated_delay": 1,
                "staleness": 1,
                "accepted": True,
                "client_loss": 0.5,
                "num_examples": 50,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "updates.csv"
            save_client_updates_csv(sample_updates, csv_path)
            self.assertTrue(csv_path.exists())

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["client_id"], "0")
                self.assertEqual(rows[0]["round"], "1")
                self.assertEqual(rows[0]["partition_method"], "iid")
                self.assertEqual(rows[0]["accepted"], "True")


class TestStalenessSimulationSmoke(unittest.TestCase):
    """End-to-end smoke test verifying staleness instrumentation during simulation."""

    def test_smoke_staleness_run(self):
        ds_cfg = SyntheticDatasetConfig(
            seed=42,
            n_samples=60,
            n_features=4,
            n_classes=2,
            class_sep=2.0,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        model_cfg = MLPConfig(
            input_dim=4,
            hidden_dims=[8],
            output_dim=2,
            lr=0.05,
        )
        fedavg_cfg = FedAvgConfig(
            n_clients=2,
            local_epochs=1,
            num_rounds=2,
            partition_method="iid",
            seed=42,
            batch_size=16,
            min_samples_per_client=5,
            delay_distribution="uniform",
            max_delay=2,
            staleness_threshold=1,
            staleness_seed=42,
        )

        results = run_fedavg_simulation(
            fedavg_config=fedavg_cfg,
            dataset_config=ds_cfg,
            model_config=model_cfg,
        )

        self.assertIn("client_updates", results)
        # 2 clients x 2 rounds = 4 total updates
        self.assertEqual(len(results["client_updates"]), 4)

        for upd in results["client_updates"]:
            self.assertIn("simulated_delay", upd)
            self.assertIn("staleness", upd)
            self.assertIn("accepted", upd)
            self.assertTrue(0 <= upd["staleness"] <= 2)
            if upd["staleness"] <= 1:
                self.assertTrue(upd["accepted"])
            else:
                self.assertFalse(upd["accepted"])

        self.assertIn("history", results)
        self.assertEqual(len(results["history"]), 2)
        for h in results["history"]:
            self.assertIn("round", h)
            self.assertIn("val_loss", h)
            self.assertIn("val_accuracy", h)
            self.assertIn("test_loss", h)
            self.assertIn("test_accuracy", h)


if __name__ == "__main__":
    unittest.main()
