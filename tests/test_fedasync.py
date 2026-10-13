"""Unit tests for asynchronous FedAsync event-driven simulation and model-version staleness."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.fedasync import (
    FedAsyncConfig,
    FedAsyncSimulator,
    compute_fedasync_weight,
    fedasync_aggregate_parameters,
    run_fedasync_simulation,
)


class TestFedAsync(unittest.TestCase):
    """Test suite for FedAsync and asynchronous event-driven simulation foundation."""

    def test_model_version_staleness_calculation(self) -> None:
        """Requirement 5a: model-version staleness is calculated correctly: tau = server_ver - client_ver."""
        # Fresh update: client trained on the current server version
        self.assertEqual(compute_fedasync_weight(staleness=0, alpha_0=1.0), 1.0)

        # Stale updates: client trained on older versions
        # tau = 1: server at version 1, client trained on version 0
        w1 = compute_fedasync_weight(staleness=1, alpha_0=1.0)
        self.assertAlmostEqual(w1, 1.0 / 2.0)

        # tau = 4: server at version 5, client trained on version 1
        w4 = compute_fedasync_weight(staleness=4, alpha_0=1.0)
        self.assertAlmostEqual(w4, 1.0 / 5.0)

        # Negative staleness must be rejected
        with self.assertRaises(ValueError):
            compute_fedasync_weight(staleness=-1, alpha_0=1.0)

    def test_intervening_server_updates_increase_staleness(self) -> None:
        """Requirement 5b: delayed updates have higher staleness when intervening server updates occur."""
        # Setup small simulation where delay ensures intervening updates
        cfg = FedAsyncConfig(
            n_clients=3,
            max_updates=5,
            local_epochs=1,
            seed=42,
            compute_time=1.0,
            delay_distribution="uniform",
            max_delay=3,
        )
        sim = FedAsyncSimulator(cfg)
        results = sim.run_simulation()
        update_log = results["update_log"]

        self.assertGreater(len(update_log), 0)
        # Verify that for every update: staleness == server_version_before - client_training_version
        for u in update_log:
            expected_staleness = u["server_version_before"] - u["client_training_version"]
            self.assertEqual(u["staleness"], expected_staleness)
            self.assertGreaterEqual(u["staleness"], 0)

        # In an asynchronous setting with intervening updates, staleness should increase above 0
        staleness_values = [u["staleness"] for u in update_log]
        self.assertTrue(any(s > 0 for s in staleness_values), "Expected some stale updates when multiple clients run asynchronously")

    def test_stale_update_receives_lower_weight(self) -> None:
        """Requirement 5c: a stale update receives a lower FedAsync weight than a fresh update."""
        alpha_0 = 1.0
        w_fresh = compute_fedasync_weight(staleness=0, alpha_0=alpha_0)
        w_stale_1 = compute_fedasync_weight(staleness=1, alpha_0=alpha_0)
        w_stale_3 = compute_fedasync_weight(staleness=3, alpha_0=alpha_0)
        w_stale_9 = compute_fedasync_weight(staleness=9, alpha_0=alpha_0)

        self.assertGreater(w_fresh, w_stale_1)
        self.assertGreater(w_stale_1, w_stale_3)
        self.assertGreater(w_stale_3, w_stale_9)

        # Test parameter aggregation with staleness discounting
        server_p = [np.ones((4, 4), dtype=np.float32)]
        client_p = [np.zeros((4, 4), dtype=np.float32)]

        # Fresh update (tau=0, w=1.0): replaces server model entirely
        agg_fresh = fedasync_aggregate_parameters(server_p, client_p, w_fresh)
        np.testing.assert_allclose(agg_fresh[0], np.zeros((4, 4), dtype=np.float32))

        # Stale update (tau=1, w=0.5): 50% server, 50% client
        agg_stale_1 = fedasync_aggregate_parameters(server_p, client_p, w_stale_1)
        np.testing.assert_allclose(agg_stale_1[0], np.full((4, 4), 0.5, dtype=np.float32))

        # Stale update (tau=3, w=0.25): 75% server, 25% client
        agg_stale_3 = fedasync_aggregate_parameters(server_p, client_p, w_stale_3)
        np.testing.assert_allclose(agg_stale_3[0], np.full((4, 4), 0.75, dtype=np.float32))

    def test_changing_alpha_0_changes_applied_weight(self) -> None:
        """Requirement 5d: changing alpha_0 changes the applied weight."""
        tau = 2
        w_default = compute_fedasync_weight(staleness=tau, alpha_0=1.0)
        w_half = compute_fedasync_weight(staleness=tau, alpha_0=0.5)
        w_custom = compute_fedasync_weight(staleness=tau, alpha_0=0.3)

        self.assertAlmostEqual(w_default, 1.0 / 3.0)
        self.assertAlmostEqual(w_half, 0.5 / 3.0)
        self.assertAlmostEqual(w_custom, 0.3 / 3.0)
        self.assertNotEqual(w_default, w_half)
        self.assertNotEqual(w_half, w_custom)

    def test_fixed_seed_produces_identical_event_ordering_and_results(self) -> None:
        """Requirement 5e: fixed seed produces identical event ordering and results."""
        cfg1 = FedAsyncConfig(
            n_clients=4,
            max_updates=8,
            local_epochs=1,
            seed=123,
            partition_method="dirichlet",
            alpha=0.5,
            delay_distribution="uniform",
            max_delay=2,
        )
        cfg2 = FedAsyncConfig(
            n_clients=4,
            max_updates=8,
            local_epochs=1,
            seed=123,
            partition_method="dirichlet",
            alpha=0.5,
            delay_distribution="uniform",
            max_delay=2,
        )

        sim1 = FedAsyncSimulator(cfg1)
        res1 = sim1.run_simulation()

        sim2 = FedAsyncSimulator(cfg2)
        res2 = sim2.run_simulation()

        # Check identical event sequence and update logs
        self.assertEqual(len(res1["event_log"]), len(res2["event_log"]))
        for e1, e2 in zip(res1["event_log"], res2["event_log"]):
            self.assertEqual(e1["time"], e2["time"])
            self.assertEqual(e1["event_type"], e2["event_type"])
            self.assertEqual(e1["client_id"], e2["client_id"])

        self.assertEqual(len(res1["update_log"]), len(res2["update_log"]))
        for u1, u2 in zip(res1["update_log"], res2["update_log"]):
            self.assertEqual(u1["staleness"], u2["staleness"])
            self.assertEqual(u1["aggregation_weight"], u2["aggregation_weight"])
            self.assertEqual(u1["client_id"], u2["client_id"])
            self.assertAlmostEqual(u1["val_accuracy"], u2["val_accuracy"], places=5)
            self.assertAlmostEqual(u1["test_accuracy"], u2["test_accuracy"], places=5)

        self.assertAlmostEqual(res1["final_test_accuracy"], res2["final_test_accuracy"], places=5)
        self.assertAlmostEqual(res1["final_test_loss"], res2["final_test_loss"], places=5)

    def test_event_lifecycle_presence(self) -> None:
        """Verify all 4 core events (DISPATCH, TRAIN_COMPLETE, UPDATE_ARRIVAL, AGGREGATE) occur."""
        cfg = FedAsyncConfig(n_clients=3, max_updates=4, seed=42)
        sim = FedAsyncSimulator(cfg)
        res = sim.run_simulation()

        event_types = {e["event_type"] for e in res["event_log"]}
        self.assertIn("DISPATCH", event_types)
        self.assertIn("TRAIN_COMPLETE", event_types)
        self.assertIn("UPDATE_ARRIVAL", event_types)
        self.assertIn("AGGREGATE", event_types)

    def test_config_json_roundtrip(self) -> None:
        """Verify FedAsyncConfig serialization to/from JSON."""
        cfg = FedAsyncConfig(
            n_clients=7,
            max_updates=25,
            alpha_0=0.8,
            delay_distribution="constant",
            max_delay=2,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tpath = f.name
        try:
            cfg.to_json(tpath)
            loaded = FedAsyncConfig.from_json(tpath)
            self.assertEqual(loaded.n_clients, 7)
            self.assertEqual(loaded.max_updates, 25)
            self.assertEqual(loaded.alpha_0, 0.8)
            self.assertEqual(loaded.delay_distribution, "constant")
            self.assertEqual(loaded.max_delay, 2)
        finally:
            Path(tpath).unlink(missing_ok=True)

    def test_run_simulation_saves_json_and_csv(self) -> None:
        """Verify simulation runner saves structured JSON and CSV outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_out = Path(tmpdir) / "test_out.json"
            csv_out = Path(tmpdir) / "test_out.csv"

            cfg = FedAsyncConfig(n_clients=3, max_updates=4, seed=42)
            results = run_fedasync_simulation(
                fedasync_config=cfg,
                results_path=json_out,
                csv_path=csv_out,
            )

            self.assertTrue(json_out.exists())
            self.assertTrue(csv_out.exists())

            with open(json_out, "r") as f:
                data = json.load(f)

            self.assertIn("config", data)
            self.assertIn("client_stats", data)
            self.assertIn("initial", data)
            self.assertIn("event_log", data)
            self.assertIn("update_log", data)
            self.assertIn("history", data)
            self.assertIn("summary", data)
            self.assertIn("final_test_accuracy", data)
            self.assertIn("final_test_loss", data)
            self.assertIn("mean_staleness", data)
            self.assertEqual(data["summary"]["total_updates"], 4)


if __name__ == "__main__":
    unittest.main()
