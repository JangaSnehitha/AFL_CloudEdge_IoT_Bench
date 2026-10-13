"""Unit and integration tests for Flower-based synchronous FedAvg implementation."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from src.client import (
    FlowerClient,
    create_client_fn,
    get_model_parameters,
    load_partitioned_dataloaders,
    set_model_parameters,
)
from src.dataset import SyntheticDatasetConfig
from src.fedavg import FedAvgConfig, run_fedavg_simulation
from src.model import MLPConfig, build_model
from src.partition import PartitionConfig
from src.strategy import FedAvgStrategy, aggregate_weighted_average, create_evaluate_fn


class TestFlowerClientParameterHandling(unittest.TestCase):
    """Tests for FlowerClient parameter extraction, loading, and local training."""

    def setUp(self):
        self.model_cfg = MLPConfig(
            input_dim=8,
            hidden_dims=[16, 8],
            output_dim=2,
            lr=0.05,
        )
        self.model = build_model(self.model_cfg)

        # Create small synthetic dataset for local client training
        x = torch.randn(40, 8, dtype=torch.float32)
        y = torch.randint(0, 2, (40,), dtype=torch.long)
        self.loader = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=False)

    def test_get_and_set_parameters(self):
        params = get_model_parameters(self.model)
        self.assertEqual(len(params), len(list(self.model.state_dict().values())))
        for p, tensor in zip(params, self.model.state_dict().values()):
            self.assertEqual(p.shape, tensor.shape)
            self.assertTrue(np.allclose(p, tensor.cpu().numpy()))

        # Modify parameters and load back
        perturbed = [p + 2.5 for p in params]
        set_model_parameters(self.model, perturbed)
        re_extracted = get_model_parameters(self.model)
        for orig, mod in zip(params, re_extracted):
            self.assertTrue(np.allclose(mod, orig + 2.5))

    def test_client_fit_updates_parameters(self):
        client = FlowerClient(
            cid="0",
            train_loader=self.loader,
            model_config=self.model_cfg,
            local_epochs=2,
        )
        initial_params = client.get_parameters()
        updated_params, num_examples, metrics = client.fit(
            initial_params, {"local_epochs": 2}
        )

        self.assertEqual(num_examples, 40)
        self.assertEqual(metrics["cid"], "0")
        self.assertIn("loss", metrics)
        self.assertGreater(metrics["loss"], 0.0)

        # Weights should change after gradient steps
        weights_changed = any(
            not np.allclose(init_p, upd_p)
            for init_p, upd_p in zip(initial_params, updated_params)
        )
        self.assertTrue(weights_changed)

    def test_client_evaluate(self):
        client = FlowerClient(
            cid="1",
            train_loader=self.loader,
            model_config=self.model_cfg,
        )
        params = client.get_parameters()
        loss, num_examples, metrics = client.evaluate(params, {})

        self.assertEqual(num_examples, 40)
        self.assertIsInstance(loss, float)
        self.assertIn("accuracy", metrics)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)


class TestSampleWeightedFedAvg(unittest.TestCase):
    """Tests for sample-size-weighted aggregation logic and strategy."""

    def test_equal_sample_size_matches_mean(self):
        w1 = [np.array([2.0, 4.0], dtype=np.float32)]
        w2 = [np.array([4.0, 6.0], dtype=np.float32)]
        results = [(w1, 50), (w2, 50)]

        aggregated = aggregate_weighted_average(results)
        expected = np.array([3.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(aggregated[0], expected)

    def test_unequal_sample_size_weights_proportionally(self):
        w1 = [np.array([1.0], dtype=np.float32)]
        w2 = [np.array([0.0], dtype=np.float32)]
        # Client 1 has 3x the samples of Client 2 -> result should be 0.75
        results = [(w1, 300), (w2, 100)]

        aggregated = aggregate_weighted_average(results)
        expected = np.array([0.75], dtype=np.float32)
        np.testing.assert_allclose(aggregated[0], expected)

    def test_empty_results_raises(self):
        with self.assertRaises(ValueError):
            aggregate_weighted_average([])

    def test_zero_samples_raises(self):
        w1 = [np.array([1.0], dtype=np.float32)]
        with self.assertRaises(ValueError):
            aggregate_weighted_average([(w1, 0)])

    def test_strategy_aggregate_fit(self):
        init_params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
        strategy = FedAvgStrategy(
            initial_parameters=init_params,
            min_fit_clients=2,
            min_available_clients=2,
        )

        w1 = [np.full((2, 2), 10.0, dtype=np.float32)]
        w2 = [np.full((2, 2), 20.0, dtype=np.float32)]

        status = Status(code=Code.OK, message="Success")
        res1 = FitRes(
            status=status,
            parameters=ndarrays_to_parameters(w1),
            num_examples=10,
            metrics={},
        )
        res2 = FitRes(
            status=status,
            parameters=ndarrays_to_parameters(w2),
            num_examples=30,
            metrics={},
        )

        proxy1, proxy2 = MagicMock(spec=ClientProxy), MagicMock(spec=ClientProxy)
        aggregated_params, metrics = strategy.aggregate_fit(
            server_round=1,
            results=[(proxy1, res1), (proxy2, res2)],
            failures=[],
        )

        self.assertIsNotNone(aggregated_params)
        self.assertEqual(metrics["num_clients"], 2)
        self.assertEqual(metrics["total_examples"], 40)

        # (10 * 10 + 30 * 20) / 40 = (100 + 600) / 40 = 17.5
        agg_ndarrays = parameters_to_ndarrays(aggregated_params)
        expected = np.full((2, 2), 17.5, dtype=np.float32)
        np.testing.assert_allclose(agg_ndarrays[0], expected)


class TestClientDataLoading(unittest.TestCase):
    """Tests for partitioning, client loading, and client_fn generation."""

    def test_load_partitioned_dataloaders_iid(self):
        ds_cfg = SyntheticDatasetConfig(
            seed=42,
            n_samples=200,
            n_features=6,
            n_classes=2,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        part_cfg = PartitionConfig(n_clients=4, method="iid", seed=42)

        client_loaders, val_loader, test_loader = load_partitioned_dataloaders(
            dataset_config=ds_cfg,
            partition_config=part_cfg,
            batch_size=16,
        )

        self.assertEqual(len(client_loaders), 4)
        total_train_samples = sum(len(l.dataset) for l in client_loaders)
        self.assertEqual(total_train_samples, 140)  # 0.7 * 200
        self.assertEqual(len(val_loader.dataset), 30)  # 0.15 * 200
        self.assertEqual(len(test_loader.dataset), 30)  # 0.15 * 200

    def test_load_partitioned_dataloaders_dirichlet(self):
        ds_cfg = SyntheticDatasetConfig(
            seed=42,
            n_samples=200,
            n_features=6,
            n_classes=2,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        part_cfg = PartitionConfig(
            n_clients=3,
            method="dirichlet",
            alpha=0.5,
            seed=42,
            min_samples_per_client=10,
        )

        client_loaders, val_loader, test_loader = load_partitioned_dataloaders(
            dataset_config=ds_cfg,
            partition_config=part_cfg,
            batch_size=16,
        )

        self.assertEqual(len(client_loaders), 3)
        for cl in client_loaders:
            self.assertGreaterEqual(len(cl.dataset), 10)
        total_train = sum(len(l.dataset) for l in client_loaders)
        self.assertEqual(total_train, 140)

    def test_create_client_fn(self):
        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))
        loader0 = DataLoader(TensorDataset(x, y), batch_size=5)
        loader1 = DataLoader(TensorDataset(x, y), batch_size=5)

        model_cfg = MLPConfig(input_dim=4, output_dim=2, hidden_dims=[8])
        client_fn = create_client_fn([loader0, loader1], model_cfg)

        # Call with string cid
        client_obj = client_fn("0")
        self.assertIsNotNone(client_obj)

        # Call with mock Context object
        mock_ctx = MagicMock()
        mock_ctx.node_config = {"partition-id": "1"}
        client_obj_1 = client_fn(mock_ctx)
        self.assertIsNotNone(client_obj_1)


class TestFlowerSimulationSmoke(unittest.TestCase):
    """End-to-end smoke test verifying Flower Simulation completes cleanly on CPU."""

    def test_tiny_simulation_smoke(self):
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
            num_rounds=1,
            partition_method="iid",
            seed=42,
            batch_size=16,
            min_samples_per_client=5,
        )

        results = run_fedavg_simulation(
            fedavg_config=fedavg_cfg,
            dataset_config=ds_cfg,
            model_config=model_cfg,
        )

        self.assertIn("config", results)
        self.assertEqual(results["config"]["algorithm"], "FedAvg")
        self.assertEqual(results["config"]["n_clients"], 2)
        self.assertEqual(results["config"]["num_rounds"], 1)

        self.assertIn("history", results)
        self.assertEqual(len(results["history"]), 1)
        r1 = results["history"][0]
        self.assertEqual(r1["round"], 1)
        self.assertIn("val_loss", r1)
        self.assertIn("val_accuracy", r1)
        self.assertIn("test_loss", r1)
        self.assertIn("test_accuracy", r1)

        self.assertIn("test", results)
        self.assertIsNotNone(results["test"]["loss"])
        self.assertIsNotNone(results["test"]["accuracy"])
        self.assertTrue(0.0 <= results["test"]["accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
