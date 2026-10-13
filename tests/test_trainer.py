"""Unit tests for metrics and the centralized training/evaluation workflow."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.metrics import compute_accuracy, compute_mean
from src.model import MLPConfig, build_criterion, build_model, build_optimizer
from src.dataset import SyntheticDatasetConfig, get_synthetic_dataloaders
from src.trainer import (
    TrainConfig,
    evaluate,
    run,
    save_results,
    set_seed,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg():
    """Tiny dataset config for fast in-test runs (120 samples, 2 classes)."""
    return SyntheticDatasetConfig(
        seed=0,
        n_samples=120,
        n_features=8,
        n_classes=2,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )


def _tiny_model_cfg():
    return MLPConfig(input_dim=8, hidden_dims=[16], output_dim=2, optimizer="sgd", lr=0.01)


def _tiny_loaders(batch_size=32):
    return get_synthetic_dataloaders(_tiny_cfg(), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

class TestComputeAccuracy(unittest.TestCase):
    def test_all_correct(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        labels = torch.tensor([0, 1])
        self.assertAlmostEqual(compute_accuracy(logits, labels), 1.0)

    def test_all_wrong(self):
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        labels = torch.tensor([1, 1])
        self.assertAlmostEqual(compute_accuracy(logits, labels), 0.0)

    def test_half_correct(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        labels = torch.tensor([0, 0])
        self.assertAlmostEqual(compute_accuracy(logits, labels), 0.5)

    def test_returns_float(self):
        logits = torch.randn(4, 3)
        labels = torch.randint(0, 3, (4,))
        self.assertIsInstance(compute_accuracy(logits, labels), float)


class TestComputeMean(unittest.TestCase):
    def test_single_value(self):
        self.assertAlmostEqual(compute_mean([5.0]), 5.0)

    def test_multiple_values(self):
        self.assertAlmostEqual(compute_mean([1.0, 2.0, 3.0]), 2.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_mean([])


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

class TestTrainConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.n_epochs, 10)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.seed, 42)

    def test_from_json(self):
        cfg = TrainConfig.from_json(Path("configs/train_config.json"))
        self.assertGreaterEqual(cfg.n_epochs, 1)
        self.assertGreaterEqual(cfg.batch_size, 1)

    def test_invalid_epochs(self):
        with self.assertRaises(ValueError):
            TrainConfig(n_epochs=0)

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            TrainConfig(batch_size=0)


# ---------------------------------------------------------------------------
# set_seed reproducibility
# ---------------------------------------------------------------------------

class TestSetSeed(unittest.TestCase):
    def test_same_seed_same_tensor(self):
        set_seed(123)
        t1 = torch.randn(4)
        set_seed(123)
        t2 = torch.randn(4)
        self.assertTrue(torch.equal(t1, t2))

    def test_different_seeds_differ(self):
        set_seed(1)
        t1 = torch.randn(4)
        set_seed(2)
        t2 = torch.randn(4)
        self.assertFalse(torch.equal(t1, t2))


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

class TestTrainOneEpoch(unittest.TestCase):
    def setUp(self):
        self.model = build_model(_tiny_model_cfg())
        self.criterion = build_criterion()
        self.optimizer = build_optimizer(self.model, _tiny_model_cfg())
        self.loaders = _tiny_loaders()

    def test_returns_loss_key(self):
        result = train_one_epoch(
            self.model, self.loaders["train"], self.criterion, self.optimizer
        )
        self.assertIn("loss", result)

    def test_loss_is_positive_finite(self):
        result = train_one_epoch(
            self.model, self.loaders["train"], self.criterion, self.optimizer
        )
        self.assertGreater(result["loss"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))

    def test_loss_is_float(self):
        result = train_one_epoch(
            self.model, self.loaders["train"], self.criterion, self.optimizer
        )
        self.assertIsInstance(result["loss"], float)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.model = build_model(_tiny_model_cfg())
        self.criterion = build_criterion()
        self.loaders = _tiny_loaders()

    def test_returns_required_keys(self):
        result = evaluate(self.model, self.loaders["val"], self.criterion)
        self.assertIn("loss", result)
        self.assertIn("accuracy", result)

    def test_accuracy_in_range(self):
        result = evaluate(self.model, self.loaders["val"], self.criterion)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_loss_finite(self):
        result = evaluate(self.model, self.loaders["val"], self.criterion)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))


# ---------------------------------------------------------------------------
# run (full workflow, tiny config — 2 epochs)
# ---------------------------------------------------------------------------

class TestRun(unittest.TestCase):
    def setUp(self):
        self.train_cfg = TrainConfig(n_epochs=2, batch_size=32, seed=7)
        self.dataset_cfg = _tiny_cfg()
        self.model_cfg = _tiny_model_cfg()

    def test_returns_required_keys(self):
        metrics = run(self.train_cfg, self.dataset_cfg, self.model_cfg)
        self.assertIn("config", metrics)
        self.assertIn("history", metrics)
        self.assertIn("test", metrics)

    def test_history_length(self):
        metrics = run(self.train_cfg, self.dataset_cfg, self.model_cfg)
        self.assertEqual(len(metrics["history"]), 2)

    def test_history_epoch_keys(self):
        metrics = run(self.train_cfg, self.dataset_cfg, self.model_cfg)
        for entry in metrics["history"]:
            for key in ("epoch", "train_loss", "val_loss", "val_accuracy"):
                self.assertIn(key, entry)

    def test_test_keys(self):
        metrics = run(self.train_cfg, self.dataset_cfg, self.model_cfg)
        self.assertIn("loss", metrics["test"])
        self.assertIn("accuracy", metrics["test"])

    def test_reproducible(self):
        m1 = run(self.train_cfg, self.dataset_cfg, self.model_cfg)
        m2 = run(self.train_cfg, self.dataset_cfg, self.model_cfg)
        self.assertEqual(m1["test"]["accuracy"], m2["test"]["accuracy"])
        self.assertEqual(m1["test"]["loss"], m2["test"]["loss"])


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults(unittest.TestCase):
    def test_writes_valid_json(self):
        metrics = {"test": {"accuracy": 0.9, "loss": 0.25}, "history": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sub" / "metrics.json"
            save_results(metrics, out)
            self.assertTrue(out.exists())
            with open(out) as f:
                loaded = json.load(f)
            self.assertAlmostEqual(loaded["test"]["accuracy"], 0.9)

    def test_creates_parent_dirs(self):
        metrics = {"x": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "a" / "b" / "c.json"
            save_results(metrics, out)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
