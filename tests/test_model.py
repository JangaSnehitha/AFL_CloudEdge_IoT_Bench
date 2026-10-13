"""Unit tests for the centralized MLP baseline."""

import copy
import unittest
from pathlib import Path

import torch

from src.model import (
    MLPConfig,
    MLP,
    build_criterion,
    build_model,
    build_optimizer,
)


class TestMLPConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MLPConfig()
        self.assertEqual(cfg.input_dim, 20)
        self.assertEqual(cfg.hidden_dims, [64, 32])
        self.assertEqual(cfg.output_dim, 4)

    def test_from_json(self):
        cfg = MLPConfig.from_json(Path("configs/model_config.json"))
        self.assertEqual(cfg.input_dim, 20)
        self.assertEqual(cfg.output_dim, 4)
        self.assertIn(cfg.optimizer, ("sgd", "adam"))

    def test_invalid_input_dim(self):
        with self.assertRaises(ValueError):
            MLPConfig(input_dim=0)

    def test_invalid_output_dim(self):
        with self.assertRaises(ValueError):
            MLPConfig(output_dim=1)

    def test_invalid_optimizer(self):
        with self.assertRaises(ValueError):
            MLPConfig(optimizer="rmsprop")


class TestMLPConstruction(unittest.TestCase):
    def setUp(self):
        self.cfg = MLPConfig(input_dim=16, hidden_dims=[32, 16], output_dim=3)
        self.model = build_model(self.cfg)

    def test_is_nn_module(self):
        self.assertIsInstance(self.model, torch.nn.Module)

    def test_parameter_count_positive(self):
        total = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total, 0)

    def test_no_dropout_layers_when_zero(self):
        # dropout=0.0 → no Dropout modules
        dropout_layers = [
            m for m in self.model.modules() if isinstance(m, torch.nn.Dropout)
        ]
        self.assertEqual(len(dropout_layers), 0)

    def test_dropout_layers_present_when_nonzero(self):
        cfg = MLPConfig(input_dim=16, hidden_dims=[32], output_dim=3, dropout=0.5)
        model = build_model(cfg)
        dropout_layers = [
            m for m in model.modules() if isinstance(m, torch.nn.Dropout)
        ]
        self.assertEqual(len(dropout_layers), 1)


class TestMLPForwardPass(unittest.TestCase):
    def setUp(self):
        self.cfg = MLPConfig(input_dim=20, hidden_dims=[64, 32], output_dim=4)
        self.model = build_model(self.cfg)
        self.model.eval()

    def test_output_shape_single_sample(self):
        x = torch.randn(1, 20)
        logits = self.model(x)
        self.assertEqual(logits.shape, torch.Size([1, 4]))

    def test_output_shape_batch(self):
        x = torch.randn(32, 20)
        logits = self.model(x)
        self.assertEqual(logits.shape, torch.Size([32, 4]))

    def test_output_is_float32(self):
        x = torch.randn(8, 20)
        logits = self.model(x)
        self.assertEqual(logits.dtype, torch.float32)

    def test_no_nan_in_output(self):
        x = torch.randn(16, 20)
        logits = self.model(x)
        self.assertFalse(torch.isnan(logits).any())


class TestLossComputation(unittest.TestCase):
    def setUp(self):
        self.cfg = MLPConfig(input_dim=20, hidden_dims=[64, 32], output_dim=4)
        self.model = build_model(self.cfg)
        self.criterion = build_criterion()

    def test_loss_is_scalar(self):
        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_is_positive(self):
        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.assertGreater(loss.item(), 0.0)

    def test_loss_is_finite(self):
        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.assertTrue(torch.isfinite(loss))


class TestOptimizerStep(unittest.TestCase):
    def setUp(self):
        self.cfg = MLPConfig(input_dim=20, hidden_dims=[64, 32], output_dim=4,
                             optimizer="sgd", lr=0.01)
        self.model = build_model(self.cfg)
        self.criterion = build_criterion()
        self.optimizer = build_optimizer(self.model, self.cfg)

    def test_parameters_change_after_step(self):
        params_before = [p.clone().detach() for p in self.model.parameters()]

        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(x), y)
        loss.backward()
        self.optimizer.step()

        params_after = list(self.model.parameters())
        changed = any(
            not torch.equal(pb, pa)
            for pb, pa in zip(params_before, params_after)
        )
        self.assertTrue(changed, "At least one parameter must change after optimizer step")

    def test_adam_optimizer_builds(self):
        cfg = MLPConfig(optimizer="adam", lr=0.001)
        model = build_model(cfg)
        opt = build_optimizer(model, cfg)
        self.assertIsInstance(opt, torch.optim.Adam)

    def test_sgd_optimizer_builds(self):
        opt = build_optimizer(self.model, self.cfg)
        self.assertIsInstance(opt, torch.optim.SGD)


if __name__ == "__main__":
    unittest.main()
