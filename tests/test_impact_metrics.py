"""Unit tests for ground-truth impact calculation and validation loss evaluation."""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.impact_metrics import compute_action_impact, evaluate_model_loss


class DummyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestImpactMetrics(unittest.TestCase):
    """Test suite for impact metrics."""

    def test_impact_sign_and_harmful_label(self) -> None:
        """Verify positive impact on improvement, negative impact on worsening, and tolerance boundary."""
        # Case 1: Validation loss improved
        res_good = compute_action_impact(val_loss_before=1.5, val_loss_after=1.2, tolerance=1e-5)
        self.assertAlmostEqual(res_good["impact"], 0.3, places=5)
        self.assertFalse(res_good["is_harmful"])

        # Case 2: Validation loss worsened significantly
        res_bad = compute_action_impact(val_loss_before=1.0, val_loss_after=1.4, tolerance=1e-5)
        self.assertAlmostEqual(res_bad["impact"], -0.4, places=5)
        self.assertTrue(res_bad["is_harmful"])

        # Case 3: Loss worsened within numerical tolerance -> NOT labeled harmful
        res_tol_safe = compute_action_impact(
            val_loss_before=1.000000, val_loss_after=1.000005, tolerance=1e-5
        )
        self.assertAlmostEqual(res_tol_safe["impact"], -0.000005, places=6)
        self.assertFalse(res_tol_safe["is_harmful"])

        # Case 4: Loss worsened beyond tolerance -> labeled harmful
        res_tol_harm = compute_action_impact(
            val_loss_before=1.000000, val_loss_after=1.000020, tolerance=1e-5
        )
        self.assertAlmostEqual(res_tol_harm["impact"], -0.000020, places=6)
        self.assertTrue(res_tol_harm["is_harmful"])

    def test_evaluate_model_loss_does_not_mutate_state(self) -> None:
        """Verify loss evaluation preserves training state, weights, and is deterministic."""
        model = DummyClassifier()
        model.train()  # Model in training mode

        orig_params = [p.clone().detach() for p in model.parameters()]

        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=5)

        loss1 = evaluate_model_loss(model, loader)
        loss2 = evaluate_model_loss(model, loader)

        # Loss must be strictly deterministic
        self.assertEqual(loss1, loss2)

        # Parameters must remain completely unchanged
        for p_orig, p_now in zip(orig_params, model.parameters()):
            self.assertTrue(torch.equal(p_orig, p_now))

        # Model must have been restored to training mode
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
