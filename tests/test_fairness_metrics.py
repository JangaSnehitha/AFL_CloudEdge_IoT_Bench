"""Unit tests for fairness, contribution equity, and per-class evaluation metrics."""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.fairness_metrics import (
    compile_fairness_summary,
    compute_jains_fairness_index,
    compute_label_exposure,
    compute_tier_shares,
    evaluate_per_class_held_out,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x):
        return self.fc(x)


class TestFairnessMetrics(unittest.TestCase):
    """Test suite for fairness metrics."""

    def test_jains_fairness_index_with_zero_contributors(self) -> None:
        """Verify Jain's index includes zero-contribution clients and handles all-zero edge case."""
        # 1. Perfectly equal contributions across 4 clients: J = 1.0
        j_eq, reason_eq = compute_jains_fairness_index([5, 5, 5, 5])
        self.assertAlmostEqual(j_eq, 1.0, places=5)
        self.assertIsNone(reason_eq)

        # 2. Complete monopoly by 1 client out of 4: J = 1 / 4 = 0.25
        j_mono, _ = compute_jains_fairness_index([10, 0, 0, 0])
        self.assertAlmostEqual(j_mono, 0.25, places=5)

        # 3. Intermediate skew: [10, 5, 5, 0]
        # sum = 20, sum^2 = 400. sum_sq = 100 + 25 + 25 + 0 = 150.
        # J = 400 / (4 * 150) = 400 / 600 = 0.666667
        j_skew, _ = compute_jains_fairness_index([10, 5, 5, 0])
        self.assertAlmostEqual(j_skew, 2.0 / 3.0, places=5)

        # 4. All zero contributions: must return None with explanation
        j_zero, reason_zero = compute_jains_fairness_index([0, 0, 0, 0])
        self.assertIsNone(j_zero)
        self.assertEqual(reason_zero, "zero_total_contributions")

        # 5. Empty population: must return None
        j_empty, reason_empty = compute_jains_fairness_index([])
        self.assertIsNone(j_empty)
        self.assertEqual(reason_empty, "empty_population")

    def test_compute_tier_shares(self) -> None:
        """Verify tier contribution shares sum to 1.0 and match client assignments."""
        applied = {0: 10, 1: 5, 2: 5}
        tiers = {0: "fast", 1: "medium", 2: "slow"}

        shares = compute_tier_shares(applied, tiers)
        self.assertAlmostEqual(shares["fast"], 0.5)
        self.assertAlmostEqual(shares["medium"], 0.25)
        self.assertAlmostEqual(shares["slow"], 0.25)

    def test_compute_label_exposure(self) -> None:
        """Verify label exposure calculation combines applied counts with partition histograms."""
        applied = {0: 2, 1: 1}
        # Client 0 has 10 samples of class 0, 5 samples of class 1
        # Client 1 has 0 samples of class 0, 20 samples of class 1
        label_counts = {
            0: {0: 10, 1: 5},
            1: {0: 0, 1: 20},
        }

        # Expected exposed:
        # Class 0: 2 * 10 + 1 * 0 = 20
        # Class 1: 2 * 5 + 1 * 20 = 30
        # Total exposed = 50
        # Fractions: Class 0: 20/50 = 0.4, Class 1: 30/50 = 0.6
        res = compute_label_exposure(applied, label_counts)
        self.assertEqual(res["total_exposed_samples"], 50)
        self.assertEqual(res["per_class_exposed_counts"]["0"], 20)
        self.assertEqual(res["per_class_exposed_counts"]["1"], 30)
        self.assertAlmostEqual(res["per_class_exposure_fractions"]["0"], 0.4)
        self.assertAlmostEqual(res["per_class_exposure_fractions"]["1"], 0.6)

    def test_evaluate_per_class_held_out(self) -> None:
        """Verify per-class accuracy, support counts, and handling of unrepresented classes."""
        model = DummyModel()
        # Set weights so prediction is argmax(x)
        with torch.no_grad():
            model.fc.weight.copy_(torch.eye(3))
            model.fc.bias.zero_()

        # Create dataset with classes 0 and 1, but NO samples of class 2
        # 3 correct for class 0, 1 incorrect for class 0
        # 2 correct for class 1
        x = torch.tensor([
            [10.0, 0.0, 0.0],  # pred 0, target 0 (correct)
            [10.0, 0.0, 0.0],  # pred 0, target 0 (correct)
            [10.0, 0.0, 0.0],  # pred 0, target 0 (correct)
            [0.0, 10.0, 0.0],  # pred 1, target 0 (incorrect)
            [0.0, 10.0, 0.0],  # pred 1, target 1 (correct)
            [0.0, 10.0, 0.0],  # pred 1, target 1 (correct)
        ])
        y = torch.tensor([0, 0, 0, 0, 1, 1])

        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        res = evaluate_per_class_held_out(model, loader, num_classes=3)

        # Class 0: 3 correct / 4 support = 0.75
        self.assertEqual(res["per_class_support"]["0"], 4)
        self.assertAlmostEqual(res["per_class_accuracy"]["0"], 0.75)

        # Class 1: 2 correct / 2 support = 1.00
        self.assertEqual(res["per_class_support"]["1"], 2)
        self.assertAlmostEqual(res["per_class_accuracy"]["1"], 1.00)

        # Class 2: 0 support -> must be None, NOT 0.0
        self.assertEqual(res["per_class_support"]["2"], 0)
        self.assertIsNone(res["per_class_accuracy"]["2"])

        # Macro accuracy across represented classes: (0.75 + 1.00) / 2 = 0.875
        self.assertAlmostEqual(res["macro_accuracy"], 0.875)
        # Worst class accuracy: 0.75 (class 0)
        self.assertAlmostEqual(res["worst_class_accuracy"], 0.75)
        self.assertEqual(res["worst_class_id"], 0)


if __name__ == "__main__":
    unittest.main()
