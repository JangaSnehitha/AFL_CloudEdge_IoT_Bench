"""Unit tests for predictor ranking, correlation, and action table analysis."""

import unittest
import tempfile
from pathlib import Path

from src.predictor_analysis import (
    analyze_action_observations,
    compute_auroc,
    compute_average_precision,
    compute_fractional_ranks,
    compute_spearman_correlation,
    export_actions_to_csv,
)


class TestPredictorAnalysis(unittest.TestCase):
    """Test suite for predictor analysis."""

    def test_fractional_ranks_with_ties(self) -> None:
        """Verify 1-based fractional rank calculation for distinct values and ties."""
        # Distinct
        ranks_d = compute_fractional_ranks([10.0, 20.0, 30.0])
        self.assertEqual(list(ranks_d), [1.0, 2.0, 3.0])

        # Tied middle values: 2nd and 3rd share rank 2.5
        ranks_t = compute_fractional_ranks([10.0, 20.0, 20.0, 30.0])
        self.assertEqual(list(ranks_t), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_correlation(self) -> None:
        """Verify Spearman rank correlation for monotonic, inverse, constant, and small data."""
        # Perfect inverse correlation (hypothesized: larger predictor -> smaller impact)
        x_inv = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_inv = [5.0, 4.0, 3.0, 2.0, 1.0]
        corr_inv, reason_inv = compute_spearman_correlation(x_inv, y_inv)
        self.assertAlmostEqual(corr_inv, -1.0, places=5)
        self.assertIsNone(reason_inv)

        # Perfect positive correlation
        corr_pos, _ = compute_spearman_correlation(x_inv, x_inv)
        self.assertAlmostEqual(corr_pos, 1.0, places=5)

        # Constant predictor
        x_const = [2.0, 2.0, 2.0, 2.0]
        corr_c, reason_c = compute_spearman_correlation(x_const, y_inv[:4])
        self.assertIsNone(corr_c)
        self.assertEqual(reason_c, "constant_predictor")

        # Insufficient observations (< 3)
        corr_small, reason_small = compute_spearman_correlation([1.0, 2.0], [2.0, 1.0])
        self.assertIsNone(corr_small)
        self.assertEqual(reason_small, "insufficient_observations")

    def test_auroc_and_average_precision(self) -> None:
        """Verify AUROC and AP under perfect discrimination, inverted scores, ties, and edge cases."""
        # Perfect separation: high score -> label 1
        scores_perf = [0.1, 0.2, 0.8, 0.9]
        labels_perf = [0, 0, 1, 1]

        auc_perf, _ = compute_auroc(scores_perf, labels_perf)
        self.assertAlmostEqual(auc_perf, 1.0, places=5)

        ap_perf, _ = compute_average_precision(scores_perf, labels_perf)
        self.assertAlmostEqual(ap_perf, 1.0, places=5)

        # Inverted separation: high score -> label 0
        scores_inv = [0.9, 0.8, 0.2, 0.1]
        auc_inv, _ = compute_auroc(scores_inv, labels_perf)
        self.assertAlmostEqual(auc_inv, 0.0, places=5)

        # Single class outcome (all zeros or all ones) -> must return None
        auc_single, reason_single = compute_auroc([0.1, 0.2, 0.3], [0, 0, 0])
        self.assertIsNone(auc_single)
        self.assertEqual(reason_single, "single_class_outcome")

        ap_single, ap_reason_single = compute_average_precision([0.1, 0.2, 0.3], [0, 0, 0])
        self.assertIsNone(ap_single)
        self.assertEqual(ap_reason_single, "single_class_outcome")

        # Constant predictor with mixed labels
        auc_const, reason_const = compute_auroc([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1])
        self.assertIsNone(auc_const)
        self.assertEqual(reason_const, "constant_predictor")

    def test_action_table_export_and_analysis(self) -> None:
        """Verify export_actions_to_csv and analyze_action_observations."""
        actions = [
            {
                "action_id": 1,
                "scenario": "S4",
                "algorithm": "FedAsync",
                "seed": 42,
                "virtual_time": 0.1,
                "client_id": 0,
                "staleness": 0,
                "update_l2_norm": 0.5,
                "relative_update_norm": 0.05,
                "current_model_divergence": 0.05,
                "cosine_disagreement": None,
                "cosine_available": False,
                "local_loss": 1.2,
                "val_loss_before": 1.5,
                "val_loss_after": 1.4,
                "impact": 0.1,
                "is_harmful": False,
            },
            {
                "action_id": 2,
                "scenario": "S4",
                "algorithm": "FedAsync",
                "seed": 42,
                "virtual_time": 0.2,
                "client_id": 1,
                "staleness": 3,
                "update_l2_norm": 1.5,
                "relative_update_norm": 0.15,
                "current_model_divergence": 0.25,
                "cosine_disagreement": 0.8,
                "cosine_available": True,
                "local_loss": 1.8,
                "val_loss_before": 1.4,
                "val_loss_after": 1.6,
                "impact": -0.2,
                "is_harmful": True,
            },
            {
                "action_id": 3,
                "scenario": "S4",
                "algorithm": "FedAsync",
                "seed": 42,
                "virtual_time": 0.3,
                "client_id": 2,
                "staleness": 1,
                "update_l2_norm": 0.8,
                "relative_update_norm": 0.08,
                "current_model_divergence": 0.10,
                "cosine_disagreement": 0.3,
                "cosine_available": True,
                "local_loss": 1.3,
                "val_loss_before": 1.6,
                "val_loss_after": 1.55,
                "impact": 0.05,
                "is_harmful": False,
            },
        ]

        analysis = analyze_action_observations(actions)
        self.assertEqual(analysis["action_count"], 3)
        self.assertEqual(analysis["harmful_count"], 1)
        self.assertAlmostEqual(analysis["harmful_prevalence"], 1.0 / 3.0, places=5)
        self.assertIn("staleness", analysis["predictors"])
        self.assertIn("current_model_divergence", analysis["predictors"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_actions.csv"
            export_actions_to_csv(actions, csv_path)
            self.assertTrue(csv_path.exists())
            lines = csv_path.read_text().strip().split("\n")
            self.assertEqual(len(lines), 4)  # Header + 3 rows


if __name__ == "__main__":
    unittest.main()
