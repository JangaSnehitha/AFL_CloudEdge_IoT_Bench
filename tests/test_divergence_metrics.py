"""Unit tests for parameter divergence and update metrics."""

import unittest
import torch
import torch.nn as nn

from src.divergence_metrics import (
    aggregate_buffer_predictors,
    compute_all_predictors,
    compute_cosine_disagreement,
    compute_model_divergence,
    compute_relative_update_norm,
    compute_update_l2_norm,
    flatten_params,
)


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=True)


class TestDivergenceMetrics(unittest.TestCase):
    """Test suite for parameter divergence metrics."""

    def test_flatten_params(self) -> None:
        """Verify parameter flattening across nn.Module, tensor list, and dict."""
        model = SimpleLinear()
        with torch.no_grad():
            model.fc.weight.copy_(torch.tensor([[1.0, 2.0]]))
            model.fc.bias.copy_(torch.tensor([3.0]))

        flat_mod = flatten_params(model)
        self.assertEqual(flat_mod.shape, torch.Size([3]))
        self.assertTrue(torch.allclose(flat_mod, torch.tensor([1.0, 2.0, 3.0])))

        flat_list = flatten_params([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
        self.assertTrue(torch.allclose(flat_list, torch.tensor([1.0, 2.0, 3.0])))

        flat_dict = flatten_params({"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])})
        self.assertTrue(torch.allclose(flat_dict, torch.tensor([1.0, 2.0, 3.0])))

    def test_known_vector_norms_and_relative_divergence(self) -> None:
        """Verify exact Euclidean norm and relative divergence against analytical values."""
        base = torch.tensor([0.0, 0.0, 0.0])
        client = torch.tensor([3.0, 4.0, 0.0])  # norm = 5.0

        l2 = compute_update_l2_norm(client, base)
        self.assertAlmostEqual(l2, 5.0, places=5)

        # Base norm is 0, relative norm with eps=1e-8
        rel = compute_relative_update_norm(client, base, epsilon=1e-8)
        self.assertAlmostEqual(rel, 5.0 / 1e-8, places=-2)

        # Nonzero base
        base_nz = torch.tensor([10.0, 0.0, 0.0])  # base norm = 10.0
        rel_nz = compute_relative_update_norm(client, base_nz, epsilon=0.0)
        # client - base_nz = [-7.0, 4.0, 0.0], norm = sqrt(49 + 16) = sqrt(65) ~= 8.062258
        self.assertAlmostEqual(rel_nz, 8.062258 / 10.0, places=5)

        # Current-model divergence
        server = torch.tensor([0.0, 0.0, 0.0])
        div = compute_model_divergence(client, server, epsilon=1e-8)
        self.assertAlmostEqual(div, 5.0 / 1e-8, places=-2)

    def test_cosine_agreement_and_disagreement(self) -> None:
        """Verify cosine disagreement across parallel, orthogonal, opposite, and zero vectors."""
        # 1. Parallel: cos = 1.0 -> disagreement = 0.0
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([5.0, 0.0])
        cos_val, avail, reason = compute_cosine_disagreement(v1, v2)
        self.assertTrue(avail)
        self.assertAlmostEqual(cos_val, 0.0, places=5)
        self.assertEqual(reason, "ok")

        # 2. Orthogonal: cos = 0.0 -> disagreement = 1.0
        v_orth = torch.tensor([0.0, 2.0])
        cos_orth, avail_orth, _ = compute_cosine_disagreement(v1, v_orth)
        self.assertTrue(avail_orth)
        self.assertAlmostEqual(cos_orth, 1.0, places=5)

        # 3. Opposite: cos = -1.0 -> disagreement = 2.0
        v_opp = torch.tensor([-3.0, 0.0])
        cos_opp, avail_opp, _ = compute_cosine_disagreement(v1, v_opp)
        self.assertTrue(avail_opp)
        self.assertAlmostEqual(cos_opp, 2.0, places=5)

        # 4. First aggregation: prev is None -> undefined
        cos_none, avail_none, reason_none = compute_cosine_disagreement(v1, None)
        self.assertFalse(avail_none)
        self.assertIsNone(cos_none)
        self.assertEqual(reason_none, "first_aggregation_no_previous_update")

        # 5. Zero vector: norm < 1e-12 -> undefined
        v_zero = torch.tensor([0.0, 0.0])
        cos_z, avail_z, reason_z = compute_cosine_disagreement(v_zero, v1)
        self.assertFalse(avail_z)
        self.assertIsNone(cos_z)
        self.assertEqual(reason_z, "zero_norm_client_update")

    def test_compute_all_predictors(self) -> None:
        """Verify extraction of all predictors and negative staleness rejection."""
        base = torch.tensor([1.0, 1.0])
        client = torch.tensor([1.5, 2.0])
        server = torch.tensor([1.2, 1.4])
        prev_delta = torch.tensor([0.2, 0.4])

        preds = compute_all_predictors(
            client_params=client,
            base_params=base,
            server_params=server,
            prev_server_delta=prev_delta,
            client_train_version=2,
            server_version=5,
            local_loss=0.85,
        )

        self.assertEqual(preds["staleness"], 3)
        self.assertGreater(preds["update_l2_norm"], 0.0)
        self.assertGreater(preds["current_model_divergence"], 0.0)
        self.assertTrue(preds["cosine_available"])
        self.assertIsNotNone(preds["cosine_disagreement"])
        self.assertEqual(preds["local_loss"], 0.85)

        # Negative staleness must raise ValueError
        with self.assertRaises(ValueError):
            compute_all_predictors(client, base, server, client_train_version=5, server_version=2)

    def test_aggregate_buffer_predictors(self) -> None:
        """Verify FedBuff buffer-level predictor summaries."""
        m1 = {
            "staleness": 2,
            "update_l2_norm": 1.0,
            "relative_update_norm": 0.1,
            "current_model_divergence": 0.2,
        }
        m2 = {
            "staleness": 4,
            "update_l2_norm": 2.0,
            "relative_update_norm": 0.2,
            "current_model_divergence": 0.4,
        }
        joint_delta = torch.tensor([1.5, 0.0])
        prev_delta = torch.tensor([1.0, 0.0])

        buf_preds = aggregate_buffer_predictors([m1, m2], joint_delta, prev_delta)
        self.assertEqual(buf_preds["buffer_size"], 2)
        self.assertEqual(buf_preds["mean_staleness"], 3.0)
        self.assertEqual(buf_preds["max_staleness"], 4)
        self.assertEqual(buf_preds["min_staleness"], 2)
        self.assertEqual(buf_preds["mean_update_l2_norm"], 1.5)
        self.assertTrue(buf_preds["buffer_cosine_available"])
        self.assertAlmostEqual(buf_preds["buffer_cosine_disagreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
