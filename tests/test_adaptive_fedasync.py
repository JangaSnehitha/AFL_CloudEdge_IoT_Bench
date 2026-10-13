"""Unit tests for Adaptive Staleness-and-Heterogeneity-Aware FedAsync."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.adaptive_fedasync import (
    AdaptiveFedAsyncConfig,
    AdaptiveFedAsyncSimulator,
    compute_adaptive_weight,
    compute_base_weight,
    compute_total_variation_distance,
    run_adaptive_fedasync_simulation,
)


class TestAdaptiveFedAsync(unittest.TestCase):
    """Deterministic test suite for Adaptive FedAsync novelty method."""

    def test_1_correct_total_variation_divergence(self) -> None:
        """Requirement C1: Correct total-variation divergence for known distributions."""
        # Identical distributions -> divergence 0.0
        p_same1 = [0.25, 0.25, 0.25, 0.25]
        p_same2 = [0.25, 0.25, 0.25, 0.25]
        self.assertAlmostEqual(compute_total_variation_distance(p_same1, p_same2), 0.0)

        # Disjoint distributions -> divergence 1.0
        p_disjoint1 = [1.0, 0.0]
        p_disjoint2 = [0.0, 1.0]
        self.assertAlmostEqual(compute_total_variation_distance(p_disjoint1, p_disjoint2), 1.0)

        # Partial known divergence: 0.5 * (|0.6-0.4| + |0.4-0.6|) = 0.5 * 0.4 = 0.2
        p_skew1 = [0.6, 0.4]
        p_skew2 = [0.4, 0.6]
        self.assertAlmostEqual(compute_total_variation_distance(p_skew1, p_skew2), 0.2)

    def test_2_iid_produces_lower_divergence_than_skewed(self) -> None:
        """Requirement C2: IID-like client distribution produces lower divergence than a skewed client."""
        p_global = [0.25, 0.25, 0.25, 0.25]
        p_iid = [0.26, 0.24, 0.25, 0.25]
        p_skewed = [0.85, 0.05, 0.05, 0.05]

        d_iid = compute_total_variation_distance(p_iid, p_global)
        d_skewed = compute_total_variation_distance(p_skewed, p_global)

        self.assertLess(d_iid, d_skewed)
        self.assertAlmostEqual(d_iid, 0.01)
        self.assertAlmostEqual(d_skewed, 0.60)

    def test_3_client_first_update_has_zero_reliability_score(self) -> None:
        """Requirement C3: A client’s first update has reliability_score = 0."""
        cfg = AdaptiveFedAsyncConfig(n_clients=3, max_updates=3, seed=42)
        sim = AdaptiveFedAsyncSimulator(cfg)
        res = sim.run_simulation()

        seen_clients = set()
        for u in res["update_log"]:
            cid = u["client_id"]
            if cid not in seen_clients:
                self.assertEqual(
                    u["reliability_score_before_update"],
                    0.0,
                    f"First update for client {cid} must have reliability_score 0.0",
                )
                seen_clients.add(cid)

    def test_4_reliability_score_reflects_only_earlier_delays(self) -> None:
        """Requirement C4: A client’s reliability score reflects only earlier delays, not current/future."""
        cfg = AdaptiveFedAsyncConfig(
            n_clients=2,
            max_updates=6,
            seed=42,
            delay_distribution="uniform",
            max_delay=4,
        )
        sim = AdaptiveFedAsyncSimulator(cfg)
        res = sim.run_simulation()

        client_delays = {0: [], 1: []}
        for u in res["update_log"]:
            cid = u["client_id"]
            rel_score = u["reliability_score_before_update"]

            # Score before update must match mean of past delays divided by max_delay
            if len(client_delays[cid]) == 0:
                expected_score = 0.0
            else:
                expected_score = float(np.mean(client_delays[cid])) / 4.0

            self.assertAlmostEqual(rel_score, expected_score, places=5)
            # Add current delay after verifying pre-update score
            client_delays[cid].append(u["delay"])

    def test_5_greater_staleness_decreases_applied_weight(self) -> None:
        """Requirement C5: Greater staleness decreases applied weight when other factors match."""
        het = 0.3
        rel = 0.2

        bw_fresh = compute_base_weight(staleness=0, alpha_0=1.0)
        bw_stale = compute_base_weight(staleness=4, alpha_0=1.0)

        _, w_fresh = compute_adaptive_weight(
            bw_fresh, heterogeneity_score=het, reliability_score=rel, min_weight=0.01
        )
        _, w_stale = compute_adaptive_weight(
            bw_stale, heterogeneity_score=het, reliability_score=rel, min_weight=0.01
        )

        self.assertGreater(w_fresh, w_stale)

    def test_6_greater_divergence_decreases_applied_weight(self) -> None:
        """Requirement C6: Greater divergence decreases applied weight when other factors match."""
        base_w = compute_base_weight(staleness=1, alpha_0=1.0)
        rel = 0.2

        _, w_low_div = compute_adaptive_weight(
            base_w, heterogeneity_score=0.1, reliability_score=rel, min_weight=0.01
        )
        _, w_high_div = compute_adaptive_weight(
            base_w, heterogeneity_score=0.8, reliability_score=rel, min_weight=0.01
        )

        self.assertGreater(w_low_div, w_high_div)

    def test_7_greater_prior_delay_decreases_applied_weight(self) -> None:
        """Requirement C7: Greater prior delay/reliability score decreases applied weight."""
        base_w = compute_base_weight(staleness=1, alpha_0=1.0)
        het = 0.4

        _, w_low_delay = compute_adaptive_weight(
            base_w, heterogeneity_score=het, reliability_score=0.1, min_weight=0.01
        )
        _, w_high_delay = compute_adaptive_weight(
            base_w, heterogeneity_score=het, reliability_score=0.9, min_weight=0.01
        )

        self.assertGreater(w_low_delay, w_high_delay)

    def test_8_min_weight_clipping_works(self) -> None:
        """Requirement C8: min_weight clipping works."""
        # Extremely small raw weight should be clipped to min_weight
        tiny_base = 0.0001
        raw_w, applied_w = compute_adaptive_weight(
            tiny_base,
            heterogeneity_score=1.0,
            reliability_score=1.0,
            lambda_divergence=2.0,
            lambda_reliability=2.0,
            min_weight=0.05,
        )
        self.assertLess(raw_w, 0.05)
        self.assertEqual(applied_w, 0.05)

        # Upper clipping at 1.0
        large_base = 2.0
        _, applied_upper = compute_adaptive_weight(
            large_base,
            heterogeneity_score=0.0,
            reliability_score=0.0,
            min_weight=0.01,
        )
        self.assertEqual(applied_upper, 1.0)

    def test_9_fixed_seed_produces_identical_logs(self) -> None:
        """Requirement C9: Fixed seed produces identical event and update logs."""
        cfg1 = AdaptiveFedAsyncConfig(n_clients=3, max_updates=6, seed=123)
        cfg2 = AdaptiveFedAsyncConfig(n_clients=3, max_updates=6, seed=123)

        res1 = AdaptiveFedAsyncSimulator(cfg1).run_simulation()
        res2 = AdaptiveFedAsyncSimulator(cfg2).run_simulation()

        self.assertEqual(len(res1["event_log"]), len(res2["event_log"]))
        for e1, e2 in zip(res1["event_log"], res2["event_log"]):
            self.assertEqual(e1["time"], e2["time"])
            self.assertEqual(e1["event_type"], e2["event_type"])
            self.assertEqual(e1["client_id"], e2["client_id"])

        self.assertEqual(len(res1["update_log"]), len(res2["update_log"]))
        for u1, u2 in zip(res1["update_log"], res2["update_log"]):
            self.assertEqual(u1["client_id"], u2["client_id"])
            self.assertEqual(u1["staleness"], u2["staleness"])
            self.assertEqual(u1["heterogeneity_score"], u2["heterogeneity_score"])
            self.assertEqual(u1["reliability_score_before_update"], u2["reliability_score_before_update"])
            self.assertAlmostEqual(u1["applied_weight"], u2["applied_weight"], places=6)
            self.assertAlmostEqual(u1["test_accuracy"], u2["test_accuracy"], places=5)

    def test_10_parameter_validations(self) -> None:
        """Validate bounds and errors on hyperparameters."""
        with self.assertRaises(ValueError):
            AdaptiveFedAsyncConfig(alpha_0=0.0)
        with self.assertRaises(ValueError):
            AdaptiveFedAsyncConfig(lambda_divergence=-0.1)
        with self.assertRaises(ValueError):
            AdaptiveFedAsyncConfig(lambda_reliability=-0.1)
        with self.assertRaises(ValueError):
            AdaptiveFedAsyncConfig(min_weight=0.0)
        with self.assertRaises(ValueError):
            AdaptiveFedAsyncConfig(min_weight=1.5)


if __name__ == "__main__":
    unittest.main()
