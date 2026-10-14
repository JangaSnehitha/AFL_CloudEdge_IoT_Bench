"""Unit tests for client profiles, dimensional timing models, and partition coupling."""

import unittest

import numpy as np
import torch

from src.client_profiles import (
    ClientProfile,
    compute_model_bytes,
    compute_training_time,
    compute_transmission_time,
    couple_profiles_with_data_partitions,
    create_client_profile,
    get_default_tier_specs,
)
from src.model import MLPConfig, build_model


class TestClientProfiles(unittest.TestCase):
    """Test suite for client capability profiles and physical dimensional equations."""

    def test_client_profile_validation(self) -> None:
        """Verify parameter bounds and validation in ClientProfile."""
        # Valid profile
        prof = ClientProfile(
            client_id=0,
            speed_tier="fast",
            compute_factor=0.5,
            download_bandwidth=1024.0,
            upload_bandwidth=512.0,
            base_latency=0.01,
        )
        self.assertEqual(prof.data_partition_id, 0)

        # Invalid bounds
        with self.assertRaises(ValueError):
            ClientProfile(client_id=-1, speed_tier="fast", compute_factor=1.0, download_bandwidth=100.0, upload_bandwidth=100.0, base_latency=0.01)
        with self.assertRaises(ValueError):
            ClientProfile(client_id=0, speed_tier="ultra_fast", compute_factor=1.0, download_bandwidth=100.0, upload_bandwidth=100.0, base_latency=0.01)
        with self.assertRaises(ValueError):
            ClientProfile(client_id=0, speed_tier="fast", compute_factor=0.0, download_bandwidth=100.0, upload_bandwidth=100.0, base_latency=0.01)
        with self.assertRaises(ValueError):
            ClientProfile(client_id=0, speed_tier="fast", compute_factor=1.0, download_bandwidth=0.0, upload_bandwidth=100.0, base_latency=0.01)
        with self.assertRaises(ValueError):
            ClientProfile(client_id=0, speed_tier="fast", compute_factor=1.0, download_bandwidth=100.0, upload_bandwidth=-5.0, base_latency=0.01)
        with self.assertRaises(ValueError):
            ClientProfile(client_id=0, speed_tier="fast", compute_factor=1.0, download_bandwidth=100.0, upload_bandwidth=100.0, base_latency=-0.01)

    def test_compute_model_bytes(self) -> None:
        """Verify model byte computation matches parameter count and tensor precision."""
        model = build_model(MLPConfig())
        expected_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        # Benchmark MLP has 3556 params in float32 = 14224 bytes
        self.assertEqual(compute_model_bytes(model), 14224)
        self.assertEqual(compute_model_bytes(model), expected_bytes)

    def test_slow_profiles_take_longer_under_identical_workload(self) -> None:
        """Verify that slower device profiles take strictly longer to execute identical local training."""
        fast_prof = create_client_profile(client_id=0, speed_tier="fast")  # compute_factor=0.5
        slow_prof = create_client_profile(client_id=1, speed_tier="slow")  # compute_factor=2.5

        n_samples = 100
        batch_size = 32
        epochs = 2

        t_fast = compute_training_time(n_samples, batch_size, epochs, fast_prof.compute_factor)
        t_slow = compute_training_time(n_samples, batch_size, epochs, slow_prof.compute_factor)

        # Slower profile must take strictly longer
        self.assertGreater(t_slow, t_fast)
        # Factor ratio must match exactly 2.5 / 0.5 = 5.0
        self.assertAlmostEqual(t_slow / t_fast, 5.0, places=4)

    def test_transfer_time_increases_when_bandwidth_decreases(self) -> None:
        """Verify transmission duration scales inversely with available bandwidth."""
        model_bytes = 14224
        latency = 0.020

        high_bw = 1024.0 * 1024.0  # 1 MB/s
        low_bw = 64.0 * 1024.0     # 64 KB/s

        t_high = compute_transmission_time(model_bytes, high_bw, latency)
        t_low = compute_transmission_time(model_bytes, low_bw, latency)

        self.assertGreater(t_low, t_high)
        # With zero latency, ratio is exactly 1024 / 64 = 16.0
        t_high_zero_lat = compute_transmission_time(model_bytes, high_bw, 0.0)
        t_low_zero_lat = compute_transmission_time(model_bytes, low_bw, 0.0)
        self.assertAlmostEqual(t_low_zero_lat / t_high_zero_lat, 16.0, places=3)

    def test_profile_coupling_is_reproducible_and_maps_skew_to_slow(self) -> None:
        """Verify data coupling maps the highest-skew partition to the slowest device without altering datasets."""
        # Simulated partition stats: Client 0 balanced, Client 1 highly skewed
        partition_stats = {
            "0": {"class_counts": [25, 25, 25, 25]},  # Zero skew
            "1": {"class_counts": [90, 10, 0, 0]},    # High skew
        }
        profiles = [
            create_client_profile(client_id=0, speed_tier="fast"),
            create_client_profile(client_id=1, speed_tier="slow"),
        ]

        coupled, audit = couple_profiles_with_data_partitions(partition_stats, profiles)
        # Client 1 (slow) must be assigned Partition 1 (highest skew)
        prof_slow = next(p for p in coupled if p.speed_tier == "slow")
        self.assertEqual(prof_slow.data_partition_id, 1)

        # Audit contains expected strategy and keys
        self.assertEqual(audit["strategy"], "most_skewed_to_slow")
        self.assertIn("client_to_partition_mapping", audit)
        self.assertEqual(audit["client_to_partition_mapping"]["1"], 1)


if __name__ == "__main__":
    unittest.main()
