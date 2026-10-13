"""Unit tests for IID and non-IID (Dirichlet) federated data partitioning."""

import unittest
from pathlib import Path

import numpy as np

from src.partition import (
    PartitionConfig,
    compute_partition_stats,
    partition_data,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_data(n_samples: int = 300, n_features: int = 8, n_classes: int = 4, seed: int = 0):
    """Return deterministic (x, y) numpy arrays."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # Balanced classes
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    y = np.concatenate([y, np.arange(n_samples % n_classes)]).astype(np.int64)
    rng.shuffle(y)
    return x, y


def _iid_cfg(**kwargs) -> PartitionConfig:
    defaults = dict(n_clients=5, method="iid", alpha=0.5, seed=42, min_samples_per_client=1)
    defaults.update(kwargs)
    return PartitionConfig(**defaults)


def _dir_cfg(**kwargs) -> PartitionConfig:
    defaults = dict(n_clients=5, method="dirichlet", alpha=0.5, seed=42, min_samples_per_client=1)
    defaults.update(kwargs)
    return PartitionConfig(**defaults)


# ---------------------------------------------------------------------------
# PartitionConfig tests
# ---------------------------------------------------------------------------

class TestPartitionConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PartitionConfig()
        self.assertEqual(cfg.n_clients, 5)
        self.assertEqual(cfg.method, "iid")
        self.assertAlmostEqual(cfg.alpha, 0.5)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.min_samples_per_client, 1)

    def test_from_json(self):
        cfg = PartitionConfig.from_json(Path("configs/partition_config.json"))
        self.assertGreater(cfg.n_clients, 0)
        self.assertIn(cfg.method, ("iid", "dirichlet"))

    def test_invalid_n_clients(self):
        with self.assertRaises(ValueError):
            PartitionConfig(n_clients=0)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            PartitionConfig(method="uniform")

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            PartitionConfig(alpha=0.0)
        with self.assertRaises(ValueError):
            PartitionConfig(alpha=-1.0)

    def test_invalid_min_samples(self):
        with self.assertRaises(ValueError):
            PartitionConfig(min_samples_per_client=0)


# ---------------------------------------------------------------------------
# IID partitioning
# ---------------------------------------------------------------------------

class TestIIDPartition(unittest.TestCase):
    def setUp(self):
        self.x, self.y = _make_data(n_samples=300, n_classes=4)
        self.cfg = _iid_cfg(n_clients=5)
        self.parts = partition_data(self.x, self.y, self.cfg)

    # --- client count ---

    def test_returns_correct_number_of_clients(self):
        self.assertEqual(len(self.parts), 5)

    # --- complete coverage, no duplicates ---

    def test_all_samples_covered(self):
        """Union of client indices must equal the full training set."""
        all_y = np.concatenate([y_c for _, y_c in self.parts])
        self.assertEqual(len(all_y), len(self.y))

    def test_no_duplicate_samples(self):
        """Concatenated feature rows must all be unique (no row shared)."""
        all_x = np.concatenate([x_c for x_c, _ in self.parts], axis=0)
        # Compare row-wise sorted unique count to total count
        unique_rows = np.unique(all_x, axis=0)
        self.assertEqual(len(unique_rows), len(self.x))

    def test_no_index_overlap(self):
        """Each x-row must appear in exactly one client partition."""
        # Track row presence via a set of rounded string hashes (fast enough)
        seen = set()
        for x_c, _ in self.parts:
            for row in x_c:
                key = row.tobytes()
                self.assertNotIn(key, seen, "Duplicate sample found across clients")
                seen.add(key)

    # --- sizes ---

    def test_sizes_nearly_equal(self):
        """IID sizes should differ by at most 1 sample."""
        sizes = [len(y_c) for _, y_c in self.parts]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_total_size_unchanged(self):
        total = sum(len(y_c) for _, y_c in self.parts)
        self.assertEqual(total, len(self.y))

    # --- reproducibility ---

    def test_same_seed_same_partition(self):
        parts2 = partition_data(self.x, self.y, _iid_cfg(n_clients=5, seed=42))
        for (x1, y1), (x2, y2) in zip(self.parts, parts2):
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)

    def test_different_seed_different_partition(self):
        parts2 = partition_data(self.x, self.y, _iid_cfg(n_clients=5, seed=999))
        # At least one client must differ
        any_diff = any(
            not np.array_equal(y1, y2)
            for (_, y1), (_, y2) in zip(self.parts, parts2)
        )
        self.assertTrue(any_diff)

    # --- min samples ---

    def test_min_samples_met(self):
        min_s = 20
        parts = partition_data(self.x, self.y, _iid_cfg(n_clients=5, min_samples_per_client=min_s))
        for _, y_c in parts:
            self.assertGreaterEqual(len(y_c), min_s)

    def test_min_samples_impossible_raises(self):
        with self.assertRaises(ValueError):
            # 5 clients × 1000 min > 300 samples
            partition_data(self.x, self.y, _iid_cfg(n_clients=5, min_samples_per_client=1000))


# ---------------------------------------------------------------------------
# Non-IID (Dirichlet) partitioning
# ---------------------------------------------------------------------------

class TestDirichletPartition(unittest.TestCase):
    def setUp(self):
        self.x, self.y = _make_data(n_samples=400, n_classes=4)
        self.cfg = _dir_cfg(n_clients=5, alpha=0.1)
        self.parts = partition_data(self.x, self.y, self.cfg)

    # --- client count ---

    def test_returns_correct_number_of_clients(self):
        self.assertEqual(len(self.parts), 5)

    # --- complete coverage, no duplicates ---

    def test_all_samples_covered(self):
        total = sum(len(y_c) for _, y_c in self.parts)
        self.assertEqual(total, len(self.y))

    def test_no_index_overlap(self):
        seen = set()
        for x_c, _ in self.parts:
            for row in x_c:
                key = row.tobytes()
                self.assertNotIn(key, seen, "Duplicate sample found across clients")
                seen.add(key)

    def test_total_size_unchanged(self):
        total = sum(len(y_c) for _, y_c in self.parts)
        self.assertEqual(total, len(self.y))

    # --- non-IID label distributions ---

    def test_label_distributions_are_non_uniform(self):
        """With alpha=0.1 (very skewed), clients should have very different
        dominant classes."""
        n_classes = 4
        dominant = []
        for _, y_c in self.parts:
            counts = np.bincount(y_c, minlength=n_classes)
            dominant.append(int(np.argmax(counts)))
        # Not all clients share the same dominant class
        self.assertGreater(len(set(dominant)), 1)

    def test_low_alpha_more_skewed_than_high_alpha(self):
        """Low alpha → higher per-client label concentration (lower entropy)."""
        x, y = _make_data(n_samples=600, n_classes=4)
        n_classes = 4

        def avg_max_fraction(parts):
            fracs = []
            for _, y_c in parts:
                counts = np.bincount(y_c, minlength=n_classes)
                fracs.append(counts.max() / len(y_c))
            return float(np.mean(fracs))

        low_alpha_parts = partition_data(x, y, _dir_cfg(n_clients=5, alpha=0.05, seed=7))
        high_alpha_parts = partition_data(x, y, _dir_cfg(n_clients=5, alpha=100.0, seed=7))

        self.assertGreater(avg_max_fraction(low_alpha_parts), avg_max_fraction(high_alpha_parts))

    def test_high_alpha_near_iid(self):
        """With very high alpha, each client's label distribution should be
        close to the global distribution (within 0.20 of each fraction)."""
        x, y = _make_data(n_samples=800, n_classes=4)
        n_classes = 4
        global_frac = np.bincount(y, minlength=n_classes) / len(y)

        parts = partition_data(x, y, _dir_cfg(n_clients=5, alpha=200.0, seed=0))
        for _, y_c in parts:
            client_frac = np.bincount(y_c, minlength=n_classes) / len(y_c)
            np.testing.assert_allclose(client_frac, global_frac, atol=0.20)

    # --- reproducibility ---

    def test_same_seed_same_partition(self):
        parts2 = partition_data(self.x, self.y, _dir_cfg(n_clients=5, alpha=0.1, seed=42))
        for (x1, y1), (x2, y2) in zip(self.parts, parts2):
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)

    def test_different_seed_different_partition(self):
        parts2 = partition_data(self.x, self.y, _dir_cfg(n_clients=5, alpha=0.1, seed=0))
        any_diff = any(
            not np.array_equal(y1, y2)
            for (_, y1), (_, y2) in zip(self.parts, parts2)
        )
        self.assertTrue(any_diff)

    # --- min samples ---

    def test_min_samples_met(self):
        min_s = 15
        parts = partition_data(
            self.x, self.y,
            _dir_cfg(n_clients=5, alpha=0.05, min_samples_per_client=min_s),
        )
        for _, y_c in parts:
            self.assertGreaterEqual(len(y_c), min_s)

    def test_min_samples_impossible_raises(self):
        with self.assertRaises(ValueError):
            partition_data(self.x, self.y, _dir_cfg(n_clients=5, min_samples_per_client=10_000))


# ---------------------------------------------------------------------------
# compute_partition_stats
# ---------------------------------------------------------------------------

class TestComputePartitionStats(unittest.TestCase):
    def setUp(self):
        self.x, self.y = _make_data(n_samples=200, n_classes=4)
        self.parts = partition_data(self.x, self.y, _iid_cfg(n_clients=4))
        self.stats = compute_partition_stats(self.parts, n_classes=4)

    def test_returns_one_entry_per_client(self):
        self.assertEqual(len(self.stats), 4)

    def test_required_keys(self):
        for s in self.stats:
            for key in ("client_id", "n_samples", "class_counts", "class_fractions"):
                self.assertIn(key, s)

    def test_n_samples_matches_partition(self):
        for s, (_, y_c) in zip(self.stats, self.parts):
            self.assertEqual(s["n_samples"], len(y_c))

    def test_class_counts_sum_to_n_samples(self):
        for s in self.stats:
            self.assertEqual(sum(s["class_counts"]), s["n_samples"])

    def test_class_fractions_sum_to_one(self):
        for s in self.stats:
            self.assertAlmostEqual(sum(s["class_fractions"]), 1.0, places=5)

    def test_class_counts_length(self):
        for s in self.stats:
            self.assertEqual(len(s["class_counts"]), 4)


if __name__ == "__main__":
    unittest.main()
