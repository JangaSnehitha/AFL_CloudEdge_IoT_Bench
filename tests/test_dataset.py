"""Unit tests for reproducible synthetic classification dataset."""

import unittest
from pathlib import Path

import numpy as np
import torch

from src.dataset import (
    SyntheticDatasetConfig,
    generate_synthetic_classification_data,
    get_synthetic_dataloaders,
    get_synthetic_datasets,
    split_data,
)


class TestSyntheticDataset(unittest.TestCase):
    def setUp(self):
        self.config = SyntheticDatasetConfig(
            seed=42,
            n_samples=600,
            n_features=16,
            n_classes=3,
            class_sep=1.5,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

    def test_config_from_json(self):
        config_path = Path("configs/dataset_config.json")
        self.assertTrue(config_path.exists(), "dataset_config.json should exist")
        cfg = SyntheticDatasetConfig.from_json(config_path)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.n_features, 20)
        self.assertEqual(cfg.n_classes, 4)

    def test_config_invalid_ratios(self):
        with self.assertRaises(ValueError):
            SyntheticDatasetConfig(train_ratio=0.8, val_ratio=0.2, test_ratio=0.1)

    def test_data_shapes_and_types(self):
        x, y = generate_synthetic_classification_data(self.config)
        self.assertEqual(x.shape, (600, 16))
        self.assertEqual(y.shape, (600,))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)

    def test_label_range_and_classes(self):
        _, y = generate_synthetic_classification_data(self.config)
        unique_labels = np.unique(y)
        self.assertEqual(len(unique_labels), 3)
        self.assertTrue(np.array_equal(unique_labels, np.array([0, 1, 2])))

    def test_reproducibility(self):
        x1, y1 = generate_synthetic_classification_data(self.config)
        x2, y2 = generate_synthetic_classification_data(self.config)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        config_diff = SyntheticDatasetConfig(
            seed=999,
            n_samples=600,
            n_features=16,
            n_classes=3,
        )
        x_diff, _ = generate_synthetic_classification_data(config_diff)
        self.assertFalse(np.array_equal(x1, x_diff))

    def test_split_proportions(self):
        x, y = generate_synthetic_classification_data(self.config)
        splits = split_data(x, y, self.config)

        n_train = len(splits["train"][0])
        n_val = len(splits["val"][0])
        n_test = len(splits["test"][0])

        self.assertEqual(n_train, int(600 * 0.7))
        self.assertEqual(n_val, int(600 * 0.15))
        self.assertEqual(n_train + n_val + n_test, 600)

    def test_torch_tensor_datasets(self):
        datasets = get_synthetic_datasets(self.config)
        self.assertIn("train", datasets)
        self.assertIn("val", datasets)
        self.assertIn("test", datasets)

        train_ds = datasets["train"]
        x_sample, y_sample = train_ds[0]

        self.assertIsInstance(x_sample, torch.Tensor)
        self.assertIsInstance(y_sample, torch.Tensor)
        self.assertEqual(x_sample.dtype, torch.float32)
        self.assertEqual(y_sample.dtype, torch.int64)
        self.assertEqual(x_sample.shape, torch.Size([16]))

        # Verify train standardization
        all_train_x = train_ds.tensors[0].numpy()
        mean = np.mean(all_train_x, axis=0)
        std = np.std(all_train_x, axis=0)
        np.testing.assert_allclose(mean, np.zeros(16), atol=1e-5)
        np.testing.assert_allclose(std, np.ones(16), atol=1e-5)

    def test_dataloaders(self):
        loaders = get_synthetic_dataloaders(self.config, batch_size=32)
        self.assertIn("train", loaders)
        self.assertIn("val", loaders)
        self.assertIn("test", loaders)

        batch_x, batch_y = next(iter(loaders["train"]))
        self.assertEqual(batch_x.shape[0], 32)
        self.assertEqual(batch_x.shape[1], 16)
        self.assertEqual(batch_y.shape[0], 32)


if __name__ == "__main__":
    unittest.main()
