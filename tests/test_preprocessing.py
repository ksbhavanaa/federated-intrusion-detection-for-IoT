"""
tests/test_preprocessing.py
============================
Tests for the data preprocessing pipeline.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from pathlib import Path

from config import PATHS, LABEL_MAP, CLASS_NAMES


def test_preprocessed_files_exist():
    """Preprocessed numpy splits must exist after running preprocessor."""
    for fname in ["train_features.npy", "train_labels.npy",
                  "val_features.npy",   "val_labels.npy",
                  "test_features.npy",  "test_labels.npy",
                  "feature_names.csv"]:
        assert (PATHS.DATA / fname).exists(), f"Missing: {fname}"


def test_scaler_exists():
    assert PATHS.SCALER.exists(), "Scaler not saved."


def test_encoder_exists():
    assert PATHS.LABEL_ENCODER.exists(), "Label encoder not saved."


def test_train_test_no_overlap():
    """Train and test features must not be identical (no data leakage)."""
    X_train = np.load(PATHS.DATA / "train_features.npy")
    X_test  = np.load(PATHS.DATA / "test_features.npy")
    # Check that the first rows are not duplicated
    assert not np.allclose(X_train[0], X_test[0]), "Train row 0 == Test row 0 (possible leakage)"


def test_label_range():
    """All labels must be valid integer class indices."""
    import joblib
    encoder = joblib.load(PATHS.LABEL_ENCODER)
    n_classes = len(encoder.classes_)

    for split in ["train_labels.npy", "val_labels.npy", "test_labels.npy"]:
        y = np.load(PATHS.DATA / split)
        assert y.min() >= 0, f"{split}: negative label"
        assert y.max() < n_classes, f"{split}: label {y.max()} >= n_classes {n_classes}"


def test_label_map_has_target_classes():
    """Verify target classes are present in CLASS_NAMES."""
    assert len(CLASS_NAMES) == 6
    for cls in ["Normal", "DDoS", "Botnet", "PortScan", "BruteForce", "WebAttack"]:
        assert cls in CLASS_NAMES


def test_feature_count_consistent():
    """Feature count must be consistent across splits."""
    X_train = np.load(PATHS.DATA / "train_features.npy")
    X_val   = np.load(PATHS.DATA / "val_features.npy")
    X_test  = np.load(PATHS.DATA / "test_features.npy")
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]


def test_no_nan_in_splits():
    """No NaN values should be present in preprocessed data."""
    for fname in ["train_features.npy", "val_features.npy", "test_features.npy"]:
        X = np.load(PATHS.DATA / fname)
        assert not np.isnan(X).any(), f"NaN found in {fname}"


def test_no_inf_in_splits():
    """No infinite values should be present in preprocessed data."""
    for fname in ["train_features.npy", "val_features.npy", "test_features.npy"]:
        X = np.load(PATHS.DATA / fname)
        assert not np.isinf(X).any(), f"Inf found in {fname}"
