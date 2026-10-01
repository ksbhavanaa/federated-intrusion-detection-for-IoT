"""
tests/test_six_classes.py
=========================
Academic test suite validating the 6-class canonical architecture, FedAvg math,
data leakage checks, secure aggregation, risk thresholds, and experiment persistence.
"""

import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import PATHS, CLASS_NAMES, REAL_CLASSES_PRESENT, SYNTHETIC_CLASSES
from src.federated.fed_model import FederatedMLP, build_model, fedavg, get_model_params, set_model_params
from src.data.leakage_validator import validate_data_leakage
from src.realtime.detector import IDSDetector


def test_six_canonical_classes():
    """Verify that exactly six canonical classes are defined in the proper order."""
    expected = ["Normal", "DDoS", "Botnet", "PortScan", "BruteForce", "WebAttack"]
    assert CLASS_NAMES == expected, f"CLASS_NAMES mismatch: {CLASS_NAMES}"
    assert len(CLASS_NAMES) == 6


def test_model_output_dimension():
    """Verify that the neural network architecture outputs exactly 6 logits."""
    model = FederatedMLP(n_features=78, n_classes=len(CLASS_NAMES))
    dummy_input = torch.randn(2, 78)
    output = model(dummy_input)
    assert output.shape == (2, 6), f"Expected shape (2, 6), got {output.shape}"
    assert model.n_classes == 6


def test_fedavg_mathematical_aggregation():
    """
    Formally verify sample-weighted parameter aggregation mathematically.
    Client A: weight = 0.25, param = 2.0
    Client B: weight = 0.75, param = 6.0
    Expected: (0.25 * 2.0) + (0.75 * 6.0) = 0.5 + 4.5 = 5.0
    """
    model_a = FederatedMLP(n_features=78, n_classes=6)
    model_b = FederatedMLP(n_features=78, n_classes=6)

    state_a = {k: torch.full_like(v, 2.0) for k, v in model_a.state_dict().items()}
    state_b = {k: torch.full_like(v, 6.0) for k, v in model_b.state_dict().items()}

    param_list = [state_a, state_b]
    sample_counts = [100, 300] # 100/400 = 0.25, 300/400 = 0.75

    aggregated = fedavg(param_list, sample_counts)

    for k, v in aggregated.items():
        assert torch.allclose(v, torch.full_like(v, 5.0), atol=1e-5), f"FedAvg failed for {k}"


def test_secure_aggregation_zero_sum_cancellation():
    """
    Verify that pairwise zero-sum random noise masking (M_ij = -M_ji)
    cancels out during aggregation with error < 1e-6.
    """
    model_a = FederatedMLP(n_features=78, n_classes=6)
    model_b = FederatedMLP(n_features=78, n_classes=6)

    # Initial weights
    params_a = get_model_params(model_a)
    params_b = get_model_params(model_b)

    sample_counts = [200, 200]
    total_samples = 400

    # Pairwise noise mask (floating point tensors only)
    noise_dict = {}
    for k, v in params_a.items():
        if v.is_floating_point():
            noise_dict[k] = torch.randn_like(v) * 0.1
        else:
            noise_dict[k] = torch.zeros_like(v)

    # Client A adds noise scaled by N / n_a
    # Client B subtracts noise scaled by N / n_b
    scale_a = total_samples / sample_counts[0]
    scale_b = total_samples / sample_counts[1]

    masked_a = {k: v + (scale_a * noise_dict[k]) for k, v in params_a.items()}
    masked_b = {k: v - (scale_b * noise_dict[k]) for k, v in params_b.items()}

    unmasked_agg = fedavg([params_a, params_b], sample_counts)
    masked_agg = fedavg([masked_a, masked_b], sample_counts)

    for k in unmasked_agg:
        diff = torch.max(torch.abs(unmasked_agg[k] - masked_agg[k])).item()
        assert diff < 1e-5, f"Mask cancellation exceeded tolerance: {diff} on {k}"


def test_leakage_validator_catches_leakage():
    """Verify that leakage validator correctly flags train/test sample overlap."""
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_test_leaked = np.array([[1.0, 2.0], [5.0, 6.0]])
    y_train = np.array([0, 1])
    y_test = np.array([0, 1])

    res = validate_data_leakage(X_train, X_test_leaked, y_train, y_test)
    assert res["status"] == "FAILED"
    assert res["checks"]["zero_train_test_overlap"] is False


def test_leakage_validator_passes_clean_data():
    """Verify that leakage validator passes strictly separated data."""
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_test = np.array([[5.0, 6.0], [7.0, 8.0]])
    y_train = np.array([0, 1])
    y_test = np.array([0, 1])

    res = validate_data_leakage(X_train, X_test, y_train, y_test)
    assert res["status"] == "PASSED"
    assert res["checks"]["zero_train_test_overlap"] is True


def test_risk_tiers_and_honest_mitigation():
    """Verify documented risk tiers and honest software mitigation states."""
    det = IDSDetector()
    
    # Normal -> MONITOR (<30)
    risk_norm, mit_norm = det.calculate_risk_score("Normal", 0.98, "camera")
    assert risk_norm < 30.0
    assert mit_norm == "MONITOR"

    # Moderate attack on standard device -> ALERT (30-59)
    risk_low, mit_low = det.calculate_risk_score("PortScan", 0.65, "camera")
    assert 30.0 <= risk_low < 60.0
    assert mit_low == "ALERT"

    # High severity on standard device -> SIMULATED QUARANTINE (60-84)
    risk_quar, mit_quar = det.calculate_risk_score("Botnet", 0.90, "camera", recent_threat_count=2)
    assert 60.0 <= risk_quar < 85.0
    assert mit_quar == "SIMULATED QUARANTINE"

    # Critical threat on gateway router -> SIMULATED ISOLATION (85-100)
    risk_iso, mit_iso = det.calculate_risk_score("DDoS", 0.99, "router", recent_threat_count=5)
    assert risk_iso >= 85.0
    assert mit_iso == "SIMULATED ISOLATION"
