"""
tests/test_federated.py
========================
Tests for the FedAvg implementation.
VERIFIES that global parameters are actually weighted averages of client parameters.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np
from src.federated.fed_model import FederatedMLP, fedavg, get_model_params, set_model_params


@pytest.fixture
def small_model():
    return FederatedMLP(n_features=10, n_classes=3, hidden_layers=[16, 8], dropout=0.0)


def test_fedavg_weighted_average_is_correct():
    """
    Core test: FedAvg MUST produce the exact weighted average of client parameters.

    Creates two clients with distinct known parameters.
    Manually computes expected weighted average.
    Asserts that fedavg() output matches the manual computation.
    """
    n_features, n_classes = 4, 2
    hidden = [8]

    client1 = FederatedMLP(n_features, n_classes, hidden_layers=hidden, dropout=0.0)
    client2 = FederatedMLP(n_features, n_classes, hidden_layers=hidden, dropout=0.0)

    # Set distinct, known parameters
    torch.manual_seed(1)
    for p in client1.parameters():
        p.data.fill_(1.0)    # all 1s

    torch.manual_seed(2)
    for p in client2.parameters():
        p.data.fill_(3.0)    # all 3s

    n1, n2 = 100, 300    # n2 has 3x more samples

    params1 = get_model_params(client1)
    params2 = get_model_params(client2)

    aggregated = fedavg([params1, params2], [n1, n2])

    # Manual expected: (100*1 + 300*3) / (100+300) = 1000/400 = 2.5
    expected_value = (n1 * 1.0 + n2 * 3.0) / (n1 + n2)

    for key, tensor in aggregated.items():
        # Skip BatchNorm running stats — they are buffers initialized to 0
        # regardless of weight values, and don't participate in FedAvg
        if 'running_mean' in key or 'running_var' in key or 'num_batches' in key:
            continue
        if tensor.is_floating_point():
            np.testing.assert_allclose(
                tensor.numpy(),
                np.full_like(tensor.numpy(), expected_value),
                rtol=1e-5, atol=1e-6,
                err_msg=f"FedAvg incorrect for key '{key}'. Expected {expected_value:.4f}"
            )


def test_fedavg_equal_weights():
    """With equal sample sizes, FedAvg must produce simple mean."""
    n_features, n_classes = 4, 2
    hidden = [8]

    c1 = FederatedMLP(n_features, n_classes, hidden_layers=hidden, dropout=0.0)
    c2 = FederatedMLP(n_features, n_classes, hidden_layers=hidden, dropout=0.0)

    for p in c1.parameters(): p.data.fill_(2.0)
    for p in c2.parameters(): p.data.fill_(4.0)

    aggregated = fedavg([get_model_params(c1), get_model_params(c2)], [50, 50])

    for key, tensor in aggregated.items():
        if 'running_mean' in key or 'running_var' in key or 'num_batches' in key:
            continue
        if tensor.is_floating_point():
            np.testing.assert_allclose(
                tensor.numpy(),
                np.full_like(tensor.numpy(), 3.0),   # (2+4)/2 = 3
                rtol=1e-5, atol=1e-6
            )


def test_fedavg_changes_model_parameters():
    """After FedAvg, a new model loaded with aggregated params must differ from initial params."""
    n_features, n_classes = 10, 3
    model_a = FederatedMLP(n_features, n_classes, hidden_layers=[16], dropout=0.0)
    model_b = FederatedMLP(n_features, n_classes, hidden_layers=[16], dropout=0.0)
    model_g = FederatedMLP(n_features, n_classes, hidden_layers=[16], dropout=0.0)

    torch.manual_seed(10)
    for p in model_a.parameters(): torch.nn.init.uniform_(p, -1, 0)
    torch.manual_seed(20)
    for p in model_b.parameters(): torch.nn.init.uniform_(p, 0, 1)

    agg = fedavg([get_model_params(model_a), get_model_params(model_b)], [100, 200])
    set_model_params(model_g, agg)

    # Global model params != either client
    g_params = get_model_params(model_g)
    a_params = get_model_params(model_a)
    b_params = get_model_params(model_b)

    for key in g_params:
        if 'running_mean' in key or 'running_var' in key or 'num_batches' in key:
            continue
        if g_params[key].is_floating_point():
            assert not torch.allclose(g_params[key], a_params[key], atol=1e-3), \
                f"Global == Client A for key {key}"
            assert not torch.allclose(g_params[key], b_params[key], atol=1e-3), \
                f"Global == Client B for key {key}"


def test_fedavg_zero_samples_raises():
    """FedAvg should raise ValueError if total samples is 0."""
    m = FederatedMLP(4, 2, hidden_layers=[8], dropout=0.0)
    with pytest.raises(ValueError, match="[Tt]otal"):
        fedavg([get_model_params(m)], [0])


def test_fedavg_empty_raises():
    """FedAvg should raise ValueError on empty input."""
    with pytest.raises(ValueError):
        fedavg([], [])


def test_model_forward_correct_shape(small_model):
    """Model output shape must match (batch, n_classes)."""
    X = torch.randn(16, 10)
    out = small_model(X)
    assert out.shape == (16, 3)


def test_model_predict_proba_sums_to_one(small_model):
    """Predict_proba output must sum to 1 per sample."""
    X = torch.randn(8, 10)
    proba = small_model.predict_proba(X)
    sums = proba.sum(dim=1)
    np.testing.assert_allclose(sums.numpy(), np.ones(8), rtol=1e-5)


def test_set_get_params_roundtrip(small_model):
    """get_model_params → set_model_params must reproduce identical tensors."""
    params = get_model_params(small_model)
    model2 = FederatedMLP(n_features=10, n_classes=3, hidden_layers=[16, 8], dropout=0.0)
    set_model_params(model2, params)
    params2 = get_model_params(model2)

    for key in params:
        torch.testing.assert_close(params[key], params2[key])
