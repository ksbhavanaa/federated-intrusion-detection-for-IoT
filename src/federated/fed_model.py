"""
src/federated/fed_model.py
==========================
Shared PyTorch MLP architecture for Federated Learning.

This SAME architecture is used by:
 - Every IoT client (local training)
 - The federated server (global model)
 - The centralized neural network baseline

Having one shared architecture ensures FedAvg parameter shapes always match.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FEDERATED


class FederatedMLP(nn.Module):
    """
    Multi-Layer Perceptron for intrusion detection.

    Architecture:
        Input(n_features) → Linear(256) → BN → ReLU → Dropout
                          → Linear(128) → BN → ReLU → Dropout
                          → Linear(64)  → BN → ReLU → Dropout
                          → Linear(n_classes)  [logits]

    BatchNorm is applied before activation for stable training.
    Dropout reduces overfitting on imbalanced attack data.
    """

    def __init__(self, n_features: int, n_classes: int,
                 hidden_layers: list = None, dropout: float = None):
        super(FederatedMLP, self).__init__()

        hidden = hidden_layers or FEDERATED.HIDDEN_LAYERS
        drop   = dropout if dropout is not None else FEDERATED.DROPOUT

        layers = []
        in_dim = n_features
        for out_dim in hidden:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(drop),
            ]
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, n_classes))
        self.network = nn.Sequential(*layers)

        # Store for reference
        self.n_features = n_features
        self.n_classes  = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        return self.predict_proba(x).argmax(dim=1)


def build_model(n_features: int, n_classes: int) -> FederatedMLP:
    """Convenience factory: create an untrained FederatedMLP."""
    return FederatedMLP(n_features=n_features, n_classes=n_classes)


def get_model_params(model: FederatedMLP) -> dict:
    """
    Return a DEEP COPY of model's state_dict (parameter tensors).
    Used by clients to 'send' parameters to the server.
    """
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def set_model_params(model: FederatedMLP, params: dict):
    """
    Load a parameter dict into the model in-place.
    Used by server to distribute global model to clients.
    """
    model.load_state_dict(params)


def fedavg(param_list: list, n_samples_list: list) -> dict:
    """
    TRUE Federated Averaging (McMahan et al., 2017).

    Computes the weighted average of model parameters:
        W_global = Σ(n_k × W_k) / Σ(n_k)

    where:
        W_k       = model parameters from client k (state_dict)
        n_k       = number of training samples at client k
        W_global  = aggregated global parameters

    Args:
        param_list    : list of state_dicts (one per client)
        n_samples_list: list of ints (training sample count per client)

    Returns:
        Aggregated state_dict (same keys/shapes as inputs)

    Raises:
        ValueError: if param_list is empty or lengths don't match
    """
    if len(param_list) == 0:
        raise ValueError("param_list cannot be empty")
    if len(param_list) != len(n_samples_list):
        raise ValueError("param_list and n_samples_list must have the same length")

    total_samples = sum(n_samples_list)
    if total_samples == 0:
        raise ValueError("Total training samples is zero")

    # Initialize aggregated params with zeros (same shape as first client)
    global_params = {}
    for key in param_list[0].keys():
        global_params[key] = torch.zeros_like(param_list[0][key], dtype=torch.float32)

    # Weighted accumulation: W_global += (n_k / N) * W_k
    for client_params, n_k in zip(param_list, n_samples_list):
        weight = n_k / total_samples
        for key in global_params:
            global_params[key] += weight * client_params[key].float()

    return global_params
