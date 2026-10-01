"""
src/federated/fed_client.py
============================
Federated Learning client implementation.

Each IoT device (camera, sensor, router, smartlock) acts as a client:
1. Receives the current global model parameters from the server
2. Loads its local dataset
3. Trains locally for LOCAL_EPOCHS epochs
4. Returns updated model state_dict + number of training samples

This is used by the federated server to collect updates for FedAvg.

IMPORTANT:
 - The model architecture is identical across all clients (FederatedMLP)
 - Labels are encoded using the GLOBAL LabelEncoder (not per-node)
 - Features are scaled using the GLOBAL StandardScaler (not per-node)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import joblib
import logging

from config import PATHS, FEDERATED, RANDOM_STATE
from src.federated.fed_model import FederatedMLP, set_model_params, get_model_params

log = logging.getLogger(__name__)


class FederatedClient:
    """
    Represents one IoT device in the federated network.

    Parameters:
        client_name: one of 'camera', 'sensor', 'router', 'smartlock'
        n_features:  number of input features (must match server's model)
        n_classes:   number of output classes (must match server's model)
    """

    def __init__(self, client_name: str, n_features: int, n_classes: int):
        self.name       = client_name
        self.n_features = n_features
        self.n_classes  = n_classes
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Client-local model (same architecture as server)
        self.model = FederatedMLP(n_features, n_classes).to(self.device)

        # Load datasets
        self.X_train, self.y_train, self.X_val, self.y_val = self._load_data()
        log.info(
            f"[{self.name}] Data loaded: "
            f"train={len(self.X_train):,}  val={len(self.X_val):,}"
        )

    def _load_data(self):
        """
        Load and preprocess client-specific data.
        Uses the GLOBAL scaler and encoder (fitted on combined data).
        """
        path_map = {
            "camera":         PATHS.CAMERA_DATA,
            "router":         PATHS.ROUTER_DATA,
            "sensor":         PATHS.SENSOR_DATA,
            "door_lock":      PATHS.DOORLOCK_DATA,
            "smart_lighting": PATHS.LIGHTING_DATA,
            "smartlock":      PATHS.SMARTLOCK_DATA,
        }
        csv_path = path_map.get(self.name, PATHS.DATA / f"{self.name}_data.csv")
        df = pd.read_csv(csv_path)

        feature_names = pd.read_csv(PATHS.DATA / "feature_names.csv")["feature"].tolist()
        X = df[feature_names].values
        y_raw = df["Label"].values

        scaler  = joblib.load(PATHS.SCALER)
        encoder = joblib.load(PATHS.LABEL_ENCODER)

        # Encode labels — only keep classes known to encoder
        known_classes = set(encoder.classes_)
        mask = np.array([lbl in known_classes for lbl in y_raw])
        X = X[mask]
        y_raw = y_raw[mask]

        X_scaled = scaler.transform(X)
        y_enc    = encoder.transform(y_raw)

        # 80/20 train/val split per client
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y_enc, test_size=0.2,
            random_state=RANDOM_STATE, stratify=y_enc
        )
        return X_tr, y_tr, X_val, y_val

    def set_global_params(self, global_params: dict):
        """Receive global model parameters from server and update local model."""
        set_model_params(self.model, global_params)

    def get_local_params(self) -> dict:
        """Return a copy of local model parameters to send back to server."""
        return get_model_params(self.model)

    @property
    def num_train_samples(self) -> int:
        return len(self.X_train)

    def train_local(
        self,
        local_epochs: int = FEDERATED.LOCAL_EPOCHS,
        batch_size:   int = FEDERATED.BATCH_SIZE,
        lr:           float = FEDERATED.LR,
    ) -> dict:
        """
        Train locally for `local_epochs` epochs and return updated params.

        Returns:
            dict with keys:
                'state_dict'   : model parameters (state_dict)
                'n_samples'    : number of training samples
                'train_loss'   : average training loss (last epoch)
                'val_loss'     : validation loss
                'val_accuracy' : validation accuracy
        """
        torch.manual_seed(RANDOM_STATE)
        self.model.train()

        X_t = torch.FloatTensor(self.X_train).to(self.device)
        y_t = torch.LongTensor(self.y_train).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        train_loss = 0.0
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)
            train_loss = epoch_loss / len(self.X_train)

        # ── Validation ──
        val_loss, val_acc = self._evaluate(self.X_val, self.y_val, criterion)

        log.info(
            f"[{self.name}] Local training done | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

        return {
            "state_dict":    get_model_params(self.model),
            "n_samples":     self.num_train_samples,
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "val_accuracy":  val_acc,
        }

    def _evaluate(self, X, y, criterion):
        """Compute loss and accuracy on a dataset split."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.LongTensor(y).to(self.device)
            logits  = self.model(X_t)
            loss    = criterion(logits, y_t).item()
            preds   = logits.argmax(dim=1)
            acc     = (preds == y_t).float().mean().item()
        self.model.train()
        return loss, acc
