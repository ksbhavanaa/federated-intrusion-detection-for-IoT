"""
src/federated/fed_server.py
============================
Federated Learning server with true FedAvg parameter aggregation.

Algorithm (McMahan et al., Communication-Efficient Learning of Deep Networks, 2017):

  For each round r = 1..NUM_ROUNDS:
    1. Broadcast global model W_r to all clients
    2. Each client k trains locally → returns (W_k, n_k)
    3. Server aggregates: W_{r+1} = Σ(n_k * W_k) / Σ(n_k)
    4. Evaluate global model on held-out test set
    5. Save round metrics

This is ACTUAL parameter aggregation — NOT metric averaging.

Usage:
    python src/federated/fed_server.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import logging
import time

from config import PATHS, FEDERATED, RANDOM_STATE, CLASS_NAMES
from src.federated.fed_model import (
    FederatedMLP, build_model, fedavg,
    get_model_params, set_model_params
)
from src.federated.fed_client import FederatedClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


class FederatedServer:
    """
    Central aggregation server for Federated Learning.
    Manages the global model and coordinates client training rounds.
    """

    def __init__(self, n_features: int, n_classes: int):
        self.n_features = n_features
        self.n_classes  = n_classes
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(RANDOM_STATE)
        self.global_model = build_model(n_features, n_classes).to(self.device)
        log.info(f"Global model initialized on {self.device}")
        log.info(f"Model params: {sum(p.numel() for p in self.global_model.parameters()):,}")

        # Load test data for global evaluation
        self.X_test, self.y_test = self._load_test_data()
        log.info(f"Test set loaded: {len(self.X_test):,} samples")

        # Round results storage
        self.round_results = []

    def _load_test_data(self):
        """Load the held-out test split (preprocessed, never seen during training)."""
        X_test = np.load(PATHS.DATA / "test_features.npy")
        y_test = np.load(PATHS.DATA / "test_labels.npy")
        return X_test, y_test

    def evaluate_global(self) -> dict:
        """
        Evaluate the current global model on the held-out test set.
        Returns dict with accuracy, precision, recall, f1 (macro + weighted).
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        self.global_model.eval()
        with torch.no_grad():
            X_t   = torch.FloatTensor(self.X_test).to(self.device)
            logits = self.global_model(X_t)
            y_pred = logits.argmax(dim=1).cpu().numpy()

        loss_fn   = nn.CrossEntropyLoss()
        y_t       = torch.LongTensor(self.y_test).to(self.device)
        test_loss = loss_fn(torch.FloatTensor(
            torch.softmax(torch.FloatTensor(
                self.global_model(torch.FloatTensor(self.X_test).to(self.device)).detach().cpu().numpy()
            ), dim=1).numpy()
        ), torch.LongTensor(self.y_test)).item() if False else 0.0  # skip for speed

        # Compute proper test loss
        with torch.no_grad():
            X_t    = torch.FloatTensor(self.X_test).to(self.device)
            y_t    = torch.LongTensor(self.y_test).to(self.device)
            logits = self.global_model(X_t)
            test_loss = nn.CrossEntropyLoss()(logits, y_t).item()

        return {
            "accuracy":   accuracy_score(self.y_test, y_pred),
            "precision_macro":  precision_score(self.y_test, y_pred, average='macro',    zero_division=0),
            "recall_macro":     recall_score(self.y_test, y_pred,    average='macro',    zero_division=0),
            "f1_macro":         f1_score(self.y_test, y_pred,        average='macro',    zero_division=0),
            "precision_weighted": precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            "recall_weighted":    recall_score(self.y_test, y_pred,    average='weighted', zero_division=0),
            "f1_weighted":        f1_score(self.y_test, y_pred,        average='weighted', zero_division=0),
            "test_loss":  test_loss,
            "y_pred":     y_pred,
        }

    def run(
        self,
        clients: list,
        num_rounds: int = FEDERATED.NUM_ROUNDS,
    ) -> list:
        """
        Execute the full federated training loop.

        Args:
            clients   : list of FederatedClient instances
            num_rounds: number of communication rounds

        Returns:
            list of per-round result dicts
        """
        PATHS.ensure_dirs()

        log.info("=" * 60)
        log.info("STARTING FEDERATED TRAINING")
        log.info(f"  Clients  : {[c.name for c in clients]}")
        log.info(f"  Rounds   : {num_rounds}")
        log.info(f"  L-epochs : {FEDERATED.LOCAL_EPOCHS}")
        log.info(f"  Batch    : {FEDERATED.BATCH_SIZE}")
        log.info(f"  LR       : {FEDERATED.LR}")
        log.info("=" * 60)

        # Evaluate initial (untrained) global model
        init_metrics = self.evaluate_global()
        log.info(f"[Round 0 / Init] Accuracy={init_metrics['accuracy']:.4f}  F1={init_metrics['f1_weighted']:.4f}")

        self.round_results = []

        for rnd in range(1, num_rounds + 1):
            t_start = time.time()
            log.info(f"\n{'─'*50}")
            log.info(f"  ROUND {rnd}/{num_rounds}")
            log.info(f"{'─'*50}")

            # ── Step 1: Distribute current global model to all clients ──
            global_params = get_model_params(self.global_model)
            for client in clients:
                client.set_global_params(global_params)

            # ── Step 2: Each client trains locally ──
            client_updates = []
            for client in clients:
                update = client.train_local(
                    local_epochs=FEDERATED.LOCAL_EPOCHS,
                    batch_size=FEDERATED.BATCH_SIZE,
                    lr=FEDERATED.LR,
                )
                client_updates.append(update)
                log.info(
                    f"  [{client.name}] n={update['n_samples']:,}  "
                    f"train_loss={update['train_loss']:.4f}  "
                    f"val_acc={update['val_accuracy']:.4f}"
                )

            # ── Step 3: FedAvg aggregation ──
            param_list    = [u["state_dict"] for u in client_updates]
            n_samples_list = [u["n_samples"]  for u in client_updates]

            aggregated_params = fedavg(param_list, n_samples_list)
            set_model_params(self.global_model, aggregated_params)

            # ── Step 4: Evaluate global model on held-out test set ──
            metrics = self.evaluate_global()

            elapsed = time.time() - t_start
            avg_client_val_acc = np.mean([u["val_accuracy"] for u in client_updates])
            avg_train_loss     = np.mean([u["train_loss"]   for u in client_updates])

            log.info(
                f"\n  [Round {rnd}] Global Accuracy={metrics['accuracy']:.4f}  "
                f"F1_weighted={metrics['f1_weighted']:.4f}  "
                f"F1_macro={metrics['f1_macro']:.4f}  "
                f"elapsed={elapsed:.1f}s"
            )

            round_record = {
                "Round":             rnd,
                "Global_Accuracy":   metrics["accuracy"],
                "Global_F1_Macro":   metrics["f1_macro"],
                "Global_F1_Weighted":metrics["f1_weighted"],
                "Global_Precision":  metrics["precision_weighted"],
                "Global_Recall":     metrics["recall_weighted"],
                "Global_Loss":       metrics["test_loss"],
                "Avg_Client_ValAcc": avg_client_val_acc,
                "Avg_Train_Loss":    avg_train_loss,
                "Total_Samples":     sum(n_samples_list),
                "Aggregation_Method": "Sample-Weighted FedAvg",
                "Aggregation_Weights": ", ".join([f"{c.name}: {n/sum(n_samples_list):.4f}" for c, n in zip(clients, n_samples_list)]),
                "Elapsed_Sec":       round(elapsed, 2),
            }
            # Per-client metrics
            for client, update in zip(clients, client_updates):
                round_record[f"{client.name}_train_loss"] = update["train_loss"]
                round_record[f"{client.name}_val_acc"]    = update["val_accuracy"]
                round_record[f"{client.name}_n_samples"]  = update["n_samples"]

            self.round_results.append(round_record)

        # ── Save round results ──
        results_df = pd.DataFrame(self.round_results)
        results_df.to_csv(PATHS.FED_ROUNDS_CSV, index=False)
        log.info(f"\nRound results saved: {PATHS.FED_ROUNDS_CSV}")

        # ── Save final global model ──
        torch.save({
            "model_state_dict": get_model_params(self.global_model),
            "n_features":       self.n_features,
            "n_classes":        self.n_classes,
            "class_names":      CLASS_NAMES,
            "num_rounds":       num_rounds,
            "final_accuracy":   self.round_results[-1]["Global_Accuracy"],
        }, PATHS.GLOBAL_MODEL)
        log.info(f"Global model saved: {PATHS.GLOBAL_MODEL}")

        final = self.round_results[-1]
        log.info("\n" + "=" * 60)
        log.info("FEDERATED TRAINING COMPLETE")
        log.info(f"  Final Accuracy   : {final['Global_Accuracy']:.4f}")
        log.info(f"  Final F1 Macro   : {final['Global_F1_Macro']:.4f}")
        log.info(f"  Final F1 Weighted: {final['Global_F1_Weighted']:.4f}")
        log.info("=" * 60)

        return self.round_results


def run_federated():
    """Entry point: load data, create clients, run federated training."""
    PATHS.ensure_dirs()

    # Load preprocessed feature count and class count
    feature_names = pd.read_csv(PATHS.DATA / "feature_names.csv")["feature"].tolist()
    encoder       = joblib.load(PATHS.LABEL_ENCODER)
    n_features    = len(feature_names)
    n_classes     = len(encoder.classes_)

    log.info(f"n_features={n_features}  n_classes={n_classes}  classes={list(encoder.classes_)}")

    # Create server
    server = FederatedServer(n_features=n_features, n_classes=n_classes)

    # Create clients
    clients = [
        FederatedClient(client_name=name, n_features=n_features, n_classes=n_classes)
        for name in FEDERATED.CLIENTS
    ]

    # Run federated training
    server.run(clients, num_rounds=FEDERATED.NUM_ROUNDS)

    return server


if __name__ == "__main__":
    run_federated()
