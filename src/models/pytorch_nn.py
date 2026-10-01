"""
src/models/pytorch_nn.py
=========================
Centralized PyTorch Neural Network baseline.
Uses the same FederatedMLP architecture for fair comparison with federated model.

Usage:
    python src/models/pytorch_nn.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import logging

from config import PATHS, TRAINING, RANDOM_STATE, CLASS_NAMES
from src.federated.fed_model import FederatedMLP

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def train_centralized_nn():
    """Train and evaluate the centralized PyTorch neural network."""
    from src.data.preprocessor import load_preprocessed
    from src.evaluation.evaluator import evaluate_and_save

    PATHS.ensure_dirs()
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load preprocessed splits
    X_train, X_val, X_test, y_train, y_val, y_test, _, encoder, _ = load_preprocessed()
    n_features = X_train.shape[1]
    n_classes  = len(encoder.classes_)
    log.info(f"Features: {n_features}, Classes: {n_classes} {list(encoder.classes_)}")

    # Tensors
    def to_tensors(X, y):
        return torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device)

    X_tr, y_tr = to_tensors(X_train, y_train)
    X_v, y_v   = to_tensors(X_val, y_val)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=TRAINING.NN_BATCH_SIZE, shuffle=True
    )

    # Model
    model = FederatedMLP(n_features=n_features, n_classes=n_classes,
                         hidden_layers=TRAINING.NN_HIDDEN,
                         dropout=TRAINING.NN_DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAINING.NN_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    log.info(f"Training Centralized Neural Network for {TRAINING.NN_EPOCHS} epochs...")
    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, TRAINING.NN_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)

        train_loss = epoch_loss / len(y_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss   = criterion(val_logits, y_v).item()
            val_acc    = (val_logits.argmax(1) == y_v).float().mean().item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:3d}/{TRAINING.NN_EPOCHS}  "
                     f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # Load best checkpoint
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "n_features":       n_features,
        "n_classes":        n_classes,
        "class_names":      list(encoder.classes_),
    }, PATHS.NN_CENTRAL)
    log.info(f"Centralized NN saved: {PATHS.NN_CENTRAL}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(device)).argmax(1).cpu().numpy()

    metrics = evaluate_and_save(
        y_true=y_test,
        y_pred=y_pred,
        class_names=list(encoder.classes_),
        model_name="Centralized_Neural_Network",
    )
    log.info(f"\nCentralized NN  Accuracy={metrics['accuracy']:.4f}  "
             f"F1_weighted={metrics['f1_weighted']:.4f}")
    return metrics


if __name__ == "__main__":
    train_centralized_nn()
