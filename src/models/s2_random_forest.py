"""
src/models/s2_random_forest.py
================================
Semester 2 — Random Forest baseline (multi-class, uses modular preprocessor).
Preserves Semester 1 results. Saves new model and evaluation to Semester 2 paths.

Usage:
    python src/models/s2_random_forest.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import joblib
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from config import PATHS, TRAINING, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def train_random_forest():
    from src.data.preprocessor import load_preprocessed
    from src.evaluation.evaluator import evaluate_and_save

    PATHS.ensure_dirs()
    X_train, X_val, X_test, y_train, y_val, y_test, _, encoder, _ = load_preprocessed()

    # Use combined train+val for final model training
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.hstack([y_train, y_val])

    log.info(f"Training Random Forest  n_estimators={TRAINING.RF_ESTIMATORS}")
    model = RandomForestClassifier(
        n_estimators=TRAINING.RF_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_fit, y_fit)

    y_pred = model.predict(X_test)
    metrics = evaluate_and_save(
        y_true=y_test,
        y_pred=y_pred,
        class_names=list(encoder.classes_),
        model_name="Random_Forest",
    )

    joblib.dump(model, PATHS.RF_MODEL)
    log.info(f"RF model saved: {PATHS.RF_MODEL}")
    return metrics


if __name__ == "__main__":
    train_random_forest()
