"""
src/models/s2_xgboost.py
=========================
Semester 2 — XGBoost baseline (multi-class, uses modular preprocessor).

Usage:
    python src/models/s2_xgboost.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import joblib
import logging
import numpy as np
from xgboost import XGBClassifier
from config import PATHS, TRAINING, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def train_xgboost():
    from src.data.preprocessor import load_preprocessed
    from src.evaluation.evaluator import evaluate_and_save

    PATHS.ensure_dirs()
    X_train, X_val, X_test, y_train, y_val, y_test, _, encoder, _ = load_preprocessed()

    X_fit = np.vstack([X_train, X_val])
    y_fit = np.hstack([y_train, y_val])

    n_classes = len(encoder.classes_)
    objective = 'multi:softprob' if n_classes > 2 else 'binary:logistic'

    log.info(f"Training XGBoost  n={TRAINING.XGB_ESTIMATORS}  depth={TRAINING.XGB_MAX_DEPTH}  lr={TRAINING.XGB_LR}")
    model = XGBClassifier(
        n_estimators=TRAINING.XGB_ESTIMATORS,
        max_depth=TRAINING.XGB_MAX_DEPTH,
        learning_rate=TRAINING.XGB_LR,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)

    y_pred = model.predict(X_test)
    metrics = evaluate_and_save(
        y_true=y_test,
        y_pred=y_pred,
        class_names=list(encoder.classes_),
        model_name="XGBoost",
    )

    joblib.dump(model, PATHS.XGB_MODEL)
    log.info(f"XGBoost model saved: {PATHS.XGB_MODEL}")
    return metrics


if __name__ == "__main__":
    train_xgboost()
