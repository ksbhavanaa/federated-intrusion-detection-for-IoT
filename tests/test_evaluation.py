"""
tests/test_evaluation.py
=========================
Tests for the evaluation module.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from pathlib import Path


def test_evaluate_and_save_returns_metrics():
    """evaluate_and_save must return a dict with all required metric keys."""
    from src.evaluation.evaluator import evaluate_and_save
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1])
    classes = ["Normal", "DDoS", "PortScan"]

    metrics = evaluate_and_save(y_true, y_pred, classes, "test_model_unit")
    required = ["model", "accuracy", "precision_macro", "recall_macro",
                "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"]
    for key in required:
        assert key in metrics, f"Missing metric: {key}"


def test_metrics_in_range():
    from src.evaluation.evaluator import evaluate_and_save
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])  # perfect predictions
    metrics = evaluate_and_save(y_true, y_pred, ["A","B","C"], "test_perfect")
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_confusion_matrix_saved():
    from src.evaluation.evaluator import evaluate_and_save
    from config import PATHS
    y = np.array([0, 1, 0, 1])
    evaluate_and_save(y, y, ["X","Y"], "test_cm_save")
    assert (PATHS.CONF_MATRICES / "test_cm_save_confusion_matrix.png").exists()


def test_classification_report_saved():
    from src.evaluation.evaluator import evaluate_and_save
    from config import PATHS
    y = np.array([0, 1, 0, 1])
    evaluate_and_save(y, y, ["X","Y"], "test_report_save")
    assert (PATHS.CLASS_REPORTS / "test_report_save_classification_report.txt").exists()
