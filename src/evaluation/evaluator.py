"""
src/evaluation/evaluator.py
============================
Shared evaluation utilities for all models.

Generates:
 - Accuracy, Precision, Recall, F1 (macro + weighted)
 - Classification reports (text + CSV)
 - Confusion matrices (saved as PNG)

Usage:
    from src.evaluation.evaluator import evaluate_and_save, save_final_comparison
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from config import PATHS

log = logging.getLogger(__name__)


def evaluate_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    model_name: str,
) -> dict:
    """
    Compute full evaluation metrics, save classification report and confusion matrix.

    Args:
        y_true:       true integer labels
        y_pred:       predicted integer labels
        class_names:  list of class name strings (ordered by class index)
        model_name:   identifier string for filenames

    Returns:
        dict with all metric values
    """
    PATHS.ensure_dirs()
    os.makedirs(PATHS.CONF_MATRICES, exist_ok=True)
    os.makedirs(PATHS.CLASS_REPORTS, exist_ok=True)

    acc    = accuracy_score(y_true, y_pred)
    p_mac  = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    r_mac  = recall_score(y_true, y_pred,    average='macro',    zero_division=0)
    f1_mac = f1_score(y_true, y_pred,        average='macro',    zero_division=0)
    p_wt   = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    r_wt   = recall_score(y_true, y_pred,    average='weighted', zero_division=0)
    f1_wt  = f1_score(y_true, y_pred,        average='weighted', zero_division=0)

    # ── Classification report ──
    present_indices = sorted(set(y_true) | set(y_pred))
    present_names   = [class_names[i] if i < len(class_names) else str(i)
                       for i in present_indices]
    report_str = classification_report(
        y_true, y_pred,
        labels=present_indices,
        target_names=present_names,
        zero_division=0
    )
    report_path = PATHS.CLASS_REPORTS / f"{model_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write(report_str)
    log.info(f"Classification report saved: {report_path}")

    # ── Confusion matrix ──
    cm = confusion_matrix(y_true, y_pred, labels=present_indices)
    fig, ax = plt.subplots(figsize=(max(6, len(present_names)), max(5, len(present_names))))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=present_names, yticklabels=present_names, ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name.replace("_", " ")}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = PATHS.CONF_MATRICES / f"{model_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Confusion matrix saved: {cm_path}")

    metrics = {
        "model":              model_name,
        "accuracy":           acc,
        "precision_macro":    p_mac,
        "recall_macro":       r_mac,
        "f1_macro":           f1_mac,
        "precision_weighted": p_wt,
        "recall_weighted":    r_wt,
        "f1_weighted":        f1_wt,
    }

    log.info(
        f"[{model_name}] acc={acc:.4f}  "
        f"p_mac={p_mac:.4f}  r_mac={r_mac:.4f}  f1_mac={f1_mac:.4f}  "
        f"f1_wt={f1_wt:.4f}"
    )

    update_comparison_entry(metrics)
    return metrics


def save_final_comparison(results_list: list):
    """
    Save final_comparison.csv with metrics from all evaluated models.

    Args:
        results_list: list of metric dicts from evaluate_and_save()
    """
    df = pd.DataFrame(results_list)
    cols = [
        "model", "accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted"
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.round(4)
    df.to_csv(PATHS.FINAL_COMPARISON_CSV, index=False)
    log.info(f"Final comparison saved: {PATHS.FINAL_COMPARISON_CSV}")
    log.info("\n" + df.to_string(index=False))
    return df


def update_comparison_entry(metric_dict: dict):
    """
    Append or replace a single model's metrics in final_comparison.csv
    and persist into research_experiments SQLite catalog.
    """
    from config import PATHS
    final_csv = PATHS.FINAL_COMPARISON_CSV
    existing = []
    if final_csv.exists():
        try:
            existing = pd.read_csv(final_csv).to_dict(orient="records")
        except Exception:
            existing = []

    model_name = metric_dict.get("model", "")
    if model_name.startswith("test_"):
        return
    existing = [r for r in existing if r.get("model") != model_name]
    existing.append(metric_dict)
    save_final_comparison(existing)

    # Persist into research_experiments catalog
    try:
        from src.database.db import save_research_experiment
        is_central = any(k in model_name for k in ["Centralized", "Forest", "XGBoost"])
        category = "CENTRALIZED_BENCHMARK" if is_central else "FEDERATED"
        save_research_experiment(
            exp_id=f"EXP-{model_name.upper()}",
            name=f"Benchmark: {model_name.replace('_', ' ')}",
            category=category,
            model_type=model_name,
            dataset_name="CICIDS2017-Derived-6Class",
            config={"evaluation": "held_out_test_split", "samples": 10247},
            results=metric_dict
        )
    except Exception as e:
        log.warning(f"Could not persist {model_name} to research_experiments: {e}")
