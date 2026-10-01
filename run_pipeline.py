"""
run_pipeline.py
================
End-to-end pipeline for the Federated IDS project.

Execution order:
    0. Setup directories
    1. Preprocessing (load CICIDS2017, map labels, split, scale)
    2. Client data creation (non-IID IoT splits)
    3. Centralized baselines: Random Forest, XGBoost, Neural Network
    4. Federated training (real FedAvg on PyTorch NN)
    5. Evaluation & final comparison
    6. Visualization
    7. SHAP explainability
    8. Summary report

Usage:
    python run_pipeline.py                  # full pipeline
    python run_pipeline.py --skip-preprocess  # skip slow CSV loading
    python run_pipeline.py --only-federated   # only run federated training
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode='w'),
    ]
)
log = logging.getLogger(__name__)

from config import PATHS

def banner(msg):
    log.info("\n" + "=" * 65)
    log.info(f"  {msg}")
    log.info("=" * 65)


def step_preprocess():
    banner("PHASE B — DATA PREPROCESSING")
    from src.data.preprocessor import load_and_preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder, feat = load_and_preprocess()
    log.info(f"  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")
    log.info(f"  Features={len(feat)}  Classes={list(encoder.classes_)}")
    return feat, encoder


def step_client_split():
    banner("PHASE C — CLIENT DATA SPLITTING (Non-IID)")
    from src.data.client_splitter import split_clients
    clients = split_clients()
    for name, df in clients.items():
        log.info(f"  {name}: {len(df):,} samples")


def step_random_forest():
    banner("PHASE D1 — CENTRALIZED: RANDOM FOREST")
    from src.models.s2_random_forest import train_random_forest
    return train_random_forest()


def step_xgboost():
    banner("PHASE D2 — CENTRALIZED: XGBOOST")
    from src.models.s2_xgboost import train_xgboost
    return train_xgboost()


def step_neural_network():
    banner("PHASE D3 — CENTRALIZED: PYTORCH NEURAL NETWORK")
    from src.models.pytorch_nn import train_centralized_nn
    return train_centralized_nn()


def step_federated():
    banner("PHASE E — TRUE FEDERATED LEARNING (FedAvg)")
    from src.federated.fed_server import run_federated
    server = run_federated()
    final = server.round_results[-1] if server.round_results else {}
    log.info(f"  Final Federated Accuracy: {final.get('Global_Accuracy', 'N/A'):.4f}")
    log.info(f"  Final F1 Weighted:        {final.get('Global_F1_Weighted', 'N/A'):.4f}")
    return server


def step_federated_evaluation(server=None):
    """Evaluate federated model and build final comparison table."""
    banner("PHASE G — EVALUATION & FINAL COMPARISON")
    import numpy as np
    import joblib
    import torch
    from src.evaluation.evaluator import evaluate_and_save, save_final_comparison
    from src.federated.fed_model import FederatedMLP
    from config import PATHS, CLASS_NAMES

    if not PATHS.GLOBAL_MODEL.exists():
        log.warning("No federated model found; skipping federated evaluation.")
        return

    # Load model
    checkpoint = torch.load(PATHS.GLOBAL_MODEL, map_location='cpu')
    model = FederatedMLP(checkpoint['n_features'], checkpoint['n_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    encoder = joblib.load(PATHS.LABEL_ENCODER)

    X_test = np.load(PATHS.DATA / "test_features.npy")
    y_test = np.load(PATHS.DATA / "test_labels.npy")

    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test)).argmax(1).numpy()

    fed_metrics = evaluate_and_save(
        y_true=y_test, y_pred=y_pred,
        class_names=list(encoder.classes_),
        model_name="Federated_Neural_Network"
    )

    # Gather all results
    from config import PATHS as P
    import pandas as pd
    results = []
    for csv in [
        P.CLASS_REPORTS.parent.parent / "results" / "random_forest_results.csv",  # legacy
    ]:
        pass  # handled below via evaluator output

    # Rebuild comparison from individual CSV files where they exist
    for model_name, result_fname in [
        ("Random_Forest",        "Random_Forest_classification_report.txt"),
        ("XGBoost",              "XGBoost_classification_report.txt"),
        ("Centralized_Neural_Network", "Centralized_Neural_Network_classification_report.txt"),
    ]:
        # Re-evaluate from saved predictions if available, else skip
        pass

    # Load from final_comparison if it exists and append fed results
    final_csv = PATHS.FINAL_COMPARISON_CSV
    if final_csv.exists():
        existing = pd.read_csv(final_csv).to_dict(orient="records")
        # Remove old federated entry if present
        existing = [r for r in existing if "Federated" not in str(r.get("model",""))]
        existing.append(fed_metrics)
        save_final_comparison(existing)
    else:
        save_final_comparison([fed_metrics])


def step_visualize():
    banner("PHASE H — VISUALIZATION")
    from src.visualize_results import run_all
    run_all()


def step_shap():
    banner("PHASE L — EXPLAINABLE AI (SHAP)")
    try:
        from src.explainability.shap_explainer import explain_xgboost, explain_nn
        log.info("  Running XGBoost SHAP (TreeExplainer)...")
        r1 = explain_xgboost(n_samples=200)
        if r1:
            log.info(f"  XGBoost SHAP plot: {r1.get('plot_path')}")
            top5 = r1["importance_df"].head(5)["feature"].tolist()
            log.info(f"  Top-5 XGBoost features: {top5}")

        log.info("  Running NN SHAP (DeepExplainer)...")
        r2 = explain_nn(n_samples=100)
        if r2:
            log.info(f"  NN SHAP plot: {r2.get('plot_path')}")
    except Exception as e:
        log.warning(f"  SHAP step failed: {e}")


def step_summary():
    banner("PHASE M — EXPERIMENT SUMMARY")
    import pandas as pd
    from datetime import datetime

    lines = [
        f"Federated IDS — Experiment Summary",
        f"Generated: {datetime.now().isoformat()}",
        f"{'='*60}",
        "",
        "DATASET",
        "  Source: CICIDS2017 + IoT device simulation",
        "  Classes: Normal, DDoS, PortScan, BruteForce, WebAttack (5)",
        "",
    ]

    if PATHS.CLIENT_DIST_CSV.exists():
        dist = pd.read_csv(PATHS.CLIENT_DIST_CSV)
        lines.append("CLIENT DISTRIBUTION")
        lines.append(dist[["Client","Total_Samples","Num_Classes"]].to_string(index=False))
        lines.append("")

    if PATHS.FINAL_COMPARISON_CSV.exists():
        comp = pd.read_csv(PATHS.FINAL_COMPARISON_CSV)
        lines.append("MODEL COMPARISON")
        lines.append(comp.to_string(index=False))
        lines.append("")

    if PATHS.FED_ROUNDS_CSV.exists():
        fed = pd.read_csv(PATHS.FED_ROUNDS_CSV)
        if not fed.empty:
            final = fed.iloc[-1]
            lines.append("FEDERATED TRAINING")
            lines.append(f"  Rounds: {len(fed)}")
            lines.append(f"  Final Global Accuracy:    {final['Global_Accuracy']:.4f}")
            lines.append(f"  Final F1 Weighted:        {final['Global_F1_Weighted']:.4f}")
            lines.append(f"  Final F1 Macro:           {final['Global_F1_Macro']:.4f}")
            lines.append("")

    lines += [
        "HOW TO RUN",
        "  Full pipeline:        python run_pipeline.py",
        "  Dashboard:            python src/dashboard/app.py",
        "  Tests:                python -m pytest tests/ -v",
    ]

    summary_text = "\n".join(lines)
    PATHS.EXPERIMENT_SUMMARY.write_text(summary_text)
    log.info(summary_text)
    log.info(f"\n  Summary saved: {PATHS.EXPERIMENT_SUMMARY}")


def main():
    parser = argparse.ArgumentParser(description="Federated IDS Pipeline")
    parser.add_argument("--skip-preprocess",  action="store_true")
    parser.add_argument("--skip-clients",     action="store_true")
    parser.add_argument("--skip-centralized", action="store_true")
    parser.add_argument("--only-federated",   action="store_true")
    parser.add_argument("--skip-shap",        action="store_true")
    args = parser.parse_args()

    PATHS.ensure_dirs()
    t0 = time.time()

    all_results = []

    # ── Preprocessing ──
    if not args.skip_preprocess and not args.only_federated:
        step_preprocess()
    else:
        log.info("[SKIP] Preprocessing")

    # ── Client splitting ──
    if not args.skip_clients and not args.only_federated:
        step_client_split()
    else:
        log.info("[SKIP] Client splitting")

    # ── Centralized baselines ──
    if not args.skip_centralized and not args.only_federated:
        from src.evaluation.evaluator import save_final_comparison
        rf_metrics  = step_random_forest()
        xgb_metrics = step_xgboost()
        nn_metrics  = step_neural_network()
        all_results = [rf_metrics, xgb_metrics, nn_metrics]
        save_final_comparison(all_results)
    else:
        log.info("[SKIP] Centralized models")

    # ── Federated training ──
    server = step_federated()

    # ── Federated evaluation ──
    step_federated_evaluation(server)

    # ── Visualization ──
    step_visualize()

    # ── SHAP ──
    if not args.skip_shap:
        step_shap()
    else:
        log.info("[SKIP] SHAP")

    # ── Summary ──
    step_summary()

    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    log.info("  Dashboard: python src/dashboard/app.py")
    log.info("  Tests:     python -m pytest tests/ -v")


if __name__ == "__main__":
    main()
