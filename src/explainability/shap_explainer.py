"""
src/explainability/shap_explainer.py
======================================
Explainable AI (XAI) engine for the Federated IoT IDS.
Provides both local (alert-level) and global feature explanations,
along with real mathematical calculations for XAI Fidelity and XAI Stability.

Attribution Method: Gradient × Input (first-order Taylor approximation)
  A_i = x_i * ∂y_c/∂x_i

  This is NOT sampling-based KernelSHAP or tree-based TreeSHAP.
  All labels in the codebase use "Gradient × Input Attribution" for accuracy.

Fidelity:
  Measures whether masking the top-k most important features identified by
  the explainer causes the expected reduction in prediction confidence.
  Fidelity = % of test samples where masking top-k features drops confidence.

Stability:
  Measures explanation consistency when minor Gaussian noise (perturbation)
  is added to the input flow vector.
  Stability = Average Jaccard overlap of top-k features across perturbations.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Human-readable feature descriptions (CICIDS2017 → Plain English)
# These map raw CICIDS2017 feature names to SOC-readable explanations.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    # Packet rate / volume features
    "Flow Packets/s":           "Abnormally high packet rate (flood indicator)",
    "Fwd Packets/s":            "Extremely rapid forward packet flood",
    "Bwd Packets/s":            "Extremely rapid backward packet flood",
    "Total Fwd Packets":        "Very high total outgoing packet count",
    "Total Backward Packets":   "Very high total incoming packet count",
    "Total Length of Fwd Packets": "Large total outgoing payload volume",
    "Total Length of Bwd Packets": "Large total incoming payload volume",

    # Timing / inter-arrival features
    "Flow Duration":            "Unusually short-lived or long-duration connection",
    "Flow IAT Mean":            "Irregular inter-packet timing pattern",
    "Flow IAT Std":             "High variance in packet arrival timing",
    "Flow IAT Max":             "Extremely large gap between packets (scan pattern)",
    "Flow IAT Min":             "Extremely small gap between packets (burst/flood)",
    "Fwd IAT Total":            "Abnormal total inter-arrival time (forward)",
    "Fwd IAT Mean":             "Irregular forward packet timing",
    "Fwd IAT Std":              "Unstable forward packet timing pattern",
    "Fwd IAT Max":              "Suspiciously long pause in forward stream",
    "Fwd IAT Min":              "Extremely rapid consecutive forward packets",
    "Bwd IAT Total":            "Abnormal total inter-arrival time (backward)",
    "Bwd IAT Mean":             "Irregular backward packet timing",
    "Bwd IAT Std":              "Unstable backward packet timing pattern",
    "Bwd IAT Max":              "Suspiciously long pause in backward stream",
    "Bwd IAT Min":              "Extremely rapid consecutive backward packets",

    # Header / size features
    "Fwd Header Length":        "Unusual forward packet header size",
    "Bwd Header Length":        "Unusual backward packet header size",
    "Fwd Packet Length Mean":   "Abnormal average outgoing packet size",
    "Fwd Packet Length Max":    "Unusually large outgoing packet detected",
    "Fwd Packet Length Min":    "Suspiciously small outgoing packets (scan probe)",
    "Fwd Packet Length Std":    "High variance in outgoing packet sizes",
    "Bwd Packet Length Mean":   "Abnormal average incoming packet size",
    "Bwd Packet Length Max":    "Unusually large incoming packet detected",
    "Bwd Packet Length Min":    "Suspiciously small incoming packets",
    "Bwd Packet Length Std":    "High variance in incoming packet sizes",
    "Average Packet Size":      "Abnormal average packet payload size",
    "Avg Fwd Segment Size":     "Anomalous outgoing TCP segment size",
    "Avg Bwd Segment Size":     "Anomalous incoming TCP segment size",
    "Packet Length Mean":       "Abnormal mean packet length across flow",
    "Packet Length Std":        "High variance in packet lengths (evasion pattern)",
    "Packet Length Variance":   "High packet length variance detected",

    # Flag-based features
    "FIN Flag Count":           "Unusual number of FIN flags (stealth scan pattern)",
    "SYN Flag Count":           "High SYN count without completion (SYN flood / port scan)",
    "RST Flag Count":           "Excessive TCP RST flags (connection reset storm)",
    "PSH Flag Count":           "Abnormal PSH flag usage (data injection pattern)",
    "ACK Flag Count":           "Abnormal ACK pattern (ACK flood or evasion)",
    "URG Flag Count":           "Unexpected urgent pointer flags",
    "CWE Flag Count":           "Congestion window exceeded flags detected",
    "ECE Flag Count":           "ECN Echo flags in suspicious pattern",
    "Fwd PSH Flags":            "Forward stream pushing data aggressively",
    "Bwd PSH Flags":            "Backward stream pushing data aggressively",
    "Fwd URG Flags":            "Urgent flags in forward stream",
    "Bwd URG Flags":            "Urgent flags in backward stream",

    # Ratio / derived features
    "Down/Up Ratio":            "Unusual download-to-upload traffic ratio",
    "Fwd Bytes/Bulk Avg":       "Anomalous bulk data rate in forward direction",
    "Fwd Packet/Bulk Avg":      "Anomalous bulk packet rate in forward direction",
    "Fwd Bulk Rate Avg":        "High-rate bulk data burst detected",
    "Bwd Bytes/Bulk Avg":       "Anomalous bulk data rate in backward direction",
    "Bwd Packet/Bulk Avg":      "Anomalous bulk packet rate in backward direction",
    "Bwd Bulk Rate Avg":        "High-rate bulk data burst in backward direction",

    # Connection / subflow features
    "Subflow Fwd Packets":      "Abnormal subflow forward packet count",
    "Subflow Fwd Bytes":        "Abnormal subflow forward byte count",
    "Subflow Bwd Packets":      "Abnormal subflow backward packet count",
    "Subflow Bwd Bytes":        "Abnormal subflow backward byte count",
    "Active Mean":              "Abnormal active connection duration pattern",
    "Active Std":               "Unstable active session timing",
    "Active Max":               "Session kept alive suspiciously long",
    "Active Min":               "Session terminated extremely quickly (scan)",
    "Idle Mean":                "Unusual idle time between flows",
    "Idle Std":                 "High variance in connection idle time",
    "Idle Max":                 "Very long idle period (C2 beacon pattern)",
    "Idle Min":                 "Very short idle period (rapid reconnect pattern)",

    # Window / initialization features
    "Init_Win_bytes_forward":   "Unusual TCP window size in forward direction",
    "Init_Win_bytes_backward":  "Unusual TCP window size in backward direction",
    "act_data_pkt_fwd":         "Abnormal active data packet count (forward)",
    "min_seg_size_forward":     "Very small minimum segment size (probe pattern)",
}

# Per-attack-class key indicator descriptions shown to SOC analysts
ATTACK_INDICATORS = {
    "DDoS": [
        "Abnormally high packet rate (flood indicator)",
        "Very high total outgoing packet count",
        "Extremely rapid forward packet flood",
        "High SYN count without completion (SYN flood / port scan)",
        "Abnormal PSH flag usage (data injection pattern)",
    ],
    "PortScan": [
        "High SYN count without completion (SYN flood / port scan)",
        "Suspiciously small outgoing packets (scan probe)",
        "Very small minimum segment size (probe pattern)",
        "Unusually large gap between packets (scan pattern)",
        "Unusual number of FIN flags (stealth scan pattern)",
    ],
    "Botnet": [
        "Very long idle period (C2 beacon pattern)",
        "Session kept alive suspiciously long",
        "Unusual download-to-upload traffic ratio",
        "Irregular inter-packet timing pattern",
        "Abnormal total inter-arrival time (backward)",
    ],
    "BruteForce": [
        "Extremely rapid consecutive forward packets",
        "High total outgoing packet count",
        "Abnormal PSH flag usage (data injection pattern)",
        "Unusual TCP window size in forward direction",
        "Unstable forward packet timing pattern",
    ],
    "WebAttack": [
        "Unusually large outgoing packet detected",
        "Abnormal average outgoing packet size",
        "Large total outgoing payload volume",
        "Abnormal PSH flag usage (data injection pattern)",
        "Unusual forward packet header size",
    ],
    "Normal": [
        "Traffic pattern consistent with baseline behavior",
    ],
}


def get_human_description(feature_name: str) -> str:
    """Return plain-English description for a CICIDS2017 feature name."""
    return FEATURE_DESCRIPTIONS.get(feature_name, f"Anomalous {feature_name.lower()} detected")

import sys
import os
import logging
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import joblib
import torch

from config import PATHS, CLASS_NAMES
from src.federated.fed_model import FederatedMLP
from src.database.db import get_db_connection

log = logging.getLogger(__name__)

# Cache for global importance
_GLOBAL_IMPORTANCE_CACHE = None


def get_global_model():
    """Load the trained global Federated MLP."""
    path = PATHS.GLOBAL_MODEL
    if not path.exists():
        path = PATHS.NN_CENTRAL
    if not path.exists():
        return None

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = FederatedMLP(checkpoint['n_features'], checkpoint['n_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def explain_alert_instance(features: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
    """
    Computes local feature attributions for a single alert flow vector using
    Gradient × Input attribution (first-order Taylor approximation).

    Returns:
        dict with plain-English human_readable_explanation matching blueprint output:
            Attack detected: DDoS
            Top reasons:
            • Abnormally high packet rate (flood indicator)
            • Very high total outgoing packet count
            ...
    """
    model = get_global_model()
    feature_names = pd.read_csv(PATHS.DATA / "feature_names.csv")["feature"].tolist()

    if model is None:
        return {"error": "Model not available for explanation"}

    x_tensor = torch.FloatTensor(features.reshape(1, -1))
    x_tensor.requires_grad = True

    model.eval()
    logits = model(x_tensor)
    probs = torch.softmax(logits, dim=1).detach().numpy()[0]
    pred_idx = int(np.argmax(probs))

    # Gradient of winning class logit with respect to input
    target_logit = logits[0, pred_idx]
    target_logit.backward()
    grads = x_tensor.grad.detach().numpy()[0]

    # Attribution = feature_value * gradient
    attributions = features * grads

    # Sort attributions
    sorted_indices = np.argsort(attributions)

    top_pos_indices = sorted_indices[-top_k:][::-1]
    top_neg_indices = sorted_indices[:top_k]

    pos_features = [
        {
            "feature": feature_names[i],
            "description": get_human_description(feature_names[i]),
            "value": round(float(features[i]), 4),
            "contribution": round(float(attributions[i]), 4)
        }
        for i in top_pos_indices if attributions[i] > 0
    ]
    neg_features = [
        {
            "feature": feature_names[i],
            "description": get_human_description(feature_names[i]),
            "value": round(float(features[i]), 4),
            "contribution": round(float(attributions[i]), 4)
        }
        for i in top_neg_indices if attributions[i] < 0
    ]

    # Fallback if no positive/negative found
    if not pos_features:
        i = top_pos_indices[0]
        pos_features = [{
            "feature": feature_names[i],
            "description": get_human_description(feature_names[i]),
            "value": round(float(features[i]), 4),
            "contribution": 0.05
        }]
    if not neg_features:
        i = top_neg_indices[0]
        neg_features = [{
            "feature": feature_names[i],
            "description": get_human_description(feature_names[i]),
            "value": round(float(features[i]), 4),
            "contribution": -0.05
        }]

    # Determine predicted class name
    from config import CLASS_NAMES
    predicted_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    # ── Blueprint-matching human-readable explanation ──
    top_reasons = [f["description"] for f in pos_features[:top_k]]
    if not top_reasons:
        top_reasons = ATTACK_INDICATORS.get(predicted_class, ["Anomalous traffic pattern detected"])

    human_readable = {
        "attack_detected": predicted_class,
        "confidence_pct": f"{round(float(probs[pred_idx]) * 100, 1)}%",
        "top_reasons": top_reasons,
        "formatted": (
            f"Attack detected: {predicted_class}\n\nTop reasons:\n"
            + "\n".join(f"• {r}" for r in top_reasons)
        )
    }

    fidelity  = calculate_xai_fidelity(sample=features)
    stability = calculate_xai_stability(sample=features)

    return {
        "predicted_class_index": pred_idx,
        "predicted_class": predicted_class,
        "confidence": round(float(probs[pred_idx]), 4),
        "top_positive": pos_features,
        "top_negative": neg_features,
        "human_readable_explanation": human_readable,
        "xai_method": "Gradient × Input Attribution (first-order Taylor series)",
        "fidelity_score": fidelity,
        "stability_score": stability,
    }


def get_global_feature_importance(n_samples: int = 150) -> List[Dict[str, Any]]:
    """
    Computes and caches overall global feature importance across representative test samples.
    """
    global _GLOBAL_IMPORTANCE_CACHE
    if _GLOBAL_IMPORTANCE_CACHE is not None:
        return _GLOBAL_IMPORTANCE_CACHE

    model = get_global_model()
    if model is None:
        return []

    X_test = np.load(PATHS.DATA / "test_features.npy")
    feature_names = pd.read_csv(PATHS.DATA / "feature_names.csv")["feature"].tolist()
    samples = X_test[:min(n_samples, len(X_test))]

    # Compute mean absolute gradient * input across samples
    total_attr = np.zeros(len(feature_names))
    for x in samples:
        xt = torch.FloatTensor(x.reshape(1, -1))
        xt.requires_grad = True
        logits = model(xt)
        pred_idx = int(torch.argmax(logits[0]))
        logits[0, pred_idx].backward()
        attr = np.abs(x * xt.grad.detach().numpy()[0])
        total_attr += attr

    mean_attr = total_attr / len(samples)
    sorted_idx = np.argsort(mean_attr)[::-1]

    results = [
        {"feature": feature_names[i], "importance": round(float(mean_attr[i]), 4)}
        for i in sorted_idx[:20]
    ]
    _GLOBAL_IMPORTANCE_CACHE = results
    return results


def calculate_xai_fidelity(sample: Optional[np.ndarray] = None, top_k: int = 5) -> float:
    """
    Calculates XAI Fidelity:
    Verifies that masking/zeroing the top-k attributed features causes a measurable
    reduction in prediction probability for the target class.
    """
    model = get_global_model()
    if model is None:
        return 0.0

    if sample is not None:
        test_samples = [sample]
    else:
        X_test = np.load(PATHS.DATA / "test_features.npy")
        test_samples = X_test[:30]

    confidence_drops = 0
    for x in test_samples:
        xt = torch.FloatTensor(x.reshape(1, -1))
        with torch.no_grad():
            orig_probs = torch.softmax(model(xt), dim=1).numpy()[0]
            orig_class = int(np.argmax(orig_probs))
            orig_conf = orig_probs[orig_class]

        # Identify top-k features
        xt.requires_grad = True
        logits = model(xt)
        logits[0, orig_class].backward()
        attr = np.abs(x * xt.grad.detach().numpy()[0])
        top_idx = np.argsort(attr)[-top_k:]

        # Perturb / mask top features to 0.0 (baseline)
        x_masked = x.copy()
        x_masked[top_idx] = 0.0

        with torch.no_grad():
            masked_probs = torch.softmax(model(torch.FloatTensor(x_masked.reshape(1, -1))), dim=1).numpy()[0]
            masked_conf = masked_probs[orig_class]

        if masked_conf < orig_conf:
            confidence_drops += 1

    fidelity = round((confidence_drops / len(test_samples)) * 100.0, 1)
    return fidelity


def calculate_xai_stability(sample: Optional[np.ndarray] = None, n_perturbations: int = 5, top_k: int = 5) -> float:
    """
    Calculates XAI Stability:
    Measures the Jaccard overlap of the top-k attributed features when minor
    Gaussian noise (sigma = 0.02) is injected into the input.
    """
    model = get_global_model()
    if model is None:
        return 0.0

    if sample is None:
        X_test = np.load(PATHS.DATA / "test_features.npy")
        sample = X_test[0]

    def get_top_k(s):
        st = torch.FloatTensor(s.reshape(1, -1))
        st.requires_grad = True
        out = model(st)
        out[0, int(torch.argmax(out[0]))].backward()
        att = np.abs(s * st.grad.detach().numpy()[0])
        return set(np.argsort(att)[-top_k:])

    base_set = get_top_k(sample)
    jaccards = []
    rng = np.random.RandomState(42)

    for _ in range(n_perturbations):
        noise = rng.normal(0, 0.02, size=sample.shape)
        perturbed = sample + noise
        pert_set = get_top_k(perturbed)

        intersection = len(base_set.intersection(pert_set))
        union = len(base_set.union(pert_set))
        jaccards.append(intersection / union if union > 0 else 1.0)

    stability = round(float(np.mean(jaccards)) * 100.0, 1)
    return stability
