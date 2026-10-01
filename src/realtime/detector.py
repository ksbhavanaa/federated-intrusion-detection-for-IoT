"""
src/realtime/detector.py
=========================
Real-time intrusion detection engine wrapping the global Federated MLP.
Performs live inference, high-resolution latency measurement, dynamic risk calculation,
software-level mitigation state assignment, and SQLite persistence.

Mitigation State Tiers:
  - MONITOR (<30): Normal baseline activity
  - ALERT (30-59): Suspicious anomaly or low-confidence event
  - SIMULATED QUARANTINE (60-84): High-confidence attack on standard IoT node
  - SIMULATED ISOLATION (85-100): Critical attack on high-priority infrastructure
"""

import sys
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Union, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import pandas as pd
import joblib

from config import PATHS, SEVERITY_MAP, LOW_CONFIDENCE_THRESHOLD, CLASS_NAMES
from src.federated.fed_model import FederatedMLP
from src.database.db import insert_alert, get_db_connection

log = logging.getLogger(__name__)

DEVICE_CRITICALITY = {
    "router": 1.25,        # Gateway/Router compromise affects entire network
    "door_lock": 1.20,     # Physical security perimeter
    "camera": 1.00,        # Video surveillance endpoint
    "sensor": 0.90,        # Environmental telemetry
    "smart_lighting": 0.85 # Ambient controller
}

ATTACK_SEVERITY_WEIGHT = {
    "Normal": 0.05,
    "WebAttack": 0.65,
    "PortScan": 0.70,
    "Botnet": 0.85,
    "BruteForce": 0.85,
    "DDoS": 0.95
}


class IDSDetector:
    """Production-grade real-time detector wrapping the trained Federated MLP."""

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.encoder = None
        self.classes = []
        self.feature_names = []
        self._load_model(model_path)

    def _load_model(self, model_path=None):
        """Loads global model and label encoder."""
        path = model_path or PATHS.GLOBAL_MODEL
        if not path.exists() and PATHS.NN_CENTRAL.exists():
            log.warning(f"Global model not found at {path}, falling back to {PATHS.NN_CENTRAL}")
            path = PATHS.NN_CENTRAL

        if not path.exists():
            log.error(f"No trained model found at {path}. Run training first.")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.n_features = checkpoint['n_features']
            self.n_classes = checkpoint['n_classes']
            self.classes = checkpoint.get('class_names', CLASS_NAMES[:self.n_classes])

            self.model = FederatedMLP(n_features=self.n_features, n_classes=self.n_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            self.encoder = joblib.load(PATHS.LABEL_ENCODER)
            feat_df = pd.read_csv(PATHS.DATA / "feature_names.csv")
            self.feature_names = feat_df["feature"].tolist()
            log.info(f"Detector initialized with {path.name} ({self.n_features} features, {self.n_classes} classes).")
        except Exception as e:
            log.error(f"Failed to initialize detector: {e}")

    def is_ready(self) -> bool:
        return self.model is not None

    def calculate_risk_score(
        self,
        predicted_class: str,
        confidence: float,
        device_id: str,
        recent_threat_count: int = 0
    ) -> Tuple[float, str]:
        """
        Calculates mathematically justified dynamic risk score (0-100).
        Formula:
          Base = SeverityWeight * Confidence * Criticality * 80.0
          FrequencyBonus = min(20.0, recent_threat_count * 2.5)
          TotalRisk = min(100.0, max(5.0, Base + FrequencyBonus))
        """
        if predicted_class == "Normal":
            # Normal traffic: low baseline risk modulated inversely by confidence
            risk = max(5.0, round(25.0 * (1.0 - confidence), 1))
            return risk, "MONITOR"

        crit = DEVICE_CRITICALITY.get(device_id.lower(), 1.0)
        sev = ATTACK_SEVERITY_WEIGHT.get(predicted_class, 0.7)
        freq_bonus = min(20.0, recent_threat_count * 2.5)

        raw_score = (sev * confidence * crit * 80.0) + freq_bonus
        risk = round(float(np.clip(raw_score, 10.0, 100.0)), 1)

        # Mitigation state tiers
        if risk < 30.0:
            mitigation = "MONITOR"
        elif risk < 60.0:
            mitigation = "ALERT"
        elif risk < 85.0:
            mitigation = "SIMULATED QUARANTINE"
        else:
            mitigation = "SIMULATED ISOLATION"

        return risk, mitigation

    def predict(
        self,
        features: np.ndarray,
        device_id: str = "camera",
        device_name: str = "Smart Camera",
        source: str = "BACKGROUND_SIMULATOR",
        persist: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on feature vector, measuring high-resolution latency.
        """
        if self.model is None:
            return {"error": "Model not loaded", "status": "ERROR"}

        if features.ndim == 1:
            features_tensor = torch.FloatTensor(features.reshape(1, -1)).to(self.device)
        else:
            features_tensor = torch.FloatTensor(features).to(self.device)

        # High-resolution inference timer
        t_start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        pred_class = self.classes[pred_idx] if pred_idx < len(self.classes) else str(pred_idx)

        # Fetch recent device threat count from SQLite for frequency bonus
        threat_count = 0
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT active_threats FROM devices WHERE id = ?;", (device_id,))
            row = cur.fetchone()
            if row:
                threat_count = row["active_threats"]
            conn.close()
        except Exception:
            pass

        risk_score, mitigation_state = self.calculate_risk_score(
            predicted_class=pred_class,
            confidence=conf,
            device_id=device_id,
            recent_threat_count=threat_count
        )

        severity = SEVERITY_MAP.get(pred_class, "MEDIUM")
        status = "ALERT" if pred_class != "Normal" else "OK"

        # Feature summary for explanation reference
        feature_dict = {}
        if len(self.feature_names) == len(features):
            top_indices = np.argsort(np.abs(features))[-5:][::-1]
            for idx in top_indices:
                feature_dict[self.feature_names[idx]] = float(features[idx])

        alert_code = None
        if persist and (status == "ALERT" or source == "MANUAL_SIMULATION"):
            alert_code = insert_alert(
                device_id=device_id,
                device_name=device_name,
                attack_type=pred_class,
                confidence=conf,
                risk_score=risk_score,
                severity=severity,
                mitigation_state=mitigation_state,
                source=source,
                latency_ms=latency_ms,
                feature_summary=feature_dict
            )

        return {
            "alert_code": alert_code or f"#PRED-{int(time.time()*1000)%10000}",
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "device": device_name,
            "prediction": pred_class,
            "confidence": round(conf, 4),
            "risk_score": risk_score,
            "severity": severity,
            "mitigation": mitigation_state,
            "latency_ms": latency_ms,
            "source": source,
            "status": status,
            "feature_vector": features.tolist()
        }
