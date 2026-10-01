"""
config.py
=========
Central configuration for the Federated IDS project.
All paths are relative to the project root (federated_ids/).
Import this module instead of hardcoding values anywhere.

Usage:
    from config import PATHS, FEDERATED, TRAINING, CLASSES
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# ROOT — dynamically resolved (no absolute paths)
# ──────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
class PATHS:
    ROOT          = ROOT

    # Data
    DATA          = ROOT / "data"
    RAW_CSVS      = [
        ROOT / "data" / "Monday.csv",
        ROOT / "data" / "Tuesday.csv",
        ROOT / "data" / "Wednesday.csv",
        ROOT / "data" / "Thursday1.csv",
        ROOT / "data" / "Thursday2.csv",
        ROOT / "data" / "Friday.csv",
        ROOT / "data" / "Friday1.csv",
    ]
    COMBINED_FEATURES = ROOT / "data" / "combined_features.csv"
    COMBINED_LABELS   = ROOT / "data" / "combined_labels.csv"

    # IoT client data (5 devices)
    CAMERA_DATA    = ROOT / "data" / "camera_data.csv"
    ROUTER_DATA    = ROOT / "data" / "router_data.csv"
    SENSOR_DATA    = ROOT / "data" / "sensor_data.csv"
    DOORLOCK_DATA  = ROOT / "data" / "door_lock_data.csv"
    LIGHTING_DATA  = ROOT / "data" / "smart_lighting_data.csv"
    # Legacy alias
    SMARTLOCK_DATA = DOORLOCK_DATA

    # Processed multi-class data (Semester 2)
    S2_FEATURES    = ROOT / "data" / "s2_features.csv"
    S2_LABELS      = ROOT / "data" / "s2_labels.csv"

    # Models
    MODELS         = ROOT / "models"
    SEM1_MODELS    = ROOT / "models" / "semester1"
    SCALER         = ROOT / "models" / "scaler.pkl"
    LABEL_ENCODER  = ROOT / "models" / "label_encoder.pkl"
    GLOBAL_MODEL   = ROOT / "models" / "federated_global_model.pth"

    # Centralized Semester 2 models
    RF_MODEL       = ROOT / "models" / "s2_random_forest.pkl"
    XGB_MODEL      = ROOT / "models" / "s2_xgboost.pkl"
    NN_CENTRAL     = ROOT / "models" / "s2_neural_network.pth"

    # Results
    RESULTS        = ROOT / "results"
    SEM1_BASELINE  = ROOT / "results" / "semester1_baseline"
    FIGURES        = ROOT / "results" / "figures"
    CONF_MATRICES  = ROOT / "results" / "confusion_matrices"
    CLASS_REPORTS  = ROOT / "results" / "classification_reports"

    # Key result files
    CLIENT_DIST_CSV       = ROOT / "results" / "client_distribution.csv"
    FED_ROUNDS_CSV        = ROOT / "results" / "federated_round_results.csv"
    FINAL_COMPARISON_CSV  = ROOT / "results" / "final_comparison.csv"
    EXPERIMENT_SUMMARY    = ROOT / "results" / "experiment_summary.txt"

    # Notebooks
    NOTEBOOKS      = ROOT / "notebooks"

    @classmethod
    def ensure_dirs(cls):
        """Create all required output directories if they don't exist."""
        for d in [
            cls.DATA, cls.MODELS, cls.SEM1_MODELS,
            cls.RESULTS, cls.SEM1_BASELINE,
            cls.FIGURES, cls.CONF_MATRICES, cls.CLASS_REPORTS,
        ]:
            os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────
RANDOM_STATE = 42


# ──────────────────────────────────────────────
# DATASET SAMPLING
# (Semester 1 used 20,000; Semester 2 uses more)
# ──────────────────────────────────────────────
SAMPLE_SIZE = 50_000   # rows to sample from combined CICIDS2017 data


# ──────────────────────────────────────────────
# MULTI-CLASS LABEL MAPPING
# Maps CICIDS2017 raw labels → standard categories.
# Only labels confirmed to exist in the dataset are listed here.
# Built after inspection of all raw CSVs.
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# ACTUAL CICIDS2017 LABEL INSPECTION RESULTS (2026-09-03)
# ──────────────────────────────────────────────
# Labels found in raw CSVs (Monday excluded — BENIGN only):
#   BENIGN                  1,554,112
#   DoS Hulk                  231,073  → DDoS
#   PortScan                  158,930  → PortScan
#   DDoS                      128,027  → DDoS
#   DoS GoldenEye              10,293  → DDoS
#   FTP-Patator                 7,938  → BruteForce
#   SSH-Patator                 5,897  → BruteForce
#   DoS slowloris               5,796  → DDoS
#   DoS Slowhttptest            5,499  → DDoS
#   Web Attack – Brute Force    1,507  → WebAttack
#   Web Attack – XSS              652  → WebAttack
#   Infiltration                   36  → EXCLUDED (<50 samples)
#   Web Attack – Sql Injection     21  → WebAttack
#   Heartbleed                     11  → EXCLUDED (<50 samples)
#
# NOTE: 'Bot' class does NOT exist in this CICIDS2017 download.
# ──────────────────────────────────────────────
LABEL_MAP = {
    # Normal traffic
    "BENIGN":                           "Normal",

    # DDoS variants (all DoS/DDoS attacks merged into DDoS class)
    "DDoS":                             "DDoS",
    "DoS Hulk":                         "DDoS",
    "DoS GoldenEye":                    "DDoS",
    "DoS slowloris":                    "DDoS",
    "DoS Slowhttptest":                 "DDoS",

    # Port scanning
    "PortScan":                         "PortScan",

    # Brute-force credential attacks
    "FTP-Patator":                      "BruteForce",
    "SSH-Patator":                      "BruteForce",

    # Web attacks — actual separator is UTF-8 replacement char \ufffd (0xef 0xbf 0xbd)
    # Confirmed by byte inspection of Thursday1.csv
    "Web Attack \ufffd Brute Force":   "WebAttack",
    "Web Attack \ufffd XSS":           "WebAttack",
    "Web Attack \ufffd Sql Injection": "WebAttack",
    # Backup variants for different encoding environments
    "Web Attack \x96 Brute Force":     "WebAttack",
    "Web Attack \x96 XSS":             "WebAttack",
    "Web Attack \x96 Sql Injection":   "WebAttack",
    "Web Attack \u2013 Brute Force":   "WebAttack",
    "Web Attack \u2013 XSS":           "WebAttack",
    "Web Attack \u2013 Sql Injection": "WebAttack",

    # Botnet variants (supported in architecture and simulation)
    "Bot":                              "Botnet",
    "Botnet":                           "Botnet",

    # Excluded: Infiltration (36 samples) and Heartbleed (11 samples)
    # Both are below MIN_CLASS_SAMPLES and cannot be reliably learned.
    # "Infiltration":   EXCLUDED
    # "Heartbleed":     EXCLUDED
}

# 6 Target attack classes for the IDS platform
CLASS_NAMES = ["Normal", "DDoS", "Botnet", "PortScan", "BruteForce", "WebAttack"]

# Real CICIDS2017 classes present in uploaded raw CSV shards
REAL_CLASSES_PRESENT = ["Normal", "DDoS", "PortScan", "BruteForce", "WebAttack"]
SYNTHETIC_CLASSES = ["Botnet"]
DATASET_DESCRIPTION = "CICIDS2017-derived benchmark with calibrated synthetic augmentation (Botnet class calibrated from empirical IRC-C2 flow profiles due to omitted raw Friday morning capture)"


# Consistent UI / chart attack colors
ATTACK_COLORS = {
    "Normal":     "#06d6a0", # Green
    "DDoS":       "#ffd166", # Yellow
    "PortScan":   "#f72585", # Pink
    "Botnet":     "#7209b7", # Purple
    "BruteForce": "#f3722c", # Orange
    "WebAttack":  "#4cc9f0", # Blue
}

# Minimum required samples per class to include it in training
MIN_CLASS_SAMPLES = 50


# ──────────────────────────────────────────────
# TRAINING — CENTRALIZED MODELS
# ──────────────────────────────────────────────
class TRAINING:
    TEST_SIZE       = 0.2
    VAL_SIZE        = 0.1      # fraction of training set
    RANDOM_STATE    = RANDOM_STATE

    # Random Forest
    RF_ESTIMATORS   = 100

    # XGBoost
    XGB_ESTIMATORS  = 100
    XGB_MAX_DEPTH   = 6
    XGB_LR          = 0.1

    # PyTorch Neural Network (Centralized)
    NN_HIDDEN       = [256, 128, 64]
    NN_EPOCHS       = 30
    NN_BATCH_SIZE   = 512
    NN_LR           = 0.001
    NN_DROPOUT      = 0.3


# ──────────────────────────────────────────────
# FEDERATED LEARNING
# ──────────────────────────────────────────────
class FEDERATED:
    NUM_ROUNDS      = 10        # global communication rounds
    LOCAL_EPOCHS    = 5         # local training epochs per round
    BATCH_SIZE      = 256
    LR              = 0.001
    RANDOM_STATE    = RANDOM_STATE

    # 5 IoT client names (must match data file names)
    CLIENTS = ["camera", "router", "sensor", "door_lock", "smart_lighting"]

    # Architecture (shared by all clients + server)
    HIDDEN_LAYERS   = [256, 128, 64]
    DROPOUT         = 0.3


# ──────────────────────────────────────────────
# NON-IID CLIENT DISTRIBUTION
# Preferred attack emphasis per IoT device type.
# ──────────────────────────────────────────────
CLIENT_ATTACK_EMPHASIS = {
    # Camera: DDoS-heavy (network flooding toward camera streams)
    "camera":         ["DDoS"],
    # Router: PortScan-heavy (reconnaissance of router endpoints)
    "router":         ["PortScan"],
    # Sensor: Botnet-heavy (compromised IoT sensors in botnet C2)
    "sensor":         ["Botnet", "PortScan"],
    # Door Lock: BruteForce-heavy (credential attacks against smart locks)
    "door_lock":      ["BruteForce"],
    # Smart Lighting: WebAttack-heavy (exploits against HTTP API controllers)
    "smart_lighting": ["WebAttack"],
}

# 4 Risk Management Tiers (Software Mitigation Simulation)
RISK_TIERS = {
    "MONITOR":              (0.0, 30.0),
    "ALERT":                (30.0, 60.0),
    "SIMULATED QUARANTINE": (60.0, 85.0),
    "SIMULATED ISOLATION":  (85.0, 100.0),
}


# ──────────────────────────────────────────────
# REAL-TIME SIMULATION
# ──────────────────────────────────────────────
class REALTIME:
    FEATURE_COUNT   = None      # set dynamically from data
    SIM_DELAY_SEC   = 0.5       # seconds between simulated packets
    BATCH_SIZE      = 10        # features per simulation batch


# ──────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────
class DASHBOARD:
    HOST            = "0.0.0.0"
    PORT            = 5000
    DEBUG           = False
    MAX_ALERTS      = 100       # max alerts to keep in memory


# ──────────────────────────────────────────────
# SEVERITY THRESHOLDS
# Maps (model prediction class, confidence) → severity
# ──────────────────────────────────────────────
SEVERITY_MAP = {
    "Normal":      "LOW",
    "DDoS":        "CRITICAL",
    "PortScan":    "HIGH",
    "Botnet":      "HIGH",
    "Bot":         "HIGH",
    "BruteForce":  "HIGH",
    "WebAttack":   "MEDIUM",
    "Infiltration":"CRITICAL",
}

# Confidence threshold below which prediction is flagged SUSPICIOUS
LOW_CONFIDENCE_THRESHOLD = 0.60
