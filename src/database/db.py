"""
src/database/db.py
===================
SQLite database interface for the Federated IoT IDS research platform.
Persists all experiments, rounds, devices, alerts, predictions, latencies,
mitigations, and XAI results across server restarts and page refreshes.

Database file: results/ids_research.db
"""

import sqlite3
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "ids_research.db"


def get_db_connection() -> sqlite3.Connection:
    """Create and return a thread-safe connection with row_factory set."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    """Create all 13 research database tables and initial device records."""
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Users / System Operator (with hashed password for Flask-Login)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL DEFAULT '',
        role TEXT DEFAULT 'SOC_ANALYST',
        is_active INTEGER DEFAULT 1,
        created_at TEXT NOT NULL
    );
    """)

    # Migrate: add password_hash column if upgrading from old schema
    try:
        cur.execute("ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT '';")
    except Exception:
        pass  # Column already exists
    try:
        cur.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1;")
    except Exception:
        pass

    # 2. IoT Devices (5 logical/simulated clients)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS devices (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        device_type TEXT NOT NULL,
        ip_address TEXT NOT NULL,
        mac_address TEXT NOT NULL,
        status TEXT DEFAULT 'ONLINE',
        risk_score REAL DEFAULT 15.0,
        risk_tier TEXT DEFAULT 'MONITOR',
        active_threats INTEGER DEFAULT 0,
        total_detections INTEGER DEFAULT 0,
        mitigation_state TEXT DEFAULT 'NORMAL_OPERATION',
        last_detection TEXT,
        attack_emphasis TEXT,
        updated_at TEXT NOT NULL
    );
    """)

    # 3. Predictions (Inference telemetry)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        device_id TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        latency_ms REAL NOT NULL,
        source TEXT NOT NULL,
        model_version TEXT NOT NULL,
        feature_vector_json TEXT,
        FOREIGN KEY (device_id) REFERENCES devices(id)
    );
    """)

    # 4. Security Alerts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_code TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        device_id TEXT NOT NULL,
        device_name TEXT NOT NULL,
        attack_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        risk_score REAL NOT NULL,
        severity TEXT NOT NULL,
        mitigation_state TEXT NOT NULL,
        source TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        status TEXT DEFAULT 'ACTIVE',
        feature_summary TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (device_id) REFERENCES devices(id)
    );
    """)

    # 5. Federated Communication Rounds
    cur.execute("""
    CREATE TABLE IF NOT EXISTS federated_rounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL,
        round_number INTEGER NOT NULL,
        global_accuracy REAL NOT NULL,
        macro_f1 REAL NOT NULL,
        weighted_f1 REAL NOT NULL,
        precision_weighted REAL,
        recall_weighted REAL,
        loss REAL NOT NULL,
        participating_clients TEXT NOT NULL,
        dropped_clients TEXT,
        training_time_sec REAL NOT NULL,
        communication_bytes INTEGER NOT NULL,
        communication_kb REAL NOT NULL,
        model_size_kb REAL NOT NULL,
        aggregation_method TEXT DEFAULT 'Sample-Weighted FedAvg',
        timestamp TEXT NOT NULL
    );
    """)

    # 6. Client Metrics per Round
    cur.execute("""
    CREATE TABLE IF NOT EXISTS client_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL,
        round_number INTEGER NOT NULL,
        client_id TEXT NOT NULL,
        train_samples INTEGER NOT NULL,
        weight REAL NOT NULL,
        train_loss REAL NOT NULL,
        val_accuracy REAL NOT NULL,
        timestamp TEXT NOT NULL
    );
    """)

    # 7. Federated Experiments (High-level run records)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fl_experiments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        distribution_mode TEXT NOT NULL, -- IID or NON_IID
        aggregation_protocol TEXT NOT NULL, -- FEDAVG, SECURE_AGG, DROPOUT
        num_clients INTEGER NOT NULL,
        num_rounds INTEGER NOT NULL,
        final_accuracy REAL,
        final_macro_f1 REAL,
        final_weighted_f1 REAL,
        total_communication_kb REAL,
        total_duration_sec REAL,
        status TEXT DEFAULT 'COMPLETED',
        created_at TEXT NOT NULL
    );
    """)

    # 8. Detection Latency Benchmarks
    cur.execute("""
    CREATE TABLE IF NOT EXISTS detection_latency (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        batch_size INTEGER DEFAULT 1,
        mean_ms REAL NOT NULL,
        median_ms REAL NOT NULL,
        p95_ms REAL NOT NULL,
        min_ms REAL NOT NULL,
        max_ms REAL NOT NULL,
        throughput_fps REAL NOT NULL,
        sample_count INTEGER NOT NULL,
        timestamp TEXT NOT NULL
    );
    """)

    # 9. XAI Results (SHAP explanations & metrics)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS xai_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_code TEXT UNIQUE NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        top_positive_features_json TEXT NOT NULL,
        top_negative_features_json TEXT NOT NULL,
        shap_values_json TEXT,
        fidelity_score REAL,
        stability_score REAL,
        computed_at TEXT NOT NULL
    );
    """)

    # 10. Mitigation History
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mitigation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_code TEXT NOT NULL,
        device_id TEXT NOT NULL,
        risk_score REAL NOT NULL,
        action TEXT NOT NULL,
        decision_rationale TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (device_id) REFERENCES devices(id)
    );
    """)

    # 11. Concept Drift Experiments
    cur.execute("""
    CREATE TABLE IF NOT EXISTS concept_drift_experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL,
        stage INTEGER NOT NULL,
        stage_name TEXT NOT NULL,
        attacks_present TEXT NOT NULL,
        before_drift_acc REAL,
        before_drift_macro_f1 REAL,
        during_drift_acc REAL,
        during_drift_macro_f1 REAL,
        post_adapt_acc REAL,
        post_adapt_macro_f1 REAL,
        adaptation_epochs INTEGER,
        timestamp TEXT NOT NULL
    );
    """)

    # 12. Scalability Experiments
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scalability_experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL,
        client_count INTEGER NOT NULL,
        rounds INTEGER NOT NULL,
        training_time_sec REAL NOT NULL,
        communication_kb REAL NOT NULL,
        final_accuracy REAL NOT NULL,
        final_macro_f1 REAL NOT NULL,
        timestamp TEXT NOT NULL
    );
    """)

    # 13. Research Experiments Catalog (Unified Experiment Registry)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS research_experiments (
        id TEXT PRIMARY KEY,
        experiment_name TEXT NOT NULL,
        category TEXT NOT NULL, -- CENTRALIZED, FL_IID, FL_NON_IID, SECURE_AGG, DROPOUT, DRIFT, SCALABILITY
        model_type TEXT NOT NULL,
        dataset_name TEXT NOT NULL,
        configuration_json TEXT NOT NULL,
        results_json TEXT NOT NULL,
        status TEXT DEFAULT 'COMPLETED',
        created_at TEXT NOT NULL
    );
    """)

    # Create indexes for fast lookup
    cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_device ON alerts(device_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fl_rounds_exp ON federated_rounds(experiment_id);")

    # Seed 5 IoT devices if not present
    default_devices = [
        ("camera", "Smart Camera", "Camera", "192.168.1.101", "00:1A:2B:3C:4D:01", "Normal + DDoS"),
        ("router", "WiFi Router", "Router", "192.168.1.1", "00:1A:2B:3C:4D:02", "Normal + PortScan"),
        ("sensor", "Temperature Sensor", "Sensor", "192.168.1.103", "00:1A:2B:3C:4D:03", "Normal + Botnet"),
        ("door_lock", "Smart Door Lock", "Door Lock", "192.168.1.104", "00:1A:2B:3C:4D:04", "Normal + BruteForce"),
        ("smart_lighting", "Smart Light Controller", "Smart Lighting", "192.168.1.105", "00:1A:2B:3C:4D:05", "Normal + WebAttack"),
    ]

    for dev_id, name, dtype, ip, mac, emphasis in default_devices:
        cur.execute("""
        INSERT OR IGNORE INTO devices (id, name, device_type, ip_address, mac_address, attack_emphasis, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (dev_id, name, dtype, ip, mac, emphasis, datetime.now().isoformat()))

    conn.commit()
    conn.close()
    log.info(f"Initialized SQLite research database at {DB_PATH}")


# ── Database Helpers ────────────────────────────────────────────────────────

def insert_alert(
    device_id: str,
    device_name: str,
    attack_type: str,
    confidence: float,
    risk_score: float,
    severity: str,
    mitigation_state: str,
    source: str,
    latency_ms: float,
    feature_summary: Optional[Dict[str, Any]] = None
) -> str:
    """Insert a new threat alert and return the generated unique alert code (e.g., #1001)."""
    conn = get_db_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    # Get next sequential ID starting from #1001
    cur.execute("SELECT COALESCE(MAX(CAST(SUBSTR(alert_code, 2) AS INTEGER)), 1000) + 1 FROM alerts WHERE alert_code LIKE '#%';")
    next_id = cur.fetchone()[0]
    if next_id < 1001:
        next_id = 1001
    alert_code = f"#{next_id}"

    cur.execute("""
    INSERT INTO alerts (
        alert_code, timestamp, device_id, device_name, attack_type,
        confidence, risk_score, severity, mitigation_state, source,
        latency_ms, feature_summary, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        alert_code, now, device_id, device_name, attack_type,
        confidence, risk_score, severity, mitigation_state, source,
        latency_ms, json.dumps(feature_summary or {}), now
    ))

    # Also record prediction
    cur.execute("""
    INSERT INTO predictions (timestamp, device_id, predicted_class, confidence, latency_ms, source, model_version)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (now, device_id, attack_type, confidence, latency_ms, source, "FederatedMLP-v2"))

    # Update device dynamic metrics
    cur.execute("""
    UPDATE devices SET
        risk_score = ?,
        risk_tier = ?,
        active_threats = active_threats + 1,
        total_detections = total_detections + 1,
        mitigation_state = ?,
        last_detection = ?,
        updated_at = ?
    WHERE id = ?;
    """, (risk_score, _calc_tier(risk_score), mitigation_state, now, now, device_id))

    # Record mitigation action
    cur.execute("""
    INSERT INTO mitigation_history (alert_code, device_id, risk_score, action, decision_rationale, timestamp)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (
        alert_code, device_id, risk_score, mitigation_state,
        f"High confidence ({confidence*100:.1f}%) {attack_type} detection on {device_name}. Risk score: {risk_score:.1f}.",
        now
    ))

    conn.commit()
    conn.close()
    return alert_code


def _calc_tier(risk_score: float) -> str:
    if risk_score < 30.0:
        return "MONITOR"
    elif risk_score < 60.0:
        return "ALERT"
    elif risk_score < 85.0:
        return "SIMULATED QUARANTINE"
    else:
        return "SIMULATED ISOLATION"


def get_all_devices() -> List[Dict[str, Any]]:
    """Fetch list of all 5 IoT devices with live risk scores and mitigation states."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM devices ORDER BY id ASC;")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_recent_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch most recent alerts."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?;", (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_experiment_history() -> List[Dict[str, Any]]:
    """Fetch completed research experiments."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM research_experiments ORDER BY created_at DESC;")
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        d['configuration'] = json.loads(d['configuration_json']) if d.get('configuration_json') else {}
        d['results'] = json.loads(d['results_json']) if d.get('results_json') else {}
        rows.append(d)
    conn.close()
    return rows


def save_research_experiment(
    exp_id: str,
    name: str,
    category: str,
    model_type: str,
    dataset_name: str,
    config: Dict[str, Any],
    results: Dict[str, Any]
):
    """Insert or replace research experiment record."""
    conn = get_db_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute("""
    INSERT OR REPLACE INTO research_experiments (
        id, experiment_name, category, model_type, dataset_name,
        configuration_json, results_json, status, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'COMPLETED', ?);
    """, (exp_id, name, category, model_type, dataset_name, json.dumps(config), json.dumps(results), now))
    conn.commit()
    conn.close()


# Initialize database upon import
init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Auth helper functions (used by Flask-Login)
# ─────────────────────────────────────────────────────────────────────────────

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Fetch a user record by username. Returns None if not found."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?;", (username,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a user record by primary key ID. Returns None if not found."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?;", (int(user_id),))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(username: str, password: str, role: str = "SOC_ANALYST") -> bool:
    """
    Create a new user with a bcrypt-hashed password.
    Returns True on success, False if username already exists.
    """
    from werkzeug.security import generate_password_hash
    pw_hash = generate_password_hash(password)
    now = datetime.now().isoformat()
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, role, is_active, created_at) VALUES (?, ?, ?, 1, ?);",
            (username, pw_hash, role, now)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        log.warning(f"create_user failed for '{username}': {e}")
        return False


def verify_user_password(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Verify login credentials.
    Returns the user dict on success, None on failure.
    """
    from werkzeug.security import check_password_hash
    user = get_user_by_username(username)
    if user is None:
        return None
    if not user.get("password_hash"):
        return None
    if check_password_hash(user["password_hash"], password):
        return user
    return None


def ensure_default_admin():
    """
    Seed the database with a default admin account if none exists.
    Default credentials:  admin / admin123
    CHANGE THIS PASSWORD after first login.
    """
    existing = get_user_by_username("admin")
    if existing is None:
        created = create_user("admin", "admin123", role="SOC_ADMIN")
        if created:
            log.info("Default admin user created: username='admin', password='admin123'")
            log.warning("SECURITY: Change the default admin password immediately!")
    # Also seed a default analyst
    if get_user_by_username("analyst") is None:
        create_user("analyst", "analyst123", role="SOC_ANALYST")
        log.info("Default analyst user created: username='analyst', password='analyst123'")


# Seed default users on import
ensure_default_admin()
