"""
tests/test_experiments.py
==========================
Automated test suite verifying the research experiment engine:
  - 5 IoT client data sharding & persistence
  - Secure Aggregation simulation (pairwise zero-sum random masking ||W_masked - W_unmasked|| < 1e-6)
  - Client dropout resilience
  - SQLite research database persistence
  - Dynamic risk score state machine
  - PDF report generation
"""

import sys
import os
import sqlite3
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import PATHS, FEDERATED
from src.database.db import get_db_connection, init_db, insert_alert, get_all_devices
from src.federated.fed_model import FederatedMLP, fedavg, get_model_params, set_model_params
from src.federated.experiments import ResearchExperimentEngine
from src.reports.report_generator import generate_pdf_report


def test_sqlite_database_tables_exist():
    """Verify that all 13 research database tables exist in results/ids_research.db."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()

    required_tables = [
        "users", "devices", "predictions", "alerts", "federated_rounds",
        "client_metrics", "fl_experiments", "detection_latency", "xai_results",
        "mitigation_history", "concept_drift_experiments", "scalability_experiments",
        "research_experiments"
    ]
    for table in required_tables:
        assert table in tables, f"Missing required database table: {table}"


def test_five_devices_seeded():
    """Verify exactly 5 IoT devices are initialized in the fleet table."""
    devices = get_all_devices()
    assert len(devices) == 5, f"Expected 5 devices, found {len(devices)}"
    device_ids = {d["id"] for d in devices}
    expected = {"camera", "router", "sensor", "door_lock", "smart_lighting"}
    assert device_ids == expected, f"Device mismatch: {device_ids}"


def test_secure_aggregation_zero_sum_cancelation():
    """
    Core Research Test:
    Verify that pairwise zero-sum random masks cancel out upon server aggregation,
    producing an aggregated global model numerically identical to unmasked FedAvg (< 1e-6).
    """
    engine = ResearchExperimentEngine()
    n_features, n_classes = 10, 3

    client1 = FederatedMLP(n_features, n_classes, hidden_layers=[16], dropout=0.0)
    client2 = FederatedMLP(n_features, n_classes, hidden_layers=[16], dropout=0.0)
    client3 = FederatedMLP(n_features, n_classes, hidden_layers=[16], dropout=0.0)

    torch.manual_seed(10)
    for p in client1.parameters(): torch.nn.init.normal_(p)
    torch.manual_seed(20)
    for p in client2.parameters(): torch.nn.init.normal_(p)
    torch.manual_seed(30)
    for p in client3.parameters(): torch.nn.init.normal_(p)

    params_list = [get_model_params(client1), get_model_params(client2), get_model_params(client3)]
    samples_list = [1000, 1500, 2500]

    # Run pairwise masking simulation
    masked_params_list, max_diff = engine._simulate_pairwise_masking(params_list, samples_list)

    # Max difference must be within floating-point tolerance
    assert max_diff < 1e-6, f"Mask cancelation failure: max diff was {max_diff}"


def test_client_dropout_aggregation():
    """
    Verify server aggregates gracefully when a subset of clients drop out.
    """
    n_features, n_classes = 10, 3
    c1 = FederatedMLP(n_features, n_classes, hidden_layers=[16], dropout=0.0)
    c2 = FederatedMLP(n_features, n_classes, hidden_layers=[16], dropout=0.0)

    torch.manual_seed(1)
    for p in c1.parameters(): p.data.fill_(1.0)
    torch.manual_seed(2)
    for p in c2.parameters(): p.data.fill_(5.0)

    # Full aggregation
    full_agg = fedavg([get_model_params(c1), get_model_params(c2)], [100, 100])
    # Dropped c2 (only c1 participating)
    dropout_agg = fedavg([get_model_params(c1)], [100])

    for k in full_agg:
        if 'running_mean' in k or 'running_var' in k or 'num_batches' in k:
            continue
        if full_agg[k].is_floating_point():
            assert np.allclose(full_agg[k].numpy(), 3.0)
            assert np.allclose(dropout_agg[k].numpy(), 1.0)


def test_dynamic_risk_scoring_tiers():
    """Verify that risk scores map strictly to the 4 software mitigation tiers."""
    from src.realtime.detector import IDSDetector
    detector = IDSDetector()

    # Normal should result in MONITOR
    risk_norm, mit_norm = detector.calculate_risk_score("Normal", confidence=0.98, device_id="camera")
    assert mit_norm == "MONITOR"
    assert risk_norm < 30.0

    # Critical attack with high confidence on router should trigger QUARANTINE or ISOLATION
    risk_ddos, mit_ddos = detector.calculate_risk_score("DDoS", confidence=0.99, device_id="router", recent_threat_count=5)
    assert mit_ddos in ["SIMULATED QUARANTINE", "SIMULATED ISOLATION"]
    assert risk_ddos >= 60.0


def test_pdf_report_builds():
    """Verify ReportLab compiles the research report without exceptions."""
    pdf_path = PATHS.RESULTS / "test_report.pdf"
    res_path = generate_pdf_report(output_path=pdf_path)
    assert res_path.exists()
    assert res_path.stat().st_size > 1000
    pdf_path.unlink() # cleanup
