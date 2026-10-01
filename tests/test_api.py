"""
tests/test_api.py
==================
Tests for the Flask dashboard API endpoints.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import json


@pytest.fixture
def client():
    from src.dashboard.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert "devices_online" in data
    assert data["devices_online"] == 5
    assert "timestamp" in data


def test_devices_endpoint(client):
    r = client.get("/api/devices")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert isinstance(data, list)
    assert len(data) == 5
    ids = {d["id"] for d in data}
    assert "smart_lighting" in ids
    assert "camera" in ids


def test_dataset_status_endpoint(client):
    r = client.get("/api/dataset/status")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert "dataset_name" in data
    assert "target_classes" in data
    assert len(data["target_classes"]) == 6


def test_alerts_endpoint(client):
    r = client.get("/api/alerts")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert isinstance(data, list)


def test_model_benchmark_endpoint(client):
    r = client.get("/api/models/benchmark")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert isinstance(data, list)


def test_federated_rounds_endpoint(client):
    r = client.get("/api/federated/rounds")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert isinstance(data, list)


def test_xai_global_endpoint(client):
    r = client.get("/api/xai/global")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert "global_feature_importances" in data
    assert "xai_fidelity_pct" in data


def test_predict_endpoint_missing_features(client):
    r = client.post("/api/predict",
                    data=json.dumps({"device": "Camera"}),
                    content_type="application/json")
    assert r.status_code in [400, 503]


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Federated IDS" in r.data


def test_login_page_renders(client):
    r = client.get("/login")
    assert r.status_code == 200
    assert b"Sign In to SOC Dashboard" in r.data


def test_login_valid_credentials(client):
    r = client.post("/login",
                    data=json.dumps({"username": "admin", "password": "admin123"}),
                    content_type="application/json")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["status"] == "SUCCESS"
    assert data["username"] == "admin"
    assert data["role"] == "SOC_ADMIN"


def test_login_invalid_credentials(client):
    r = client.post("/login",
                    data=json.dumps({"username": "admin", "password": "wrongpassword"}),
                    content_type="application/json")
    assert r.status_code == 401
    data = json.loads(r.data)
    assert data["status"] == "ERROR"


def test_auth_me_endpoint(client):
    # After logging in
    client.post("/login",
                data=json.dumps({"username": "admin", "password": "admin123"}),
                content_type="application/json")
    r = client.get("/api/auth/me")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["authenticated"] is True
    assert data["username"] == "admin"


def test_logout_endpoint(client):
    r = client.get("/logout")
    assert r.status_code in [200, 302]


def test_feature_selection_endpoint(client):
    r = client.get("/api/dataset/feature-selection")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert "original_feature_count" in data
    assert "selected_feature_count" in data
    assert data["original_feature_count"] == 78
    assert data["selected_feature_count"] == 54
    assert len(data["dropped_high_correlation"]) == 24
