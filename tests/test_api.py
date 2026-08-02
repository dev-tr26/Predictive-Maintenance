import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.inference import engine


@pytest.fixture(scope="module", autouse=True)
def ensure_models_loaded():
    if not engine.loaded:
        engine.load()
    yield


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


def test_predict_endpoint_returns_valid_schema(client):
    payload = {
        "unit_id": "test-unit-1",
        "cycle": 100,
        "sensor_2": 520, "sensor_3": 610, "sensor_4": 615,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["unit_id"] == "test-unit-1"
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["status"] in ("healthy", "warning", "critical")


def test_predict_batch_endpoint(client):
    payload = [
        {"unit_id": "batch-a", "cycle": 10},
        {"unit_id": "batch-b", "cycle": 20},
    ]
    res = client.post("/api/predict/batch", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 2
    assert {b["unit_id"] for b in body} == {"batch-a", "batch-b"}


def test_predict_rejects_missing_required_fields(client):
    res = client.post("/api/predict", json={"unit_id": "no-cycle"})
    assert res.status_code == 422  # pydantic validation error


def test_fleet_summary_endpoint(client):
    res = client.get("/api/fleet/summary")
    assert res.status_code == 200
    body = res.json()
    assert "healthy" in body and "warning" in body and "critical" in body


def test_metrics_endpoint_returns_offline_eval(client):
    res = client.get("/api/metrics")
    assert res.status_code == 200
    body = res.json()
    assert "classifier_metrics" in body
    assert "roc_auc" in body["classifier_metrics"]


def test_unknown_unit_history_returns_404(client):
    res = client.get("/api/fleet/history/does-not-exist")
    assert res.status_code == 404
