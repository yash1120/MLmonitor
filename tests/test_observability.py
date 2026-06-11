from __future__ import annotations

from fastapi.testclient import TestClient

from mlmonitor.main import app

client = TestClient(app)


def test_metrics_endpoint_exposes_prometheus_format() -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "mlmonitor_drift_checks_total" in resp.text or "mlmonitor" in resp.text


def test_dashboard_serves_html() -> None:
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Agentic MLOps Monitor" in resp.text


def test_drift_check_updates_gauges(sandbox) -> None:
    client.post("/train/baseline")
    client.post("/simulate/batch", json={"n": 500, "drift_mode": "none", "seed": 1})
    client.post("/monitor/check", json={"window_hours": 720, "run_agent": False})
    resp = client.get("/metrics")
    assert 'mlmonitor_drift_checks_total{status="ok"}' in resp.text
    assert "mlmonitor_last_psi_max" in resp.text
