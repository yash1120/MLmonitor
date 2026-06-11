from __future__ import annotations

from fastapi.testclient import TestClient

from mlmonitor.main import app

client = TestClient(app)


def test_healthz() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_before_training_returns_409(sandbox) -> None:
    resp = client.post("/predict", json={"instances": [{"monthly_spend": 100.0}]})
    assert resp.status_code == 409


def test_full_flow_train_predict_simulate_check(sandbox) -> None:
    # train
    resp = client.post("/train/baseline")
    assert resp.status_code == 200
    # imbalanced classes (22% positive) + 2% label noise → modest absolute F1
    assert resp.json()["metrics"]["f1"] > 0.5

    # predict happy path
    instance = {
        "account_age_days": 120.0,
        "monthly_spend": 250.0,
        "credit_utilisation": 0.4,
        "logins_last_30d": 12.0,
        "support_tickets": 1.0,
        "tenure_months": 24.0,
        "n_products": 2.0,
        "balance_to_limit": 0.3,
    }
    resp = client.post("/predict", json={"instances": [instance]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictions"][0] in (0, 1)
    assert 0.0 <= body["probabilities"][0] <= 1.0

    # predict rejects missing features and non-finite values
    resp = client.post("/predict", json={"instances": [{"monthly_spend": 1.0}]})
    assert resp.status_code == 400
    bad = dict(instance, monthly_spend="Infinity")
    raw = '{"instances": [' + str(bad).replace("'", '"') + "]}"
    resp = client.post("/predict", content=raw.replace('"Infinity"', "Infinity"),
                       headers={"content-type": "application/json"})
    assert resp.status_code in (400, 422)

    # simulate drifted production traffic
    resp = client.post(
        "/simulate/batch", json={"n": 1500, "drift_mode": "both", "intensity": 1.0, "seed": 5}
    )
    assert resp.status_code == 200
    assert resp.json()["n_rows_appended"] == 1500

    # drift check (agent off — no API key needed in CI)
    resp = client.post("/monitor/check", json={"window_hours": 720, "run_agent": False})
    assert resp.status_code == 200
    report = resp.json()["drift_report"]
    assert report["status"] in {"warn", "alert"}
    assert report["psi_max"] > 0.1

    # history endpoints reflect the check
    resp = client.get("/monitor/checks")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_monitor_check_without_model_returns_409(sandbox) -> None:
    resp = client.post("/monitor/check", json={"window_hours": 24, "run_agent": False})
    assert resp.status_code == 409
