from __future__ import annotations

import httpx

from mlmonitor import notifications
from mlmonitor.config import settings

REPORT = {
    "status": "alert",
    "psi_max": 0.4,
    "perf_drop": 0.1,
    "prediction_drift_psi": 0.2,
    "drifted_features": ["monthly_spend"],
    "n_samples": 2000,
    "n_labeled": 1800,
}


def test_notify_noops_without_webhook(monkeypatch):
    monkeypatch.setattr(settings, "alert_webhook_url", "")
    assert notifications.notify_drift(REPORT) is False


def test_notify_posts_when_configured(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent["url"] = url
        sent["json"] = json
        return httpx.Response(200)

    monkeypatch.setattr(settings, "alert_webhook_url", "https://hooks.example/abc")
    monkeypatch.setattr(notifications.httpx, "post", fake_post)
    assert notifications.notify_drift(REPORT) is True
    assert sent["url"] == "https://hooks.example/abc"
    assert "ALERT" in sent["json"]["text"]
    assert "monthly_spend" in sent["json"]["text"]


def test_notify_survives_webhook_error(monkeypatch):
    def boom(url, json, timeout):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(settings, "alert_webhook_url", "https://hooks.example/abc")
    monkeypatch.setattr(notifications.httpx, "post", boom)
    assert notifications.notify_drift(REPORT) is False
