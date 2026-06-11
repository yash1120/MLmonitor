from __future__ import annotations

from datetime import UTC, datetime

from mlmonitor.storage.db import (
    recent_agent_reports,
    recent_drift_checks,
    save_agent_report,
    save_drift_check,
)


def test_drift_check_round_trip(sandbox) -> None:
    report = {
        "status": "alert",
        "psi_max": 0.42,
        "psi_mean": 0.2,
        "perf_f1": 0.81,
        "perf_drop": 0.07,
        "window_start": datetime(2026, 6, 1, tzinfo=UTC),
        "window_end": datetime(2026, 6, 2, tzinfo=UTC),
        "mlflow_run_id": "abc123",
        "drifted_features": ["monthly_spend"],
    }
    check_id = save_drift_check(report)
    assert check_id >= 1

    rows = recent_drift_checks(limit=5)
    assert rows[0]["status"] == "alert"
    assert rows[0]["psi_max"] == 0.42


def test_agent_report_round_trip(sandbox) -> None:
    report_id = save_agent_report(
        drift_check_id=1,
        diagnosis="covariate shift in monthly_spend",
        recommendations=[{"action": "retrain", "priority": "high", "reason": "f1 drop"}],
        triggered_retraining=True,
    )
    assert report_id >= 1
    rows = recent_agent_reports(limit=5)
    assert rows[0]["triggered_retraining"] is True
    assert rows[0]["recommendations"][0]["action"] == "retrain"


def test_recent_ordering(sandbox) -> None:
    for psi in (0.1, 0.2, 0.3):
        save_drift_check({"status": "ok", "psi_max": psi, "psi_mean": psi})
    rows = recent_drift_checks(limit=2)
    assert len(rows) == 2
    assert rows[0]["id"] > rows[1]["id"]
