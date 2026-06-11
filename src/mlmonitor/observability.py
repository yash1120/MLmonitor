"""Prometheus metrics for the monitor service (scrape at GET /metrics)."""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

PREDICTIONS_TOTAL = Counter(
    "mlmonitor_predictions_total", "Rows scored via POST /predict"
)
DRIFT_CHECKS_TOTAL = Counter(
    "mlmonitor_drift_checks_total", "Drift checks run, by resulting status", ["status"]
)
AGENT_RUNS_TOTAL = Counter(
    "mlmonitor_agent_runs_total", "Diagnostic agent invocations, by outcome", ["outcome"]
)
RETRAIN_TRIGGERS_TOTAL = Counter(
    "mlmonitor_retrain_triggers_total", "Retraining workflows dispatched by the agent"
)
LAST_PSI_MAX = Gauge("mlmonitor_last_psi_max", "psi_max from the most recent drift check")
LAST_F1 = Gauge("mlmonitor_last_f1", "Production F1 from the most recent drift check")
LAST_F1_DROP = Gauge("mlmonitor_last_f1_drop", "F1 drop vs baseline from the most recent check")
CHECK_DURATION = Histogram(
    "mlmonitor_drift_check_seconds", "Wall-clock duration of a drift check"
)


def record_drift_check(report: dict) -> None:
    DRIFT_CHECKS_TOTAL.labels(status=str(report.get("status", "unknown"))).inc()
    LAST_PSI_MAX.set(float(report.get("psi_max", 0.0) or 0.0))
    if report.get("perf_f1") is not None:
        LAST_F1.set(float(report["perf_f1"]))
    if report.get("perf_drop") is not None:
        LAST_F1_DROP.set(float(report["perf_drop"]))


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
