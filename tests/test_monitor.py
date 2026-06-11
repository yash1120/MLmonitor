from __future__ import annotations

from mlmonitor.config import settings
from mlmonitor.models.train import train_baseline
from mlmonitor.monitor import _classify_status, run_drift_check
from mlmonitor.simulator.production_stream import (
    append_to_production_log,
    generate_production_batch,
)


def test_classify_status_thresholds() -> None:
    assert _classify_status(psi_max=0.05, perf_drop=0.0) == "ok"
    assert _classify_status(psi_max=settings.psi_warn_threshold, perf_drop=0.0) == "warn"
    assert _classify_status(psi_max=settings.psi_alert_threshold, perf_drop=0.0) == "alert"
    assert _classify_status(psi_max=0.0, perf_drop=settings.perf_drop_alert) == "alert"
    assert _classify_status(psi_max=0.0, perf_drop=None) == "ok"


def test_run_drift_check_no_data(sandbox) -> None:
    train_baseline()
    report = run_drift_check(window_hours=24)
    assert report["status"] == "no_data"
    assert report["n_samples"] == 0


def test_run_drift_check_insufficient_samples(sandbox) -> None:
    train_baseline()
    append_to_production_log(generate_production_batch(n=20, seed=1))
    report = run_drift_check(window_hours=24 * 14)
    assert report["status"] == "insufficient_data"
    assert 0 < report["n_samples"] < settings.min_samples_for_check


def test_run_drift_check_clean_vs_drifted(sandbox) -> None:
    train_baseline()
    append_to_production_log(generate_production_batch(n=2_000, drift_mode="none", seed=2))
    clean = run_drift_check(window_hours=24 * 14)
    assert clean["status"] == "ok"

    append_to_production_log(
        generate_production_batch(n=2_000, drift_mode="covariate", intensity=1.5, seed=3)
    )
    drifted = run_drift_check(window_hours=24 * 14)
    assert drifted["status"] == "alert"
    assert drifted["psi_max"] > clean["psi_max"]
    assert drifted["drift_check_id"] > clean["drift_check_id"]
