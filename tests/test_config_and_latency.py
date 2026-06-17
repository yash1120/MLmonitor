from __future__ import annotations

import pytest

from mlmonitor.config import Settings, settings
from mlmonitor.models.train import train_baseline
from mlmonitor.monitor import run_drift_check
from mlmonitor.simulator.production_stream import (
    append_to_production_log,
    generate_production_batch,
)


def test_config_rejects_inverted_psi_thresholds():
    with pytest.raises(ValueError):
        Settings(psi_warn_threshold=0.3, psi_alert_threshold=0.25)


def test_config_rejects_bad_perf_drop():
    with pytest.raises(ValueError):
        Settings(perf_drop_alert=1.5)


def test_config_rejects_negative_label_delay():
    with pytest.raises(ValueError):
        Settings(label_delay_days=-1)


def test_baseline_f1_comes_from_bundle_not_magic_default(sandbox):
    """Regression: the old 0.85 default flagged spurious concept drift on clean data."""
    result = train_baseline()
    assert result.metrics["f1"] > 0.5
    assert result.metrics["roc_auc"] > 0.7
    append_to_production_log(generate_production_batch(n=1500, drift_mode="none", seed=4))
    report = run_drift_check(window_hours=24 * 14)
    # clean traffic must NOT be flagged as concept drift (baseline is the served model's real F1)
    assert report["status"] == "ok"
    assert report["concept_drift"] is False
    assert 0.0 < report["baseline_f1"] < 1.0
    assert report["label_coverage"] == 1.0  # label_delay_days defaults to 0


def test_label_latency_masks_unlabeled_rows(sandbox, monkeypatch):
    train_baseline()
    append_to_production_log(generate_production_batch(n=1500, drift_mode="none", seed=5))
    # withhold labels for 30 days → none of the last-7-days batch is labelled yet
    monkeypatch.setattr(settings, "label_delay_days", 30)
    report = run_drift_check(window_hours=24 * 14)
    assert report["label_coverage"] < 1.0
    assert report["n_labeled"] < report["n_samples"]
    # with no usable labels, concept drift is not assessed (no phantom drop)
    assert report["concept_drift"] is False


def test_prediction_drift_psi_present_in_report(sandbox):
    train_baseline()
    append_to_production_log(generate_production_batch(n=1500, drift_mode="none", seed=6))
    report = run_drift_check(window_hours=24 * 14)
    assert "prediction_drift_psi" in report
    assert isinstance(report["prediction_drift_psi"], float)
    assert "reference_hash" in report and report["reference_hash"]
