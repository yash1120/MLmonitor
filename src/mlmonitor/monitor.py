"""Orchestrates a single end-to-end monitoring pass:
   load production window -> drift -> performance -> agent -> persist -> retrain hook."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from mlmonitor.agent.monitor_agent import run_diagnostic_agent
from mlmonitor.config import settings
from mlmonitor.drift.concept_drift import evaluate_performance, prediction_drift
from mlmonitor.drift.data_drift import feature_drift, summarise_data_drift
from mlmonitor.logging_utils import get_logger
from mlmonitor.mlflow_utils.tracking import log_drift_check
from mlmonitor.models.train import TARGET_NAME, load_model
from mlmonitor.simulator.production_stream import load_production_window
from mlmonitor.storage.db import save_agent_report, save_drift_check

logger = get_logger(__name__)


def _classify_status(psi_max: float, perf_drop: float | None) -> str:
    if psi_max >= settings.psi_alert_threshold or (perf_drop or 0) >= settings.perf_drop_alert:
        return "alert"
    if psi_max >= settings.psi_warn_threshold:
        return "warn"
    return "ok"


def run_drift_check(window_hours: int = 24) -> dict[str, Any]:
    pipeline, feature_names = load_model()
    reference = pd.read_parquet(settings.reference_data_path)
    production = load_production_window(window_hours=window_hours)

    if production.empty:
        logger.warning("Drift check ran with no production data in window (%sh)", window_hours)
        return _empty_report("no_data", n_samples=0)

    if len(production) < settings.min_samples_for_check:
        logger.warning(
            "Only %d production samples in window (< %d required) — drift stats would be unreliable",
            len(production),
            settings.min_samples_for_check,
        )
        return _empty_report("insufficient_data", n_samples=int(len(production)))

    drift_results = feature_drift(
        reference=reference,
        production=production,
        features=feature_names,
        psi_alert=settings.psi_alert_threshold,
    )
    drift_summary = summarise_data_drift(drift_results)

    perf = evaluate_performance(
        pipeline=pipeline,
        production=production,
        feature_names=feature_names,
        target_name=TARGET_NAME,
        baseline_f1=_baseline_f1_or_default(),
        drop_threshold=settings.perf_drop_alert,
    )

    ref_probs = pipeline.predict_proba(reference[feature_names])[:, 1]
    prod_probs = pipeline.predict_proba(production[feature_names])[:, 1]
    pred_drift_psi = prediction_drift(ref_probs, prod_probs)

    window_start = pd.to_datetime(production["event_ts"]).min()
    window_end = pd.to_datetime(production["event_ts"]).max()

    report: dict[str, Any] = {
        **drift_summary,
        "perf_f1": perf.f1,
        "perf_roc_auc": perf.roc_auc,
        "perf_drop": perf.f1_drop,
        "baseline_f1": perf.baseline_f1,
        "concept_drift": perf.concept_drift,
        "prediction_drift_psi": pred_drift_psi,
        "n_samples": int(len(production)),
        "window_start": window_start.to_pydatetime() if pd.notna(window_start) else None,
        "window_end": window_end.to_pydatetime() if pd.notna(window_end) else None,
        "checked_at": datetime.now(UTC),
    }
    report["status"] = _classify_status(report["psi_max"], report["perf_drop"])
    logger.info(
        "Drift check: status=%s psi_max=%.4f perf_drop=%.4f n=%d drifted=%s",
        report["status"],
        report["psi_max"],
        report["perf_drop"] or 0.0,
        report["n_samples"],
        report["drifted_features"],
    )

    mlflow_run_id = log_drift_check(report)
    report["mlflow_run_id"] = mlflow_run_id
    drift_check_id = save_drift_check(report)
    report["drift_check_id"] = drift_check_id

    return report


def diagnose_with_agent(report: dict[str, Any]) -> dict[str, Any]:
    verdict = run_diagnostic_agent(report)
    save_agent_report(
        drift_check_id=int(report.get("drift_check_id", 0)),
        diagnosis=verdict.get("diagnosis", ""),
        recommendations=verdict.get("recommendations", []),
        triggered_retraining=bool(verdict.get("actually_triggered", False)),
    )
    return verdict


def _empty_report(status: str, n_samples: int) -> dict[str, Any]:
    return {
        "status": status,
        "psi_max": 0.0,
        "psi_mean": 0.0,
        "ks_features_flagged": 0,
        "drifted_features": [],
        "by_feature": [],
        "n_samples": n_samples,
        "window_start": None,
        "window_end": None,
    }


def _baseline_f1_or_default() -> float:
    """Read baseline F1 from latest MLflow training run; fall back to 0.85 if unavailable."""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(settings.mlflow_experiment_name)
        if exp is None:
            return 0.85
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.mlflow.runName = 'baseline-train'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs and "f1" in runs[0].data.metrics:
            return float(runs[0].data.metrics["f1"])
    except Exception:
        pass
    return 0.85
