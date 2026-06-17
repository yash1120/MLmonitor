"""Orchestrates a single end-to-end monitoring pass:
   load production window -> drift -> performance -> agent -> persist -> retrain hook."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from mlmonitor.agent.monitor_agent import run_diagnostic_agent
from mlmonitor.config import settings
from mlmonitor.drift.attribution import build_attribution
from mlmonitor.drift.concept_drift import evaluate_performance, prediction_drift
from mlmonitor.drift.data_drift import feature_drift, summarise_data_drift
from mlmonitor.logging_utils import get_logger
from mlmonitor.mlflow_utils.tracking import log_drift_check
from mlmonitor.models.train import TARGET_NAME, load_bundle
from mlmonitor.notifications import notify_drift
from mlmonitor.simulator.production_stream import load_production_window
from mlmonitor.storage.db import save_agent_report, save_drift_check

logger = get_logger(__name__)


def _classify_status(
    psi_max: float, perf_drop: float | None, pred_drift_psi: float = 0.0
) -> str:
    """Tier the check. Alert on data drift, confirmed performance drop, OR a strong
    label-free prediction-distribution shift — so concept drift is detectable online
    even before ground-truth labels arrive."""
    pred = pred_drift_psi or 0.0
    if (
        psi_max >= settings.psi_alert_threshold
        or (perf_drop or 0) >= settings.perf_drop_alert
        or pred >= settings.pred_drift_alert_threshold
    ):
        return "alert"
    if psi_max >= settings.psi_warn_threshold or pred >= settings.pred_drift_warn_threshold:
        return "warn"
    return "ok"


def _labeled_slice(production: pd.DataFrame, window_end: pd.Timestamp) -> pd.DataFrame:
    """The rows whose ground-truth label would actually be known by `window_end`,
    modelling label latency: a row's label lands at event_ts + label_delay_days."""
    if settings.label_delay_days <= 0:
        return production
    cutoff = window_end - pd.Timedelta(days=settings.label_delay_days)
    return production[pd.to_datetime(production["event_ts"]) <= cutoff]


def run_drift_check(window_hours: int = 24) -> dict[str, Any]:
    bundle = load_bundle()
    pipeline, feature_names = bundle.pipeline, bundle.feature_names
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

    window_start = pd.to_datetime(production["event_ts"]).min()
    window_end = pd.to_datetime(production["event_ts"]).max()

    # Concept drift needs labels, which lag in reality — score only the slice whose
    # labels would be available by now.
    labeled = _labeled_slice(production, window_end)
    label_coverage = float(len(labeled) / len(production)) if len(production) else 0.0
    perf = evaluate_performance(
        pipeline=pipeline,
        production=labeled,
        feature_names=feature_names,
        target_name=TARGET_NAME,
        baseline_f1=bundle.baseline_f1,
        drop_threshold=settings.perf_drop_alert,
        threshold=bundle.decision_threshold,
    )

    # Label-free online proxy: shift in the predicted-probability distribution.
    # Reference probs are cached at train time (invariant between retrains).
    ref_probs = (
        bundle.ref_probs
        if getattr(bundle, "ref_probs", np.empty(0)).size
        else pipeline.predict_proba(reference[feature_names])[:, 1]
    )
    prod_probs = pipeline.predict_proba(production[feature_names])[:, 1]
    pred_drift_psi = prediction_drift(ref_probs, prod_probs)

    # Localise drift in time (velocity: transient blip vs sustained ramp) and by cohort.
    attribution = build_attribution(
        reference, production, feature_names, psi_alert=settings.psi_alert_threshold
    )

    report: dict[str, Any] = {
        **drift_summary,
        "attribution": attribution,
        "drift_sustained": attribution.get("sustained", False),
        "perf_f1": perf.f1,
        "perf_roc_auc": perf.roc_auc,
        "perf_drop": perf.f1_drop,
        "baseline_f1": perf.baseline_f1,
        "decision_threshold": bundle.decision_threshold,
        "concept_drift": perf.concept_drift,
        "prediction_drift_psi": pred_drift_psi,
        "n_samples": int(len(production)),
        "n_labeled": int(len(labeled)),
        "label_coverage": label_coverage,
        "reference_hash": bundle.reference_hash,
        "n_ref_rows": bundle.n_ref_rows,
        "window_start": window_start.to_pydatetime() if pd.notna(window_start) else None,
        "window_end": window_end.to_pydatetime() if pd.notna(window_end) else None,
        "checked_at": datetime.now(UTC),
    }
    report["status"] = _classify_status(
        report["psi_max"], report["perf_drop"], report["prediction_drift_psi"]
    )
    logger.info(
        "Drift check: status=%s psi_max=%.4f perf_drop=%.4f pred_psi=%.4f n=%d (labeled=%d) drifted=%s",
        report["status"],
        report["psi_max"],
        report["perf_drop"] or 0.0,
        report["prediction_drift_psi"],
        report["n_samples"],
        report["n_labeled"],
        report["drifted_features"],
    )

    mlflow_run_id = log_drift_check(report)
    report["mlflow_run_id"] = mlflow_run_id
    drift_check_id = save_drift_check(report)
    report["drift_check_id"] = drift_check_id

    if report["status"] in {"warn", "alert"}:
        notify_drift(report)

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
