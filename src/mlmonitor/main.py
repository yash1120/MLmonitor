from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from mlmonitor import __version__
from mlmonitor import observability as obs
from mlmonitor.config import settings
from mlmonitor.dashboard import DASHBOARD_HTML
from mlmonitor.logging_utils import get_logger
from mlmonitor.models.train import load_bundle, train_baseline
from mlmonitor.monitor import diagnose_with_agent, run_drift_check
from mlmonitor.simulator.production_stream import (
    append_to_production_log,
    generate_production_batch,
)
from mlmonitor.storage.db import (
    recent_agent_reports,
    recent_drift_checks,
    recent_retrain_audits,
)

logger = get_logger(__name__)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Shared-secret guard on state-changing / quota-spending endpoints. No-ops when
    MONITOR_API_KEY is unset (localhost dev); enforced once a key is configured."""
    if settings.monitor_api_key and x_api_key != settings.monitor_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

app = FastAPI(
    title="Agentic MLOps Monitor",
    version=__version__,
    description=(
        "Continuously monitors a production churn-prediction model. "
        "Detects data + concept drift, runs a LangChain agent over the results, "
        "and triggers GitHub Actions retraining when thresholds are breached."
    ),
)


class PredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="List of feature dicts matching the trained model schema.",
    )


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]


class CheckRequest(BaseModel):
    window_hours: int = Field(24, ge=1, le=720)
    run_agent: bool = True


class SimulateRequest(BaseModel):
    n: int = Field(1000, ge=10, le=20_000)
    drift_mode: Literal["none", "covariate", "concept", "both"] = "none"
    intensity: float = Field(1.0, ge=0.0, le=3.0)
    seed: int | None = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "agentic-mlops-monitor",
        "version": __version__,
        "endpoints": [
            "POST /train/baseline",
            "POST /predict",
            "POST /simulate/batch",
            "POST /monitor/check",
            "POST /monitor/diagnose",
            "GET  /monitor/checks",
            "GET  /monitor/reports",
            "GET  /monitor/audits",
            "GET  /dashboard",
            "GET  /metrics",
        ],
    }


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    payload, content_type = obs.metrics_payload()
    return Response(content=payload, media_type=content_type)


@app.post("/train/baseline")
def train_baseline_route(_: None = Depends(require_api_key)) -> dict[str, Any]:
    result = train_baseline()
    return {
        "model_path": result.model_path,
        "mlflow_run_id": result.mlflow_run_id,
        "metrics": result.metrics,
        "reference_path": result.reference_path,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: None = Depends(require_api_key)) -> PredictResponse:
    try:
        bundle = load_bundle()
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail="Baseline model not trained yet. POST /train/baseline first.",
        ) from None
    pipeline, feature_names = bundle.pipeline, bundle.feature_names
    df = pd.DataFrame(req.instances)
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {sorted(missing)}")
    extra = set(df.columns) - set(feature_names)
    if extra:
        # reject rather than silently drop — a junk-key payload is a client bug or a
        # memory-amplification attempt, not something to swallow
        raise HTTPException(status_code=400, detail=f"Unknown features: {sorted(extra)}")
    df = df[feature_names]
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise HTTPException(status_code=400, detail="Instances contain NaN or infinite values.")
    probs = pipeline.predict_proba(df)[:, 1].astype(float)
    threshold = bundle.decision_threshold or 0.5
    preds = (probs >= threshold).astype(int).tolist()
    obs.PREDICTIONS_TOTAL.inc(len(preds))
    return PredictResponse(predictions=preds, probabilities=probs.tolist())


@app.post("/simulate/batch")
def simulate_batch(req: SimulateRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
    df = generate_production_batch(
        n=req.n, drift_mode=req.drift_mode, intensity=req.intensity, seed=req.seed
    )
    path = append_to_production_log(df)
    return {
        "n_rows_appended": int(len(df)),
        "drift_mode": req.drift_mode,
        "intensity": req.intensity,
        "production_log_path": str(path),
    }


@app.post("/monitor/check")
def monitor_check(req: CheckRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
    try:
        with obs.CHECK_DURATION.time():
            report = run_drift_check(window_hours=req.window_hours)
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail="Baseline model or reference data missing. POST /train/baseline first.",
        ) from None
    obs.record_drift_check(report)
    response: dict[str, Any] = {"drift_report": _serialise(report)}

    if req.run_agent and report.get("status") in {"warn", "alert"}:
        try:
            verdict = diagnose_with_agent(report)
            response["agent_verdict"] = verdict
            obs.AGENT_RUNS_TOTAL.labels(outcome="ok").inc()
            if verdict.get("actually_triggered"):
                obs.RETRAIN_TRIGGERS_TOTAL.inc()
        except Exception:
            obs.AGENT_RUNS_TOTAL.labels(outcome="error").inc()
            logger.exception("Agent diagnosis failed")
            # don't leak stack/exception text to (possibly unauthenticated) clients
            response["agent_error"] = "agent diagnosis failed"
    return response


@app.post("/monitor/diagnose")
def monitor_diagnose(
    window_hours: int = 24, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    """Force the agent to run regardless of status (useful for demos)."""
    try:
        report = run_drift_check(window_hours=window_hours)
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="Train baseline first.") from None
    verdict = diagnose_with_agent(report)
    return {"drift_report": _serialise(report), "agent_verdict": verdict}


@app.get("/monitor/checks")
def list_checks(limit: int = 20) -> list[dict[str, Any]]:
    return recent_drift_checks(limit=limit)


@app.get("/monitor/reports")
def list_reports(limit: int = 20) -> list[dict[str, Any]]:
    return recent_agent_reports(limit=limit)


@app.get("/monitor/audits")
def list_audits(limit: int = 20) -> list[dict[str, Any]]:
    """Audit trail of every retrain-dispatch attempt (allowed, blocked, dry-run)."""
    return recent_retrain_audits(limit=limit)


def _serialise(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in report.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out
