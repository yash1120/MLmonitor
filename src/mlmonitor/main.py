from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlmonitor import __version__
from mlmonitor.config import settings
from mlmonitor.models.train import FEATURE_NAMES, train_baseline, load_model
from mlmonitor.monitor import diagnose_with_agent, run_drift_check
from mlmonitor.simulator.production_stream import (
    append_to_production_log,
    generate_production_batch,
)
from mlmonitor.storage.db import recent_agent_reports, recent_drift_checks


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
        ..., description="List of feature dicts matching the trained model schema."
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
        ],
    }


@app.post("/train/baseline")
def train_baseline_route() -> dict[str, Any]:
    result = train_baseline()
    return {
        "model_path": result.model_path,
        "mlflow_run_id": result.mlflow_run_id,
        "metrics": result.metrics,
        "reference_path": result.reference_path,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        pipeline, feature_names = load_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail="Baseline model not trained yet. POST /train/baseline first.",
        )
    df = pd.DataFrame(req.instances)
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {sorted(missing)}")
    df = df[feature_names]
    preds = pipeline.predict(df).astype(int).tolist()
    probs = pipeline.predict_proba(df)[:, 1].astype(float).tolist()
    return PredictResponse(predictions=preds, probabilities=probs)


@app.post("/simulate/batch")
def simulate_batch(req: SimulateRequest) -> dict[str, Any]:
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
def monitor_check(req: CheckRequest) -> dict[str, Any]:
    try:
        report = run_drift_check(window_hours=req.window_hours)
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail="Baseline model or reference data missing. POST /train/baseline first.",
        )
    response: dict[str, Any] = {"drift_report": _serialise(report)}

    if req.run_agent and report.get("status") in {"warn", "alert"}:
        try:
            verdict = diagnose_with_agent(report)
            response["agent_verdict"] = verdict
        except Exception as exc:
            response["agent_error"] = str(exc)
    return response


@app.post("/monitor/diagnose")
def monitor_diagnose(window_hours: int = 24) -> dict[str, Any]:
    """Force the agent to run regardless of status (useful for demos)."""
    try:
        report = run_drift_check(window_hours=window_hours)
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="Train baseline first.")
    verdict = diagnose_with_agent(report)
    return {"drift_report": _serialise(report), "agent_verdict": verdict}


@app.get("/monitor/checks")
def list_checks(limit: int = 20) -> list[dict[str, Any]]:
    return recent_drift_checks(limit=limit)


@app.get("/monitor/reports")
def list_reports(limit: int = 20) -> list[dict[str, Any]]:
    return recent_agent_reports(limit=limit)


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
