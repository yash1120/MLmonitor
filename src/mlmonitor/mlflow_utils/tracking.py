from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn

from mlmonitor.config import settings


def _ensure_experiment() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_training_run(pipeline, metrics: dict, reference_path: Path) -> str:
    _ensure_experiment()
    with mlflow.start_run(run_name="baseline-train") as run:
        mlflow.log_params(
            {
                "model_type": type(pipeline.named_steps["clf"]).__name__,
                "n_estimators": pipeline.named_steps["clf"].n_estimators,
                "max_depth": pipeline.named_steps["clf"].max_depth,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(reference_path), artifact_path="reference_data")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        try:
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name="churn-baseline",
            )
        except Exception:
            pass
        return run.info.run_id


def log_drift_check(report: dict) -> str:
    _ensure_experiment()
    with mlflow.start_run(run_name="drift-check", nested=False) as run:
        flat_metrics = {
            "psi_max": report.get("psi_max", 0.0),
            "psi_mean": report.get("psi_mean", 0.0),
            "ks_features_flagged": report.get("ks_features_flagged", 0),
            "perf_f1": report.get("perf_f1", 0.0),
            "perf_drop": report.get("perf_drop", 0.0),
        }
        mlflow.log_metrics({k: float(v) for k, v in flat_metrics.items() if v is not None})
        mlflow.set_tags(
            {
                "drift_status": report.get("status", "unknown"),
                "window_start": str(report.get("window_start", "")),
                "window_end": str(report.get("window_end", "")),
            }
        )
        return run.info.run_id


def latest_model_uri() -> str | None:
    _ensure_experiment()
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions("name='churn-baseline'")
    except Exception:
        return None
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/churn-baseline/{latest.version}"
