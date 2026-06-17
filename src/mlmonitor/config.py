from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    mlflow_tracking_uri: str = f"sqlite:///{(DATA_DIR / 'mlflow.db').as_posix()}"
    mlflow_experiment_name: str = "ml-monitor"

    metrics_db_url: str = f"sqlite:///{(DATA_DIR / 'metrics.db').as_posix()}"

    # Data-drift (PSI) thresholds
    psi_warn_threshold: float = 0.1
    psi_alert_threshold: float = 0.25
    # Concept-drift (F1 drop vs baseline) threshold
    perf_drop_alert: float = 0.05
    # Label-free prediction-distribution PSI thresholds (online concept-drift proxy)
    pred_drift_warn_threshold: float = 0.1
    pred_drift_alert_threshold: float = 0.25
    min_samples_for_check: int = 100
    log_level: str = "INFO"

    # Retraining safety: deterministic gate is always enforced; these add a global
    # kill-switch, a dry-run mode, and a minimum interval between dispatches.
    retrain_enabled: bool = True
    retrain_dry_run: bool = False
    retrain_min_interval_minutes: int = 60

    # Optional shared-secret auth on state-changing endpoints (empty = open, localhost-only)
    monitor_api_key: str = ""

    # Optional free outbound alerting (Slack/Discord incoming webhook); empty = no-op
    alert_webhook_url: str = ""

    # Simulated label latency in days (0 = labels arrive instantly; >0 withholds
    # the churn label until event_ts + label_delay_days, modelling real ground-truth lag)
    label_delay_days: int = 0

    github_owner: str = ""
    github_repo: str = ""
    github_workflow: str = "retrain.yml"
    github_ref: str = "main"
    github_token: str = ""

    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    azure_workspace_name: str = ""

    model_artifact_path: str = str(DATA_DIR / "baseline_model.joblib")
    reference_data_path: str = str(DATA_DIR / "reference.parquet")
    production_data_path: str = str(DATA_DIR / "production.parquet")

    @model_validator(mode="after")
    def _check_threshold_invariants(self) -> "Settings":
        if not 0 < self.psi_warn_threshold < self.psi_alert_threshold:
            raise ValueError(
                f"Require 0 < psi_warn_threshold ({self.psi_warn_threshold}) "
                f"< psi_alert_threshold ({self.psi_alert_threshold})"
            )
        if not 0 < self.pred_drift_warn_threshold < self.pred_drift_alert_threshold:
            raise ValueError(
                "Require 0 < pred_drift_warn_threshold < pred_drift_alert_threshold"
            )
        if not 0 < self.perf_drop_alert < 1:
            raise ValueError(f"Require 0 < perf_drop_alert < 1 (got {self.perf_drop_alert})")
        if self.min_samples_for_check < 1:
            raise ValueError("min_samples_for_check must be >= 1")
        if self.label_delay_days < 0:
            raise ValueError("label_delay_days must be >= 0")
        return self


settings = Settings()
