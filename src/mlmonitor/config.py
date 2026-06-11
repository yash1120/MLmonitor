from pathlib import Path

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

    psi_warn_threshold: float = 0.1
    psi_alert_threshold: float = 0.25
    perf_drop_alert: float = 0.05
    min_samples_for_check: int = 100
    log_level: str = "INFO"

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


settings = Settings()
