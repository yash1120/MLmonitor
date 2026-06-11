from __future__ import annotations

import pytest

from mlmonitor.config import settings


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point every artifact/DB path at a temp dir so tests never touch data/."""
    monkeypatch.setattr(settings, "model_artifact_path", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(settings, "reference_data_path", str(tmp_path / "reference.parquet"))
    monkeypatch.setattr(settings, "production_data_path", str(tmp_path / "production.parquet"))
    monkeypatch.setattr(
        settings, "metrics_db_url", f"sqlite:///{(tmp_path / 'metrics.db').as_posix()}"
    )
    monkeypatch.setattr(
        settings, "mlflow_tracking_uri", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    )
    return tmp_path
