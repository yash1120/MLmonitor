from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from mlmonitor.config import settings


class Base(DeclarativeBase):
    pass


class DriftCheck(Base):
    __tablename__ = "drift_checks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    status = Column(String(16), nullable=False, index=True)
    psi_max = Column(Float, nullable=False)
    psi_mean = Column(Float, nullable=False)
    perf_f1 = Column(Float, nullable=True)
    perf_drop = Column(Float, nullable=True)
    window_start = Column(DateTime, nullable=True)
    window_end = Column(DateTime, nullable=True)
    mlflow_run_id = Column(String(64), nullable=True)
    report = Column(JSON, nullable=False)


class AgentReport(Base):
    __tablename__ = "agent_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    drift_check_id = Column(Integer, nullable=False, index=True)
    diagnosis = Column(String, nullable=False)
    recommendations = Column(JSON, nullable=False)
    triggered_retraining = Column(Integer, default=0, nullable=False)


_engine = create_engine(settings.metrics_db_url, future=True)
Base.metadata.create_all(_engine)


def save_drift_check(report: dict[str, Any]) -> int:
    with Session(_engine) as session:
        row = DriftCheck(
            status=report.get("status", "ok"),
            psi_max=float(report.get("psi_max", 0.0)),
            psi_mean=float(report.get("psi_mean", 0.0)),
            perf_f1=report.get("perf_f1"),
            perf_drop=report.get("perf_drop"),
            window_start=report.get("window_start"),
            window_end=report.get("window_end"),
            mlflow_run_id=report.get("mlflow_run_id"),
            report=json.loads(json.dumps(report, default=str)),
        )
        session.add(row)
        session.commit()
        return int(row.id)


def save_agent_report(
    drift_check_id: int,
    diagnosis: str,
    recommendations: list[dict],
    triggered_retraining: bool,
) -> int:
    with Session(_engine) as session:
        row = AgentReport(
            drift_check_id=drift_check_id,
            diagnosis=diagnosis,
            recommendations=recommendations,
            triggered_retraining=1 if triggered_retraining else 0,
        )
        session.add(row)
        session.commit()
        return int(row.id)


def recent_drift_checks(limit: int = 20) -> list[dict]:
    with Session(_engine) as session:
        rows = session.execute(
            select(DriftCheck).order_by(DriftCheck.ts.desc()).limit(limit)
        ).scalars().all()
        return [
            {
                "id": r.id,
                "ts": r.ts.isoformat(),
                "status": r.status,
                "psi_max": r.psi_max,
                "psi_mean": r.psi_mean,
                "perf_f1": r.perf_f1,
                "perf_drop": r.perf_drop,
                "mlflow_run_id": r.mlflow_run_id,
            }
            for r in rows
        ]


def recent_agent_reports(limit: int = 20) -> list[dict]:
    with Session(_engine) as session:
        rows = session.execute(
            select(AgentReport).order_by(AgentReport.ts.desc()).limit(limit)
        ).scalars().all()
        return [
            {
                "id": r.id,
                "ts": r.ts.isoformat(),
                "drift_check_id": r.drift_check_id,
                "diagnosis": r.diagnosis,
                "recommendations": r.recommendations,
                "triggered_retraining": bool(r.triggered_retraining),
            }
            for r in rows
        ]
