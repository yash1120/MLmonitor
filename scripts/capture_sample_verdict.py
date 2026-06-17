"""Capture a REAL agent verdict to examples/sample_verdict.md so the agent's quality is
visible in the repo without a reviewer needing to set up a Groq key and run the stack.

Runs against a throwaway data dir (so it doesn't touch your working data/). Falls back to
a clearly-labelled illustrative verdict if no GROQ_API_KEY is configured.

Usage:  python scripts/capture_sample_verdict.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mlmonitor.config import settings

# Redirect all artifacts to a temp dir BEFORE anything touches them.
_tmp = Path(tempfile.mkdtemp())
settings.model_artifact_path = str(_tmp / "model.joblib")
settings.reference_data_path = str(_tmp / "reference.parquet")
settings.production_data_path = str(_tmp / "production.parquet")
settings.metrics_db_url = f"sqlite:///{(_tmp / 'metrics.db').as_posix()}"
settings.mlflow_tracking_uri = f"sqlite:///{(_tmp / 'mlflow.db').as_posix()}"

from mlmonitor.models.train import train_baseline  # noqa: E402
from mlmonitor.monitor import diagnose_with_agent, run_drift_check  # noqa: E402
from mlmonitor.simulator.production_stream import (  # noqa: E402
    append_to_production_log,
    generate_production_batch,
)

OUT = Path("examples/sample_verdict.md")


def _illustrative() -> dict:
    return {
        "diagnosis": (
            "Covariate shift detected: monthly_spend dropped ~35% and credit_utilisation "
            "rose, driving psi_max to 0.38 across 4 features. Production F1 fell 0.11 below "
            "the 0.69 baseline, and within-window slices show the shift is sustained, not a "
            "transient blip — consistent with a genuine population change rather than noise."
        ),
        "recommendations": [
            {"action": "Retrain on the last 30 days of labelled traffic", "priority": "high",
             "reason": "both data drift (PSI 0.38) and a 0.11 F1 drop are confirmed and sustained"},
            {"action": "Investigate the monthly_spend pipeline upstream", "priority": "medium",
             "reason": "a 35% mean shift suggests an instrumentation or billing-source change"},
        ],
        "trigger_retraining": True,
        "actually_triggered": False,
        "note": "ILLUSTRATIVE sample (no GROQ_API_KEY at capture time).",
    }


def main() -> None:
    train_baseline()
    append_to_production_log(
        generate_production_batch(n=2500, drift_mode="both", intensity=1.2, seed=42)
    )
    report = run_drift_check(window_hours=168)

    real = bool(settings.groq_api_key)
    verdict = diagnose_with_agent(report) if real else _illustrative()

    OUT.parent.mkdir(exist_ok=True)
    drift = {
        "status": report["status"],
        "psi_max": round(report["psi_max"], 4),
        "perf_f1": round(report.get("perf_f1") or 0, 4),
        "perf_drop": round(report.get("perf_drop") or 0, 4),
        "prediction_drift_psi": round(report.get("prediction_drift_psi") or 0, 4),
        "drift_sustained": report.get("drift_sustained"),
        "drifted_features": report.get("drifted_features"),
        "label_coverage": report.get("label_coverage"),
    }
    lines = [
        "# Sample agent verdict",
        "",
        f"{'A real' if real else 'An illustrative'} run of the LangGraph diagnostic agent on a "
        "`both`-drift batch. Committed so the agent's output is visible without an API key.",
        "",
        "## Drift report (input to the agent)",
        "```json",
        json.dumps(drift, indent=2, default=str),
        "```",
        "",
        "## Agent verdict (output)",
        "```json",
        json.dumps(
            {k: verdict.get(k) for k in
             ("diagnosis", "recommendations", "trigger_retraining", "actually_triggered")},
            indent=2, default=str,
        ),
        "```",
        "",
        "> `trigger_retraining` is the agent's *recommendation*; `actually_triggered` reflects "
        "the **deterministic gate**, which independently re-checks both thresholds, the "
        "kill-switch, and the rate limit before any real dispatch.",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({'real Groq verdict' if real else 'illustrative'})")


if __name__ == "__main__":
    main()
