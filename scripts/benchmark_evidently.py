"""Benchmark our hand-rolled PSI/KS detectors against Evidently — the drift library AU
MLOps interviewers name first. Runs the SAME simulated batches through both and writes
eval/BENCHMARK.md. Evidently is an optional extra (`pip install -e .[benchmark]`); if it
isn't installed, this prints how to get it and exits cleanly.

Usage:  python scripts/benchmark_evidently.py
"""
from __future__ import annotations

from pathlib import Path

from mlmonitor.config import settings
from mlmonitor.drift.data_drift import feature_drift, summarise_data_drift
from mlmonitor.models.train import FEATURE_NAMES, generate_reference_dataset
from mlmonitor.simulator.production_stream import generate_production_batch

SCENARIOS = [("none", 0.0), ("covariate", 0.5), ("covariate", 1.0), ("concept", 1.0), ("both", 1.0)]


def _ours(reference, batch) -> dict:
    summary = summarise_data_drift(
        feature_drift(reference, batch, FEATURE_NAMES, psi_alert=settings.psi_alert_threshold)
    )
    return {
        "psi_max": round(summary["psi_max"], 3),
        "n_drifted": len(summary["drifted_features"]),
        "alert": summary["psi_max"] >= settings.psi_alert_threshold,
    }


def _evidently(reference, batch):
    """Return (n_drifted_columns, dataset_drift_bool) via Evidently, or None if unavailable."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except Exception:
        return None
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[FEATURE_NAMES], current_data=batch[FEATURE_NAMES])
    result = report.as_dict()
    drift = result["metrics"][0]["result"]
    return int(drift.get("number_of_drifted_columns", 0)), bool(drift.get("dataset_drift", False))


def main() -> None:
    reference = generate_reference_dataset()
    out_dir = Path("eval")
    out_dir.mkdir(exist_ok=True)

    rows = []
    evidently_available = None
    for mode, intensity in SCENARIOS:
        batch = generate_production_batch(n=2000, drift_mode=mode, intensity=intensity, seed=11)
        ours = _ours(reference, batch)
        eviv = _evidently(reference, batch)
        evidently_available = eviv is not None
        rows.append((f"{mode} @ {intensity}", ours, eviv))

    lines = [
        "# Benchmark vs Evidently",
        "",
        "Same simulated batches, two detectors. We hand-rolled PSI/KS to *understand* the math;",
        "this is the deliberate-scoping comparison against the standard library.",
        "",
        "| Scenario | Ours: PSI max | Ours: drift? | Evidently: #cols drifted | Evidently: drift? |",
        "|----------|:---:|:---:|:---:|:---:|",
    ]
    for name, ours, eviv in rows:
        if eviv is None:
            ev_cols, ev_drift = "n/a", "n/a"
        else:
            ev_cols, ev_drift = eviv[0], ("yes" if eviv[1] else "no")
        lines.append(
            f"| {name} | {ours['psi_max']} | {'yes' if ours['alert'] else 'no'} | "
            f"{ev_cols} | {ev_drift} |"
        )

    lines += [
        "",
        "**Where each wins / the scoping choice:**",
        "- Evidently gives a polished per-column report and a large metric catalogue out of the box.",
        "- Ours adds what neither library does here: an **LLM agent that diagnoses** the drift in "
        "plain English and a **gated, audited retrain trigger** — and a label-free "
        "prediction-PSI + label-latency model (NannyML-style performance estimation territory).",
        "- Verdict: for pure column drift, use Evidently. The value of this project is the "
        "**agentic decision layer and the safety gate** on top of correct, understood detectors.",
    ]
    if not evidently_available:
        lines += [
            "",
            "> Evidently columns show `n/a` because it isn't installed in this environment. "
            "Install with `pip install -e .[benchmark]` and re-run `python scripts/benchmark_evidently.py`.",
        ]

    (out_dir / "BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
    status = "with Evidently" if evidently_available else "(Evidently not installed — ours only)"
    print(f"Wrote {out_dir / 'BENCHMARK.md'} {status}")


if __name__ == "__main__":
    main()
