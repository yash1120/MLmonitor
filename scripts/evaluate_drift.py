"""Offline evaluation of the drift detectors: detection rate vs false-positive rate.

Runs the full detection pipeline (PSI + KS + F1-drop classification) over many
simulated production batches at varying drift modes and intensities, with no
MLflow/DB side effects, and reports per-scenario detection rates.

Usage:
    PYTHONPATH=src python scripts/evaluate_drift.py --trials 30 --n 2000
Writes eval/results.json and eval/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlmonitor.config import settings
from mlmonitor.drift.concept_drift import evaluate_performance
from mlmonitor.drift.data_drift import feature_drift, summarise_data_drift
from mlmonitor.models.train import FEATURE_NAMES, TARGET_NAME, generate_reference_dataset
from mlmonitor.simulator.production_stream import generate_production_batch

SCENARIOS = [
    ("none", 0.0),
    ("covariate", 0.25),
    ("covariate", 0.5),
    ("covariate", 1.0),
    ("concept", 0.5),
    ("concept", 1.0),
    ("both", 0.5),
    ("both", 1.0),
]


def _train_eval_model(reference) -> tuple[Pipeline, float]:
    X, y = reference[FEATURE_NAMES], reference[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    baseline_f1 = float(f1_score(y_test, pipeline.predict(X_test)))
    return pipeline, baseline_f1


def run_trial(pipeline, baseline_f1, reference, mode, intensity, n, seed) -> dict:
    batch = generate_production_batch(n=n, drift_mode=mode, intensity=intensity, seed=seed)
    summary = summarise_data_drift(
        feature_drift(reference, batch, FEATURE_NAMES, psi_alert=settings.psi_alert_threshold)
    )
    perf = evaluate_performance(
        pipeline=pipeline,
        production=batch,
        feature_names=FEATURE_NAMES,
        target_name=TARGET_NAME,
        baseline_f1=baseline_f1,
        drop_threshold=settings.perf_drop_alert,
    )
    data_alert = summary["psi_max"] >= settings.psi_alert_threshold
    data_warn = summary["psi_max"] >= settings.psi_warn_threshold
    return {
        "psi_max": summary["psi_max"],
        "ks_flagged": summary["ks_features_flagged"],
        "f1_drop": perf.f1_drop,
        "data_alert": bool(data_alert),
        "data_warn_or_alert": bool(data_warn),
        "concept_alert": bool(perf.concept_drift),
        "any_alert": bool(data_alert or perf.concept_drift),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30, help="trials per scenario")
    parser.add_argument("--n", type=int, default=2000, help="batch size per trial")
    parser.add_argument("--out-dir", default="eval")
    args = parser.parse_args()

    reference = generate_reference_dataset()
    pipeline, baseline_f1 = _train_eval_model(reference)

    results = []
    for mode, intensity in SCENARIOS:
        trials = [
            run_trial(pipeline, baseline_f1, reference, mode, intensity, args.n, seed)
            for seed in range(args.trials)
        ]
        expected_data = mode in ("covariate", "both")
        expected_concept = mode in ("concept", "both")
        row = {
            "mode": mode,
            "intensity": intensity,
            "trials": args.trials,
            "data_drift_detection_rate": float(np.mean([t["data_alert"] for t in trials])),
            "warn_or_alert_rate": float(np.mean([t["data_warn_or_alert"] for t in trials])),
            "concept_drift_detection_rate": float(np.mean([t["concept_alert"] for t in trials])),
            "any_alert_rate": float(np.mean([t["any_alert"] for t in trials])),
            "mean_psi_max": float(np.mean([t["psi_max"] for t in trials])),
            "mean_f1_drop": float(np.mean([t["f1_drop"] for t in trials])),
            "expected_data_drift": expected_data,
            "expected_concept_drift": expected_concept,
        }
        results.append(row)
        print(
            f"{mode:>9} @ {intensity:<4} | data={row['data_drift_detection_rate']:.0%} "
            f"concept={row['concept_drift_detection_rate']:.0%} "
            f"psi_max={row['mean_psi_max']:.3f} f1_drop={row['mean_f1_drop']:.3f}"
        )

    fp_rate = next(r for r in results if r["mode"] == "none")["any_alert_rate"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"baseline_f1": baseline_f1, "false_positive_rate": fp_rate,
                    "batch_size": args.n, "scenarios": results}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Drift-Detection Evaluation",
        "",
        f"Pipeline: PSI (alert ≥ {settings.psi_alert_threshold}) + KS test + "
        f"F1-drop (alert ≥ {settings.perf_drop_alert}) — the same code path the live monitor runs.",
        "",
        f"- Trials per scenario: **{args.trials}**, batch size: **{args.n}**",
        f"- Baseline F1 (held-out): **{baseline_f1:.3f}**",
        f"- **False-positive rate on clean data: {fp_rate:.1%}**",
        "",
        "| Mode | Intensity | Data-drift detection | Concept-drift detection | Any alert | Mean PSI max | Mean F1 drop |",
        "|------|-----------|---------------------|------------------------|-----------|--------------|--------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['mode']} | {r['intensity']} | {r['data_drift_detection_rate']:.0%} | "
            f"{r['concept_drift_detection_rate']:.0%} | {r['any_alert_rate']:.0%} | "
            f"{r['mean_psi_max']:.3f} | {r['mean_f1_drop']:.3f} |"
        )
    lines += [
        "",
        "Notes:",
        "- `none` row measures false positives: any alert on clean data is a false alarm.",
        "- Covariate intensity 0.25 is a deliberately subtle shift; partial detection there is expected",
        "  and is exactly the regime where the warn band (PSI ≥ 0.1) plus the agent's trend check earn their keep.",
        "- Regenerate with: `PYTHONPATH=src python scripts/evaluate_drift.py --trials 30 --n 2000`",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir / 'RESULTS.md'} and {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
