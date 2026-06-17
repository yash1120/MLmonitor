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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlmonitor.config import settings
from mlmonitor.drift.concept_drift import evaluate_performance, prediction_drift
from mlmonitor.drift.data_drift import feature_drift, summarise_data_drift
from mlmonitor.models.train import (
    FEATURE_NAMES,
    TARGET_NAME,
    _tune_threshold,
    generate_reference_dataset,
)
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


def _train_eval_model(reference) -> tuple[Pipeline, float, float, float, np.ndarray]:
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
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    threshold, _ = _tune_threshold(y_test.to_numpy(), test_probs)
    baseline_f1 = float(f1_score(y_test, (test_probs >= threshold).astype(int)))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    ref_probs = pipeline.predict_proba(reference[FEATURE_NAMES])[:, 1]
    return pipeline, baseline_f1, roc_auc, threshold, ref_probs


def run_trial(pipeline, baseline_f1, threshold, ref_probs, reference, mode, intensity, n, seed) -> dict:
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
        threshold=threshold,
    )
    prod_probs = pipeline.predict_proba(batch[FEATURE_NAMES])[:, 1]
    pred_psi = prediction_drift(ref_probs, prod_probs)
    data_alert = summary["psi_max"] >= settings.psi_alert_threshold
    data_warn = summary["psi_max"] >= settings.psi_warn_threshold
    pred_alert = pred_psi >= settings.pred_drift_alert_threshold
    return {
        "psi_max": summary["psi_max"],
        "ks_flagged": summary["ks_features_flagged"],
        "f1_drop": perf.f1_drop,
        "pred_psi": pred_psi,
        "data_alert": bool(data_alert),
        "data_warn_or_alert": bool(data_warn),
        "concept_alert": bool(perf.concept_drift),
        # Mirror the live _classify_status: alert on data drift, confirmed F1 drop,
        # OR a strong label-free prediction-distribution shift.
        "any_alert": bool(data_alert or perf.concept_drift or pred_alert),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30, help="trials per scenario")
    parser.add_argument("--n", type=int, default=2000, help="batch size per trial")
    parser.add_argument("--out-dir", default="eval")
    args = parser.parse_args()

    reference = generate_reference_dataset()
    pipeline, baseline_f1, roc_auc, threshold, ref_probs = _train_eval_model(reference)

    results = []
    for mode, intensity in SCENARIOS:
        trials = [
            run_trial(
                pipeline, baseline_f1, threshold, ref_probs, reference, mode, intensity, args.n, seed
            )
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
        json.dumps({"baseline_f1": baseline_f1, "roc_auc": roc_auc,
                    "decision_threshold": threshold, "false_positive_rate": fp_rate,
                    "batch_size": args.n, "scenarios": results}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Drift-Detection Evaluation",
        "",
        f"Pipeline: PSI (alert ≥ {settings.psi_alert_threshold}) + label-free prediction-PSI "
        f"+ F1-drop (alert ≥ {settings.perf_drop_alert}) — the same code path the live monitor runs.",
        "",
        f"- Trials per scenario: **{args.trials}**, batch size: **{args.n}**",
        f"- Baseline model: **F1 {baseline_f1:.3f}** / **ROC-AUC {roc_auc:.3f}** "
        f"(tuned decision threshold {threshold:.3f})",
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
        "- **PSI is bounded** (production is clipped into the reference support), so values are "
        "interpretable against the standard 0.10/0.25 bands rather than saturating on tail shifts.",
        "- **KS is diagnostic only** — its p-value collapses toward 0 at large N, so it does NOT "
        "drive alerts; PSI (an effect-size measure) is the data-drift gate. KS is reported with a "
        "≥0.10 statistic floor as supporting colour.",
        "- Concept drift uses the labelled slice only (label latency is modelled); the label-free "
        "prediction-PSI is the online early-warning signal.",
        "- Regenerate with: `python scripts/evaluate_drift.py --trials 30 --n 2000`",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir / 'RESULTS.md'} and {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
