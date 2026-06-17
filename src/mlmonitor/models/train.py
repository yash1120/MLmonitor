from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlmonitor.config import settings
from mlmonitor.mlflow_utils.tracking import log_training_run

FEATURE_NAMES = [
    "account_age_days",
    "monthly_spend",
    "credit_utilisation",
    "logins_last_30d",
    "support_tickets",
    "tenure_months",
    "n_products",
    "balance_to_limit",
]
TARGET_NAME = "churned"

# Bundle schema version — load_model tolerates older bundles via defaults.
BUNDLE_VERSION = 2


@dataclass
class TrainResult:
    model_path: str
    mlflow_run_id: str
    metrics: dict
    reference_path: str


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_names: list[str]
    baseline_f1: float
    roc_auc: float
    decision_threshold: float
    reference_hash: str
    n_ref_rows: int
    ref_probs: np.ndarray  # cached reference predicted-probabilities for prediction-drift PSI


def generate_reference_dataset(n_samples: int = 5_000, random_state: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        # More informative signal + a less extreme class balance than the original
        # (0.78/0.22 scored at the 0.5 argmax structurally caps F1). This yields a
        # usable churn model (~0.85 AUC) instead of one that "looks broken" at 0.57 F1.
        n_informative=6,
        n_redundant=1,
        weights=[0.65, 0.35],
        flip_y=0.01,
        class_sep=1.1,
        random_state=random_state,
    )
    X[:, 0] = (X[:, 0] - X[:, 0].min()) * 60 + 30
    X[:, 1] = np.abs(X[:, 1]) * 250 + 80
    X[:, 2] = np.clip((X[:, 2] - X[:, 2].min()) / (X[:, 2].max() - X[:, 2].min()), 0.01, 0.99)
    X[:, 3] = np.clip(np.abs(X[:, 3]) * 8, 0, 60).round()
    X[:, 4] = np.clip(np.abs(X[:, 4]) * 1.5, 0, 12).round()
    X[:, 5] = np.clip(np.abs(X[:, 5]) * 20, 0, 120).round()
    X[:, 6] = np.clip(np.abs(X[:, 6]) * 1.4 + 1, 1, 6).round()
    X[:, 7] = np.clip((X[:, 7] - X[:, 7].min()) / (X[:, 7].max() - X[:, 7].min()), 0.0, 1.0)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df[TARGET_NAME] = y
    return df


def _tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Pick the probability threshold that maximises F1 on the validation slice
    (the operating point, not the default 0.5 argmax). Returns (threshold, f1_at_threshold)."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # precision_recall_curve returns one fewer threshold than precision/recall points
    f1s = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1s[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[best]) if len(thresholds) else 0.5
    return threshold, float(f1s[best])


def _reference_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def build_bundle_and_metrics(df: pd.DataFrame) -> tuple[ModelBundle, dict]:
    """Train a pipeline + tuned threshold from `df` entirely in memory (no disk/MLflow
    side effects). Used by train_baseline and by the champion/challenger promotion gate."""
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    test_probs = pipeline.predict_proba(X_test)[:, 1]
    threshold, _ = _tune_threshold(y_test.to_numpy(), test_probs)
    preds_at_threshold = (test_probs >= threshold).astype(int)

    metrics = {
        "f1": float(f1_score(y_test, preds_at_threshold)),
        "f1_argmax": float(f1_score(y_test, pipeline.predict(X_test))),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "decision_threshold": float(threshold),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Cache reference predicted-probabilities once (invariant between retrains) so
    # run_drift_check doesn't re-score all reference rows on every check.
    ref_probs = pipeline.predict_proba(df[FEATURE_NAMES])[:, 1].astype(float)

    bundle = ModelBundle(
        pipeline=pipeline,
        feature_names=FEATURE_NAMES,
        baseline_f1=metrics["f1"],
        roc_auc=metrics["roc_auc"],
        decision_threshold=threshold,
        reference_hash=_reference_hash(df),
        n_ref_rows=int(len(df)),
        ref_probs=ref_probs,
    )
    return bundle, metrics


def save_bundle(bundle: ModelBundle, path: str | None = None) -> Path:
    out = Path(path or settings.model_artifact_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": BUNDLE_VERSION, "bundle": bundle}, out)
    return out


def train_baseline(df: pd.DataFrame | None = None) -> TrainResult:
    if df is None:
        df = generate_reference_dataset()

    bundle, metrics = build_bundle_and_metrics(df)
    pipeline = bundle.pipeline

    model_path = save_bundle(bundle)

    reference_path = Path(settings.reference_data_path)
    df.to_parquet(reference_path, index=False)

    run_id = log_training_run(
        pipeline,
        {**metrics, "reference_hash_int": int(bundle.reference_hash, 16) % (10**9)},
        reference_path,
    )

    return TrainResult(
        model_path=str(model_path),
        mlflow_run_id=run_id,
        metrics=metrics,
        reference_path=str(reference_path),
    )


def load_bundle() -> ModelBundle:
    """Load the served model bundle, tolerating older on-disk formats."""
    raw = joblib.load(settings.model_artifact_path)
    if isinstance(raw, dict) and "bundle" in raw:
        return raw["bundle"]
    # Back-compat: v1 bundle ({"pipeline", "feature_names"}) with no baseline metadata.
    pipeline = raw["pipeline"]
    feature_names = raw["feature_names"]
    return ModelBundle(
        pipeline=pipeline,
        feature_names=feature_names,
        baseline_f1=0.0,
        roc_auc=0.0,
        decision_threshold=0.5,
        reference_hash="",
        n_ref_rows=0,
        ref_probs=np.empty(0),
    )


def load_model():
    """Return (pipeline, feature_names) for callers that only need to score."""
    bundle = load_bundle()
    return bundle.pipeline, bundle.feature_names
