from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
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


@dataclass
class TrainResult:
    model_path: str
    mlflow_run_id: str
    metrics: dict
    reference_path: str


def generate_reference_dataset(n_samples: int = 5_000, random_state: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=5,
        n_redundant=1,
        weights=[0.78, 0.22],
        flip_y=0.02,
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


def train_baseline(df: pd.DataFrame | None = None) -> TrainResult:
    if df is None:
        df = generate_reference_dataset()

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=150, max_depth=3, random_state=42
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    model_path = Path(settings.model_artifact_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_names": FEATURE_NAMES}, model_path)

    reference_path = Path(settings.reference_data_path)
    df.to_parquet(reference_path, index=False)

    run_id = log_training_run(pipeline, metrics, reference_path)

    return TrainResult(
        model_path=str(model_path),
        mlflow_run_id=run_id,
        metrics=metrics,
        reference_path=str(reference_path),
    )


def load_model():
    bundle = joblib.load(settings.model_artifact_path)
    return bundle["pipeline"], bundle["feature_names"]
