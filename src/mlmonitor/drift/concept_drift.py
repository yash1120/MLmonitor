from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class PerformanceResult:
    f1: float
    roc_auc: float
    n_samples: int
    baseline_f1: float
    f1_drop: float
    concept_drift: bool


def evaluate_performance(
    pipeline,
    production: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    baseline_f1: float,
    drop_threshold: float = 0.05,
) -> PerformanceResult:
    """Score production-with-labels and flag concept drift if F1 drops vs baseline."""
    if target_name not in production.columns or production.empty:
        return PerformanceResult(
            f1=0.0,
            roc_auc=0.0,
            n_samples=int(len(production)),
            baseline_f1=baseline_f1,
            f1_drop=0.0,
            concept_drift=False,
        )

    X = production[feature_names]
    y = production[target_name].to_numpy()

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]

    f1 = float(f1_score(y, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = 0.0

    drop = max(baseline_f1 - f1, 0.0)
    return PerformanceResult(
        f1=f1,
        roc_auc=auc,
        n_samples=int(len(production)),
        baseline_f1=float(baseline_f1),
        f1_drop=float(drop),
        concept_drift=drop >= drop_threshold,
    )


def prediction_drift(
    reference_probs: np.ndarray, production_probs: np.ndarray, n_bins: int = 10
) -> float:
    """PSI on predicted-probability distribution as a label-free concept-drift proxy."""
    from mlmonitor.drift.data_drift import population_stability_index

    return population_stability_index(reference_probs, production_probs, n_bins=n_bins)
