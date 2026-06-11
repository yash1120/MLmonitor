from __future__ import annotations

import numpy as np
import pandas as pd

from mlmonitor.drift.concept_drift import evaluate_performance, prediction_drift


class PerfectPipeline:
    """Predicts the `signal` column exactly."""

    def predict(self, X):
        return (X["signal"] > 0.5).astype(int).to_numpy()

    def predict_proba(self, X):
        p = (X["signal"] > 0.5).astype(float).to_numpy()
        return np.column_stack([1 - p, np.clip(p, 0.01, 0.99)])


def _frame(n: int, label_noise: float, rng) -> pd.DataFrame:
    signal = rng.random(n)
    y = (signal > 0.5).astype(int)
    flips = rng.random(n) < label_noise
    y[flips] = 1 - y[flips]
    return pd.DataFrame({"signal": signal, "churned": y})


def test_no_drop_when_labels_match() -> None:
    rng = np.random.default_rng(0)
    prod = _frame(2_000, label_noise=0.0, rng=rng)
    result = evaluate_performance(
        PerfectPipeline(), prod, ["signal"], "churned", baseline_f1=1.0, drop_threshold=0.05
    )
    assert result.f1 == 1.0
    assert result.concept_drift is False


def test_concept_drift_flagged_on_label_noise() -> None:
    rng = np.random.default_rng(0)
    prod = _frame(2_000, label_noise=0.3, rng=rng)
    result = evaluate_performance(
        PerfectPipeline(), prod, ["signal"], "churned", baseline_f1=1.0, drop_threshold=0.05
    )
    assert result.f1 < 0.9
    assert result.f1_drop > 0.05
    assert result.concept_drift is True


def test_missing_labels_returns_neutral_result() -> None:
    prod = pd.DataFrame({"signal": [0.1, 0.9]})
    result = evaluate_performance(
        PerfectPipeline(), prod, ["signal"], "churned", baseline_f1=0.9
    )
    assert result.concept_drift is False
    assert result.f1_drop == 0.0


def test_prediction_drift_psi_detects_prob_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.beta(2, 5, size=5_000)
    same = rng.beta(2, 5, size=5_000)
    shifted = rng.beta(5, 2, size=5_000)
    assert prediction_drift(ref, same) < 0.05
    assert prediction_drift(ref, shifted) > 0.5
