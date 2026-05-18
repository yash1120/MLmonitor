from __future__ import annotations

import numpy as np
import pandas as pd

from mlmonitor.drift.data_drift import (
    feature_drift,
    population_stability_index,
    summarise_data_drift,
)


def test_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5_000)
    psi = population_stability_index(x, x.copy())
    assert psi < 1e-6


def test_psi_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(loc=0, scale=1, size=5_000)
    prod = rng.normal(loc=2, scale=1, size=5_000)
    psi = population_stability_index(ref, prod)
    assert psi > 0.5


def test_feature_drift_flags_drifted_only() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {
            "stable": rng.normal(size=2_000),
            "shifted": rng.normal(size=2_000),
        }
    )
    prod = pd.DataFrame(
        {
            "stable": rng.normal(size=2_000),
            "shifted": rng.normal(loc=3.0, size=2_000),
        }
    )
    results = feature_drift(ref, prod, features=["stable", "shifted"], psi_alert=0.25)
    by_name = {r.feature: r for r in results}
    assert by_name["stable"].drifted is False
    assert by_name["shifted"].drifted is True


def test_summary_keys() -> None:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=1_000)})
    prod = pd.DataFrame({"a": rng.normal(loc=1.5, size=1_000)})
    results = feature_drift(ref, prod, features=["a"])
    summary = summarise_data_drift(results)
    assert {"psi_max", "psi_mean", "drifted_features", "by_feature"}.issubset(summary)
