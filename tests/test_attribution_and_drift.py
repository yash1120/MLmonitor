from __future__ import annotations

import numpy as np
import pandas as pd

from mlmonitor.drift.attribution import build_attribution, is_sustained
from mlmonitor.drift.data_drift import (
    categorical_psi,
    feature_drift,
    population_stability_index,
)
from mlmonitor.models.train import FEATURE_NAMES, generate_reference_dataset
from mlmonitor.simulator.production_stream import generate_production_batch


def test_psi_is_bounded_under_extreme_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5_000)
    prod = rng.normal(50, 1, 5_000)  # massively out of reference support
    psi = population_stability_index(ref, prod)
    assert np.isfinite(psi)
    assert psi < 20  # clipped into ref support → bounded, not inf


def test_psi_flags_rising_missingness():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=2_000)
    prod = rng.normal(size=2_000)
    prod[: int(0.4 * len(prod))] = np.nan  # 40% missing in production
    psi = population_stability_index(ref, prod)
    assert psi > 0.1


def test_categorical_psi_detects_mix_shift():
    ref = pd.Series([1] * 800 + [2] * 200)
    prod = pd.Series([1] * 200 + [2] * 800)
    assert categorical_psi(ref, prod) > 0.25


def test_low_cardinality_feature_treated_as_categorical():
    reference = generate_reference_dataset()
    batch = generate_production_batch(n=2000, drift_mode="none", seed=2)
    results = {r.feature: r for r in feature_drift(reference, batch, FEATURE_NAMES)}
    assert results["n_products"].is_categorical is True
    assert results["monthly_spend"].is_categorical is False


def test_temporal_and_sustained_attribution():
    reference = generate_reference_dataset()
    drifted = generate_production_batch(n=2000, drift_mode="covariate", intensity=1.5, seed=3)
    attribution = build_attribution(reference, drifted, FEATURE_NAMES)
    assert attribution["temporal"]  # slices produced
    # a strong, whole-window covariate shift should read as sustained
    assert attribution["sustained"] is True


def test_transient_blip_not_sustained():
    # only one slice breaches → not sustained
    temporal = [
        {"slice": 0, "psi_max": 0.02},
        {"slice": 1, "psi_max": 0.5},
        {"slice": 2, "psi_max": 0.02},
    ]
    assert is_sustained(temporal) is False
