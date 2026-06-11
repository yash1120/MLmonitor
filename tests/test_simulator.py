from __future__ import annotations

import pandas as pd

from mlmonitor.config import settings
from mlmonitor.drift.data_drift import feature_drift, summarise_data_drift
from mlmonitor.models.train import FEATURE_NAMES, TARGET_NAME, generate_reference_dataset
from mlmonitor.simulator.production_stream import (
    append_to_production_log,
    generate_production_batch,
    load_production_window,
)


def test_clean_batch_matches_reference_distribution() -> None:
    """Regression test: clean batches must NOT register as drifted.

    The simulator once re-seeded the dataset generator per batch, which changed
    the underlying distribution and made every batch (even drift_mode='none')
    look drifted against the reference.
    """
    reference = generate_reference_dataset()
    batch = generate_production_batch(n=2_000, drift_mode="none", seed=7)
    summary = summarise_data_drift(
        feature_drift(reference, batch, FEATURE_NAMES, psi_alert=0.25)
    )
    assert summary["psi_max"] < settings.psi_warn_threshold
    assert summary["drifted_features"] == []


def test_covariate_batch_is_detectably_drifted() -> None:
    reference = generate_reference_dataset()
    batch = generate_production_batch(n=2_000, drift_mode="covariate", intensity=1.0, seed=7)
    summary = summarise_data_drift(
        feature_drift(reference, batch, FEATURE_NAMES, psi_alert=0.25)
    )
    assert summary["psi_max"] >= settings.psi_alert_threshold
    assert "monthly_spend" in summary["drifted_features"]


def test_concept_batch_changes_labels_not_features() -> None:
    reference = generate_reference_dataset()
    batch = generate_production_batch(n=2_000, drift_mode="concept", intensity=1.0, seed=7)
    summary = summarise_data_drift(
        feature_drift(reference, batch, FEATURE_NAMES, psi_alert=0.25)
    )
    assert summary["psi_max"] < settings.psi_warn_threshold  # features untouched
    clean = generate_production_batch(n=2_000, drift_mode="none", seed=7)
    assert (batch[TARGET_NAME] != clean[TARGET_NAME]).mean() > 0.05  # labels moved


def test_batch_schema_and_determinism() -> None:
    a = generate_production_batch(n=100, seed=3)
    b = generate_production_batch(n=100, seed=3)
    assert list(a.columns) == ["event_ts", *FEATURE_NAMES, TARGET_NAME]
    pd.testing.assert_frame_equal(
        a.drop(columns="event_ts"), b.drop(columns="event_ts")
    )


def test_production_log_append_and_window(sandbox) -> None:
    batch = generate_production_batch(n=200, seed=1)
    append_to_production_log(batch)
    append_to_production_log(generate_production_batch(n=300, seed=2))
    window = load_production_window(window_hours=24 * 14)
    assert len(window) == 500
    assert load_production_window(window_hours=1).shape[0] <= 500
