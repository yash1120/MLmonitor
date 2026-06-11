from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from mlmonitor.config import settings
from mlmonitor.models.train import FEATURE_NAMES, TARGET_NAME, generate_reference_dataset

DriftMode = Literal["none", "covariate", "concept", "both"]


def _inject_covariate_drift(df: pd.DataFrame, intensity: float, rng: np.random.Generator) -> pd.DataFrame:
    """Shift several feature distributions (mean shifts, variance changes)."""
    out = df.copy()
    out["monthly_spend"] = out["monthly_spend"] * (1.0 - 0.35 * intensity) + rng.normal(
        loc=-40 * intensity, scale=15, size=len(out)
    )
    out["credit_utilisation"] = np.clip(
        out["credit_utilisation"] + 0.18 * intensity + rng.normal(0, 0.04, len(out)),
        0.01,
        0.99,
    )
    out["logins_last_30d"] = np.clip(
        out["logins_last_30d"] * (1.0 - 0.5 * intensity), 0, 60
    ).round()
    out["support_tickets"] = np.clip(
        out["support_tickets"] + rng.poisson(lam=1.5 * intensity, size=len(out)), 0, 20
    )
    return out


def _inject_concept_drift(df: pd.DataFrame, intensity: float, rng: np.random.Generator) -> pd.DataFrame:
    """Change the decision boundary: high-utilisation users churn more often."""
    out = df.copy()
    if TARGET_NAME not in out.columns:
        return out
    flip_prob = 0.35 * intensity
    high_util_mask = out["credit_utilisation"] > 0.5
    flips = rng.random(len(out)) < flip_prob
    affected = high_util_mask & flips
    # invert the label (not just set to 1) so the relationship the model
    # learned actually breaks — this is what concept drift means
    out.loc[affected, TARGET_NAME] = 1 - out.loc[affected, TARGET_NAME].astype(int)
    return out


def generate_production_batch(
    n: int = 1_000,
    drift_mode: DriftMode = "none",
    intensity: float = 1.0,
    base_time: datetime | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_time = base_time or datetime.now(UTC)

    # Sample from the SAME distribution as the reference set (fixed generator seed),
    # then jitter slightly so rows aren't literal duplicates. Re-seeding
    # make_classification would change the distribution itself and make even
    # "clean" batches register as drifted.
    # Must match the reference dataset's exact (n_samples, seed): the generator
    # rescales features by the sample's own min/max, so any other size/seed
    # yields a *differently scaled* distribution, not just different rows.
    pool = generate_reference_dataset(n_samples=5_000, random_state=42)
    df = pool.sample(n=n, replace=True, random_state=rng.integers(0, 2**31 - 1)).reset_index(
        drop=True
    )
    discrete = {"logins_last_30d", "support_tickets", "tenure_months", "n_products"}
    for feat in FEATURE_NAMES:
        std = float(pool[feat].std()) or 1.0
        df[feat] = df[feat] + rng.normal(0, 0.01 * std, size=n)
        if feat in discrete:
            # keep count features integral — fractional jitter on a discrete
            # column reads as distribution shift to PSI/KS
            df[feat] = df[feat].round().clip(lower=0)
    if drift_mode in ("covariate", "both"):
        df = _inject_covariate_drift(df, intensity, rng)
    if drift_mode in ("concept", "both"):
        df = _inject_concept_drift(df, intensity, rng)

    timestamps = [
        base_time - timedelta(seconds=int(s))
        for s in rng.integers(0, 60 * 60 * 24 * 7, size=n)
    ]
    df.insert(0, "event_ts", timestamps)
    return df[["event_ts", *FEATURE_NAMES, TARGET_NAME]]


def append_to_production_log(df: pd.DataFrame) -> Path:
    path = Path(settings.production_data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path


def load_production_window(window_hours: int = 24) -> pd.DataFrame:
    path = Path(settings.production_data_path)
    if not path.exists():
        return pd.DataFrame(columns=["event_ts", *FEATURE_NAMES, TARGET_NAME])
    df = pd.read_parquet(path)
    if df.empty:
        return df
    cutoff = df["event_ts"].max() - pd.Timedelta(hours=window_hours)
    return df[df["event_ts"] >= cutoff].copy()
