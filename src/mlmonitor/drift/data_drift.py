from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Columns with at most this many distinct reference values are treated as categorical
# (value-frequency PSI) instead of being quantile-binned as continuous.
CATEGORICAL_MAX_CARDINALITY = 15
# Minimum KS statistic (CDF gap) for a feature to be *reported* as KS-flagged. KS
# p-values collapse toward 0 at large N, so an effect-size floor keeps the diagnostic honest.
KS_EFFECT_FLOOR = 0.1


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    ref_mean: float
    prod_mean: float
    ref_std: float
    prod_std: float
    missing_rate_delta: float
    is_categorical: bool
    drifted: bool


def _bin_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Finite, reference-derived quantile edges. Production values are clipped into
    [ref_min, ref_max] before binning, so PSI stays bounded and interpretable against
    the standard 0.1/0.25 bands (infinite end-bins made PSI saturate on any tail shift)."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        lo, hi = float(reference.min()), float(reference.max())
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        edges = np.array([lo, hi])
    return edges


def _psi_from_counts(ref_counts: np.ndarray, prod_counts: np.ndarray) -> float:
    """PSI with Laplace smoothing applied as a pseudo-count BEFORE normalisation, so
    both distributions remain proper (sum to 1)."""
    ref_counts = ref_counts.astype(float) + 1.0
    prod_counts = prod_counts.astype(float) + 1.0
    ref_pct = ref_counts / ref_counts.sum()
    prod_pct = prod_counts / prod_counts.sum()
    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def population_stability_index(
    reference: np.ndarray, production: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between two 1-D arrays. <0.1 stable, 0.1-0.25 minor, >0.25 major shift.

    NaNs are treated as their own bin so a rise in missingness registers as drift."""
    reference = np.asarray(reference, dtype=float)
    production = np.asarray(production, dtype=float)
    if reference.size == 0 or production.size == 0:
        return 0.0

    ref_nan = int(np.isnan(reference).sum())
    prod_nan = int(np.isnan(production).sum())
    ref_finite = reference[~np.isnan(reference)]
    prod_finite = production[~np.isnan(production)]

    if ref_finite.size == 0:
        return 0.0

    edges = _bin_edges(ref_finite, n_bins=n_bins)
    # Clip production into the reference support so out-of-range mass lands in the
    # nearest finite bin rather than blowing up an infinite tail bin.
    prod_clipped = np.clip(prod_finite, edges[0], edges[-1])
    ref_counts, _ = np.histogram(ref_finite, bins=edges)
    prod_counts, _ = np.histogram(prod_clipped, bins=edges)

    # Append a dedicated missing-value bin.
    ref_counts = np.append(ref_counts, ref_nan)
    prod_counts = np.append(prod_counts, prod_nan)
    return _psi_from_counts(ref_counts, prod_counts)


def categorical_psi(reference: pd.Series, production: pd.Series) -> float:
    """PSI on the value-frequency table for discrete / low-cardinality features
    (quantile binning collapses to 2-3 bins on these and hides mix shifts)."""
    ref = reference.copy()
    prod = production.copy()
    categories = sorted(set(ref.dropna().unique()) | set(prod.dropna().unique()))
    ref_counts = np.array([(ref == c).sum() for c in categories] + [int(ref.isna().sum())])
    prod_counts = np.array([(prod == c).sum() for c in categories] + [int(prod.isna().sum())])
    if ref_counts.sum() == 0 or prod_counts.sum() == 0:
        return 0.0
    return _psi_from_counts(ref_counts, prod_counts)


def _is_categorical(ref_series: pd.Series) -> bool:
    return ref_series.dropna().nunique() <= CATEGORICAL_MAX_CARDINALITY


def feature_drift(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: Sequence[str],
    psi_alert: float = 0.25,
) -> list[FeatureDriftResult]:
    results: list[FeatureDriftResult] = []
    for feat in features:
        ref_series = reference[feat]
        prod_series = production[feat]
        ref_vals = ref_series.to_numpy(dtype=float)
        prod_vals = prod_series.to_numpy(dtype=float)

        categorical = _is_categorical(ref_series)
        if categorical:
            psi = categorical_psi(ref_series, prod_series)
        else:
            psi = population_stability_index(ref_vals, prod_vals)

        ref_finite = ref_vals[~np.isnan(ref_vals)]
        prod_finite = prod_vals[~np.isnan(prod_vals)]
        try:
            ks_stat, ks_p = stats.ks_2samp(ref_finite, prod_finite)
        except ValueError:
            ks_stat, ks_p = 0.0, 1.0

        ref_missing = float(np.isnan(ref_vals).mean()) if ref_vals.size else 0.0
        prod_missing = float(np.isnan(prod_vals).mean()) if prod_vals.size else 0.0

        results.append(
            FeatureDriftResult(
                feature=feat,
                psi=psi,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_p),
                ref_mean=float(np.nanmean(ref_vals)) if ref_finite.size else 0.0,
                prod_mean=float(np.nanmean(prod_vals)) if prod_finite.size else 0.0,
                ref_std=float(np.nanstd(ref_vals)) if ref_finite.size else 0.0,
                prod_std=float(np.nanstd(prod_vals)) if prod_finite.size else 0.0,
                missing_rate_delta=float(prod_missing - ref_missing),
                is_categorical=categorical,
                # PSI is the effect-size gate that drives alerts. KS is diagnostic only
                # (its p-value collapses at large N), so it does NOT flag drift here.
                drifted=bool(psi >= psi_alert),
            )
        )
    return results


def summarise_data_drift(results: list[FeatureDriftResult]) -> dict:
    if not results:
        return {"psi_max": 0.0, "psi_mean": 0.0, "ks_features_flagged": 0, "drifted_features": []}
    psis = [r.psi for r in results]
    drifted = [r.feature for r in results if r.drifted]
    return {
        "psi_max": float(np.max(psis)),
        "psi_mean": float(np.mean(psis)),
        # KS flag now requires a meaningful effect size (statistic >= floor) AND a small
        # p-value — diagnostic colour, not an alert driver.
        "ks_features_flagged": int(
            sum(1 for r in results if r.ks_pvalue < 0.01 and r.ks_statistic >= KS_EFFECT_FLOOR)
        ),
        "max_missing_rate_delta": float(max((r.missing_rate_delta for r in results), default=0.0)),
        "drifted_features": drifted,
        "by_feature": [r.__dict__ for r in results],
    }
