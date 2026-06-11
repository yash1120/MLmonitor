from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


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
    drifted: bool


def _bin_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        edges = np.array([reference.min() - 1e-6, reference.max() + 1e-6])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: np.ndarray, production: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between two 1-D arrays. <0.1 stable, 0.1-0.25 minor, >0.25 major shift."""
    reference = np.asarray(reference, dtype=float)
    production = np.asarray(production, dtype=float)
    if reference.size == 0 or production.size == 0:
        return 0.0

    edges = _bin_edges(reference, n_bins=n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    prod_counts, _ = np.histogram(production, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    prod_pct = prod_counts / max(prod_counts.sum(), 1) + eps

    psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
    return psi


def feature_drift(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: Sequence[str],
    psi_alert: float = 0.25,
) -> list[FeatureDriftResult]:
    results: list[FeatureDriftResult] = []
    for feat in features:
        ref_vals = reference[feat].to_numpy(dtype=float)
        prod_vals = production[feat].to_numpy(dtype=float)
        psi = population_stability_index(ref_vals, prod_vals)
        try:
            ks_stat, ks_p = stats.ks_2samp(ref_vals, prod_vals)
        except ValueError:
            ks_stat, ks_p = 0.0, 1.0
        results.append(
            FeatureDriftResult(
                feature=feat,
                psi=psi,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_p),
                ref_mean=float(np.mean(ref_vals)),
                prod_mean=float(np.mean(prod_vals)),
                ref_std=float(np.std(ref_vals)),
                prod_std=float(np.std(prod_vals)),
                drifted=bool(psi >= psi_alert or ks_p < 0.01),
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
        "ks_features_flagged": int(sum(1 for r in results if r.ks_pvalue < 0.01)),
        "drifted_features": drifted,
        "by_feature": [r.__dict__ for r in results],
    }
