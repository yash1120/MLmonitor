"""Localise drift in time and across segments, so triage can say *where* and *whether
it's accelerating* — not just 'psi_max=0.35 globally'."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mlmonitor.drift.data_drift import feature_drift, summarise_data_drift


def temporal_drift(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: Sequence[str],
    n_slices: int = 3,
    psi_alert: float = 0.25,
) -> list[dict]:
    """Split the window into equal time slices and compute psi_max per slice, exposing
    drift velocity (is it a one-off blip or a sustained ramp?)."""
    if production.empty or "event_ts" not in production:
        return []
    prod = production.sort_values("event_ts").reset_index(drop=True)
    slices = np.array_split(prod, min(n_slices, max(1, len(prod) // 50)))
    out = []
    for i, sl in enumerate(slices):
        if sl.empty:
            continue
        summary = summarise_data_drift(feature_drift(reference, sl, features, psi_alert))
        out.append(
            {
                "slice": i,
                "n": int(len(sl)),
                "ts_start": str(sl["event_ts"].min()),
                "ts_end": str(sl["event_ts"].max()),
                "psi_max": round(summary["psi_max"], 4),
            }
        )
    return out


def is_sustained(temporal: list[dict], psi_alert: float = 0.25) -> bool:
    """True if the majority of recent slices breach the alert band (a sustained ramp),
    False if drift is confined to one slice (a transient blip)."""
    if not temporal:
        return False
    breached = sum(1 for s in temporal if s["psi_max"] >= psi_alert)
    return breached >= max(2, (len(temporal) + 1) // 2)


def segment_drift(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: Sequence[str],
    segment_col: str,
    psi_alert: float = 0.25,
    max_segments: int = 6,
) -> list[dict]:
    """Compute psi_max within each value-bucket of a low-cardinality segment column,
    so drift can be attributed to a specific cohort."""
    if segment_col not in production or segment_col not in reference:
        return []
    out = []
    for value in sorted(production[segment_col].dropna().unique())[:max_segments]:
        ref_seg = reference[reference[segment_col] == value]
        prod_seg = production[production[segment_col] == value]
        if len(prod_seg) < 30 or ref_seg.empty:
            continue
        summary = summarise_data_drift(feature_drift(ref_seg, prod_seg, features, psi_alert))
        out.append(
            {"segment": f"{segment_col}={value}", "n": int(len(prod_seg)),
             "psi_max": round(summary["psi_max"], 4)}
        )
    return out


def build_attribution(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: Sequence[str],
    segment_col: str = "n_products",
    psi_alert: float = 0.25,
) -> dict:
    temporal = temporal_drift(reference, production, features, psi_alert=psi_alert)
    return {
        "temporal": temporal,
        "sustained": is_sustained(temporal, psi_alert=psi_alert),
        "segments": segment_drift(reference, production, features, segment_col, psi_alert),
    }
