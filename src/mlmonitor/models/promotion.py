"""Champion/challenger promotion gate — closes the retrain loop.

A retrained candidate must BEAT the incumbent champion on the latest labelled production
window before it can become the served model. This turns "I dispatch a CI job" into
"I run a controlled, gated model-promotion loop" — and it's entirely free/local.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import f1_score

from mlmonitor.config import settings
from mlmonitor.logging_utils import get_logger
from mlmonitor.models.train import (
    FEATURE_NAMES,
    TARGET_NAME,
    ModelBundle,
    build_bundle_and_metrics,
    generate_reference_dataset,
    load_bundle,
    save_bundle,
)
from mlmonitor.simulator.production_stream import load_production_window

logger = get_logger(__name__)


@dataclass
class PromotionResult:
    promoted: bool
    reason: str
    champion_f1: float
    challenger_f1: float
    eval_n: int
    eval_basis: str  # "production_window" | "heldout"


def _score_on(bundle: ModelBundle, df: pd.DataFrame) -> float | None:
    if df is None or df.empty or TARGET_NAME not in df.columns:
        return None
    probs = bundle.pipeline.predict_proba(df[FEATURE_NAMES])[:, 1]
    preds = (probs >= (bundle.decision_threshold or 0.5)).astype(int)
    return float(f1_score(df[TARGET_NAME], preds, zero_division=0))


def evaluate_and_promote(
    df: pd.DataFrame | None = None, epsilon: float = 0.0, window_hours: int = 24 * 30
) -> PromotionResult:
    """Train a challenger on `df` (fresh reference if None) and promote it over the current
    champion only if it scores at least champion_f1 - epsilon on the latest labelled window
    (falling back to held-out F1 when no labelled production data exists)."""
    if df is None:
        df = generate_reference_dataset()

    challenger, ch_metrics = build_bundle_and_metrics(df)

    try:
        champion = load_bundle()
        has_champion = champion.baseline_f1 > 0
    except FileNotFoundError:
        champion, has_champion = None, False

    if not has_champion:
        save_bundle(challenger)
        df.to_parquet(settings.reference_data_path, index=False)
        logger.info("No champion present — challenger promoted as the first champion.")
        return PromotionResult(True, "no incumbent champion", 0.0, challenger.baseline_f1,
                               0, "heldout")

    # Prefer a head-to-head on real labelled production traffic; fall back to held-out F1.
    window = load_production_window(window_hours=window_hours)
    champ_f1 = _score_on(champion, window)
    chal_f1 = _score_on(challenger, window)
    if champ_f1 is not None and chal_f1 is not None and len(window):
        basis, eval_n = "production_window", int(len(window))
    else:
        champ_f1, chal_f1 = champion.baseline_f1, challenger.baseline_f1
        basis, eval_n = "heldout", challenger.n_ref_rows

    if chal_f1 >= champ_f1 - epsilon:
        save_bundle(challenger)
        df.to_parquet(settings.reference_data_path, index=False)
        _register_champion(challenger, ch_metrics, df)
        logger.info(
            "Challenger PROMOTED (%s F1 %.4f >= champion %.4f - %.3f)",
            basis, chal_f1, champ_f1, epsilon,
        )
        return PromotionResult(True, "challenger >= champion", champ_f1, chal_f1, eval_n, basis)

    logger.warning(
        "Challenger REJECTED (%s F1 %.4f < champion %.4f - %.3f) — champion retained",
        basis, chal_f1, champ_f1, epsilon,
    )
    return PromotionResult(False, "challenger worse than champion", champ_f1, chal_f1, eval_n, basis)


def _register_champion(bundle: ModelBundle, metrics: dict, df: pd.DataFrame) -> None:
    """Log the promoted model to MLflow and tag it @champion (best-effort, never fatal)."""
    try:
        from mlmonitor.mlflow_utils.tracking import log_training_run, promote_latest_to_champion

        log_training_run(bundle.pipeline, metrics, settings.reference_data_path)
        uri = promote_latest_to_champion()
        if uri:
            logger.info("Registry champion alias -> %s", uri)
    except Exception as exc:
        logger.warning("MLflow champion registration skipped: %s", exc)
