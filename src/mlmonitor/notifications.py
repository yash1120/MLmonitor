"""Free outbound alerting via a Slack/Discord incoming webhook.

No-ops unless ALERT_WEBHOOK_URL is set, exactly like the GitHub-token guard — so the
loop the STORY promises ("watch the watchers") is actually closed on free infra.
"""
from __future__ import annotations

from typing import Any

import httpx

from mlmonitor.config import settings
from mlmonitor.logging_utils import get_logger

logger = get_logger(__name__)


def _format_message(report: dict[str, Any]) -> str:
    status = str(report.get("status", "?")).upper()
    drifted = ", ".join(report.get("drifted_features", []) or []) or "none"
    return (
        f":warning: ML Monitor — *{status}*\n"
        f"• psi_max: {report.get('psi_max', 0):.3f}  "
        f"• F1 drop: {report.get('perf_drop', 0) or 0:.3f}  "
        f"• pred-PSI: {report.get('prediction_drift_psi', 0) or 0:.3f}\n"
        f"• drifted features: {drifted}\n"
        f"• samples: {report.get('n_samples', 0)} "
        f"(labeled {report.get('n_labeled', 0)})"
    )


def notify_drift(report: dict[str, Any]) -> bool:
    """Send a drift alert to the configured webhook. Returns True if a request was sent."""
    url = settings.alert_webhook_url
    if not url:
        return False
    # Slack and Discord both accept a JSON body with a "text"/"content" field; send both keys.
    text = _format_message(report)
    try:
        resp = httpx.post(url, json={"text": text, "content": text}, timeout=10.0)
        if resp.status_code >= 400:
            logger.warning("Alert webhook returned %s", resp.status_code)
            return False
    except httpx.HTTPError as exc:
        logger.warning("Alert webhook failed: %s", exc)
        return False
    logger.info("Drift alert sent to webhook (status=%s)", report.get("status"))
    return True
