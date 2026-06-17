from __future__ import annotations

import contextvars
import json
import re
import time
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from mlmonitor.agent.prompts import INITIAL_HUMAN_PROMPT, SYSTEM_PROMPT
from mlmonitor.config import settings
from mlmonitor.logging_utils import get_logger
from mlmonitor.storage.db import (
    recent_drift_checks,
    save_retrain_audit,
    seconds_since_last_dispatch,
)

logger = get_logger(__name__)

# Per-invocation active report. A ContextVar is copied per thread/task, so concurrent
# /monitor/check requests on the anyio threadpool can NOT clobber each other's report
# (the bug a module-global had). Tools read it lazily at call time.
_active_report: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "active_report", default=None
)


def _report() -> dict[str, Any]:
    return _active_report.get() or {}


# Reject anything that isn't plain text before it reaches the workflow-dispatch input.
_RETRAIN_REASON_RE = re.compile(r"^[\w\s.,:;()\-/%>=<]+$")


class Recommendation(BaseModel):
    action: str
    priority: str = "medium"
    reason: str = ""


class Verdict(BaseModel):
    """Typed agent verdict — bound via with_structured_output so the model returns
    schema-valid JSON directly instead of us brace-scraping free text."""

    diagnosis: str = ""
    recommendations: list[Recommendation] = Field(default_factory=list)
    trigger_retraining: bool = False


@tool
def get_top_drifting_features(top_k: int = 5) -> str:
    """Return the top K features ranked by PSI value with ref/prod means and stds."""
    by_feature = _report().get("by_feature", [])
    ranked = sorted(by_feature, key=lambda r: r.get("psi", 0.0), reverse=True)[:top_k]
    if not ranked:
        return "No per-feature drift data available."
    return json.dumps(ranked, default=str)


@tool
def inspect_feature(feature: str) -> str:
    """Inspect a single feature: its PSI, KS test, and reference vs production statistics."""
    by_feature = _report().get("by_feature", [])
    match = next((r for r in by_feature if r.get("feature") == feature), None)
    if not match:
        available = [r.get("feature") for r in by_feature]
        return f"Feature {feature!r} not found. Available: {available}"
    return json.dumps(match, default=str)


@tool
def get_performance_metrics() -> str:
    """Return current production F1/AUC and the drop vs baseline."""
    report = _report()
    perf = {
        "perf_f1": report.get("perf_f1"),
        "perf_roc_auc": report.get("perf_roc_auc"),
        "perf_drop": report.get("perf_drop"),
        "baseline_f1": report.get("baseline_f1"),
        "prediction_drift_psi": report.get("prediction_drift_psi"),
        "n_samples": report.get("n_samples"),
        "concept_drift": report.get("concept_drift"),
    }
    return json.dumps(perf, default=str)


@tool
def get_recent_drift_history(limit: int = 10) -> str:
    """Return the most recent drift checks (status + PSI + F1) so trends can be assessed."""
    history = recent_drift_checks(limit=limit)
    return json.dumps(history, default=str)


@tool
def get_drift_attribution() -> str:
    """Return within-window temporal slices (drift velocity), whether drift is sustained
    vs a transient blip, and per-segment drift — to localise drift and judge persistence."""
    attribution = _report().get("attribution", {})
    if not attribution:
        return "No attribution data available."
    return json.dumps(attribution, default=str)


def _gate_retraining(report: dict[str, Any]) -> tuple[bool, str]:
    """Deterministic retrain gate — the LLM proposes, this code disposes.

    Returns (allowed, message). Enforces the both-conditions invariant, the global
    kill-switch, and the minimum dispatch interval, independent of what the LLM claims.
    """
    psi_max = float(report.get("psi_max", 0.0) or 0.0)
    perf_drop = float(report.get("perf_drop", 0.0) or 0.0)

    if not settings.retrain_enabled:
        return False, "BLOCKED: retraining disabled (RETRAIN_ENABLED=false)."
    if psi_max < settings.psi_alert_threshold:
        return False, (
            f"BLOCKED: psi_max={psi_max:.4f} < {settings.psi_alert_threshold} — "
            "data-drift condition not met."
        )
    if perf_drop < settings.perf_drop_alert:
        return False, (
            f"BLOCKED: f1_drop={perf_drop:.4f} < {settings.perf_drop_alert} — "
            "performance-degradation condition not met."
        )
    since = seconds_since_last_dispatch()
    if since is not None and since < settings.retrain_min_interval_minutes * 60:
        return False, (
            f"BLOCKED: last retrain was {since / 60:.1f}m ago "
            f"(< {settings.retrain_min_interval_minutes}m rate limit)."
        )
    return True, "OK"


@tool
def trigger_retraining(reason: str) -> str:
    """Request the GitHub Actions retraining workflow. The request is only honoured if the
    deterministic gate (both drift conditions met, enabled, not rate-limited) passes."""
    report = _report()
    psi_max = float(report.get("psi_max", 0.0) or 0.0)
    perf_drop = float(report.get("perf_drop", 0.0) or 0.0)

    if not _RETRAIN_REASON_RE.match(reason or ""):
        reason = "drift thresholds breached"  # reject anything that isn't plain text
    reason = reason[:200]

    allowed, gate_msg = _gate_retraining(report)
    if not allowed:
        save_retrain_audit("blocked", reason, psi_max, perf_drop, gate_msg)
        logger.warning("Retrain BLOCKED: %s", gate_msg)
        return gate_msg

    if settings.retrain_dry_run:
        save_retrain_audit("dry_run", reason, psi_max, perf_drop, "dry-run; not dispatched")
        logger.info("Retrain DRY-RUN (would dispatch): %s", reason)
        return f"DRY-RUN: gate passed, retraining NOT dispatched. Reason: {reason}"

    if not (settings.github_owner and settings.github_repo and settings.github_token):
        save_retrain_audit("skipped", reason, psi_max, perf_drop, "no GitHub creds")
        return "Gate passed but GitHub Actions not configured (GITHUB_OWNER/REPO/TOKEN). Skipped."

    url = (
        f"https://api.github.com/repos/{settings.github_owner}/{settings.github_repo}"
        f"/actions/workflows/{settings.github_workflow}/dispatches"
    )
    payload = {"ref": settings.github_ref, "inputs": {"reason": reason}}
    try:
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=payload,
            timeout=15.0,
        )
    except httpx.HTTPError as exc:
        save_retrain_audit("blocked", reason, psi_max, perf_drop, "dispatch error")
        logger.error("Retrain dispatch failed: %s", exc)
        return "Workflow dispatch failed (network error)."
    if resp.status_code == 204:
        save_retrain_audit("dispatched", reason, psi_max, perf_drop, "204")
        logger.info("Retraining workflow dispatched. Reason: %s", reason)
        return f"Retraining workflow dispatched. Reason: {reason}"
    save_retrain_audit("blocked", reason, psi_max, perf_drop, f"status {resp.status_code}")
    logger.error("Retrain dispatch returned %s", resp.status_code)
    return f"Workflow dispatch failed (status {resp.status_code})."


TOOLS = [
    get_top_drifting_features,
    inspect_feature,
    get_performance_metrics,
    get_recent_drift_history,
    get_drift_attribution,
    trigger_retraining,
]


def _make_llm() -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it in your .env file."
        )
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
        max_tokens=1024,
    )


def _build_agent(llm: ChatGroq | None = None):
    return create_react_agent(llm or _make_llm(), TOOLS)


def _extract_final_json(text: str) -> dict[str, Any] | None:
    candidates = []
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
        start = text.find("{", start + 1)

    parsed: list[dict[str, Any]] = []
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)

    if not parsed:
        return None
    # Prefer the verdict object (it carries these keys), not a nested
    # recommendation sub-object that also happens to be valid JSON.
    for obj in parsed:
        if "diagnosis" in obj or "trigger_retraining" in obj:
            return obj
    return max(parsed, key=lambda d: len(json.dumps(d)))


def _coerce_verdict(llm: ChatGroq, final_text: str) -> dict[str, Any]:
    """Turn the agent's final message into a typed Verdict. Prefer structured output
    (schema-valid by construction); fall back to brace-scraping only if that fails."""
    try:
        structured = llm.with_structured_output(Verdict)
        verdict: Verdict = structured.invoke(
            "Extract the final verdict from this ML-monitoring analysis as structured "
            f"data. If it recommends retraining, set trigger_retraining true.\n\n{final_text}"
        )
        return verdict.model_dump()
    except Exception as exc:  # network/parse issue — degrade to the heuristic parser
        logger.warning("Structured verdict coercion failed (%s); using fallback parser", exc)
        return _extract_final_json(final_text) or {}


def run_diagnostic_agent(drift_report: dict[str, Any]) -> dict[str, Any]:
    """Run the LangChain/LangGraph agent over a drift report and return its structured verdict."""
    token = _active_report.set(drift_report)
    try:
        llm = _make_llm()
        agent = _build_agent(llm)
        initial_human = INITIAL_HUMAN_PROMPT.format(
            status=drift_report.get("status"),
            psi_max=drift_report.get("psi_max", 0.0),
            psi_warn=settings.psi_warn_threshold,
            psi_alert=settings.psi_alert_threshold,
            psi_mean=drift_report.get("psi_mean", 0.0),
            ks_features_flagged=drift_report.get("ks_features_flagged", 0),
            drifted_features=drift_report.get("drifted_features", []),
            perf_f1=drift_report.get("perf_f1"),
            perf_drop=drift_report.get("perf_drop"),
            prediction_drift_psi=drift_report.get("prediction_drift_psi"),
            window_start=drift_report.get("window_start"),
            window_end=drift_report.get("window_end"),
            n_samples=drift_report.get("n_samples"),
        )
        system_prompt = SYSTEM_PROMPT.format(
            psi_alert=settings.psi_alert_threshold,
            perf_drop=settings.perf_drop_alert,
        )

        messages = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=initial_human),
            ]
        }
        last_exc: Exception | None = None
        state = None
        for attempt in range(3):
            try:
                state = agent.invoke(messages, config={"recursion_limit": 20})
                break
            except Exception as exc:  # Groq free tier rate-limits; back off and retry
                last_exc = exc
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        else:
            raise RuntimeError(f"Agent failed after 3 attempts: {last_exc}") from last_exc

        final_text = ""
        triggered = False
        for msg in reversed(state["messages"]):
            if final_text == "" and getattr(msg, "content", None):
                final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if getattr(msg, "name", None) == "trigger_retraining":
                payload = msg.content if isinstance(msg.content, str) else str(msg.content)
                if "dispatched" in payload.lower():
                    triggered = True

        parsed = _coerce_verdict(llm, final_text)
        return {
            "diagnosis": parsed.get("diagnosis") or final_text.strip(),
            "recommendations": parsed.get("recommendations", []),
            # What the LLM proposed (advisory) vs what the deterministic gate actually did:
            "trigger_retraining": bool(parsed.get("trigger_retraining", triggered)),
            "actually_triggered": triggered,
            "raw_final_message": final_text,
        }
    finally:
        _active_report.reset(token)
