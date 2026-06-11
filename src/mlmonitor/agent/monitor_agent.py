from __future__ import annotations

import json
import time
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from mlmonitor.agent.prompts import INITIAL_HUMAN_PROMPT, SYSTEM_PROMPT
from mlmonitor.config import settings
from mlmonitor.storage.db import recent_drift_checks

_current_report: dict[str, Any] = {}


def _set_active_report(report: dict[str, Any]) -> None:
    """Stash the active drift report so tool calls can read it without re-querying."""
    global _current_report
    _current_report = report


@tool
def get_top_drifting_features(top_k: int = 5) -> str:
    """Return the top K features ranked by PSI value with ref/prod means and stds."""
    by_feature = _current_report.get("by_feature", [])
    ranked = sorted(by_feature, key=lambda r: r.get("psi", 0.0), reverse=True)[:top_k]
    if not ranked:
        return "No per-feature drift data available."
    return json.dumps(ranked, default=str)


@tool
def inspect_feature(feature: str) -> str:
    """Inspect a single feature: its PSI, KS test, and reference vs production statistics."""
    by_feature = _current_report.get("by_feature", [])
    match = next((r for r in by_feature if r.get("feature") == feature), None)
    if not match:
        available = [r.get("feature") for r in by_feature]
        return f"Feature {feature!r} not found. Available: {available}"
    return json.dumps(match, default=str)


@tool
def get_performance_metrics() -> str:
    """Return current production F1/AUC and the drop vs baseline."""
    perf = {
        "perf_f1": _current_report.get("perf_f1"),
        "perf_roc_auc": _current_report.get("perf_roc_auc"),
        "perf_drop": _current_report.get("perf_drop"),
        "baseline_f1": _current_report.get("baseline_f1"),
        "n_samples": _current_report.get("n_samples"),
        "concept_drift": _current_report.get("concept_drift"),
    }
    return json.dumps(perf, default=str)


@tool
def get_recent_drift_history(limit: int = 10) -> str:
    """Return the most recent drift checks (status + PSI + F1) so trends can be assessed."""
    history = recent_drift_checks(limit=limit)
    return json.dumps(history, default=str)


@tool
def trigger_retraining(reason: str) -> str:
    """Dispatch the GitHub Actions retraining workflow. Use only when criteria are met."""
    if not (settings.github_owner and settings.github_repo and settings.github_token):
        return "GitHub Actions not configured (missing GITHUB_OWNER/REPO/TOKEN). Skipped."
    url = (
        f"https://api.github.com/repos/{settings.github_owner}/{settings.github_repo}"
        f"/actions/workflows/{settings.github_workflow}/dispatches"
    )
    payload = {"ref": settings.github_ref, "inputs": {"reason": reason[:200]}}
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
        return f"Failed to dispatch workflow: {exc}"
    if resp.status_code == 204:
        return f"Retraining workflow dispatched. Reason: {reason}"
    return f"GitHub returned {resp.status_code}: {resp.text[:200]}"


TOOLS = [
    get_top_drifting_features,
    inspect_feature,
    get_performance_metrics,
    get_recent_drift_history,
    trigger_retraining,
]


def _build_agent():
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it in your .env file."
        )
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
        max_tokens=1024,
    )
    return create_react_agent(llm, TOOLS)


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


def run_diagnostic_agent(drift_report: dict[str, Any]) -> dict[str, Any]:
    """Run the LangChain/LangGraph agent over a drift report and return its structured verdict."""
    _set_active_report(drift_report)

    agent = _build_agent()
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
        window_start=drift_report.get("window_start"),
        window_end=drift_report.get("window_end"),
        n_samples=drift_report.get("n_samples"),
    )

    messages = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=initial_human),
        ]
    }
    last_exc: Exception | None = None
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

    parsed = _extract_final_json(final_text) or {}
    return {
        "diagnosis": parsed.get("diagnosis", final_text.strip()),
        "recommendations": parsed.get("recommendations", []),
        "trigger_retraining": bool(parsed.get("trigger_retraining", triggered)),
        "actually_triggered": triggered,
        "raw_final_message": final_text,
    }
