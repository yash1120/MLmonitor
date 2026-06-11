from __future__ import annotations

from mlmonitor.agent.monitor_agent import _extract_final_json


def test_extracts_plain_json() -> None:
    text = '{"diagnosis": "covariate shift", "recommendations": [], "trigger_retraining": false}'
    parsed = _extract_final_json(text)
    assert parsed is not None
    assert parsed["diagnosis"] == "covariate shift"


def test_extracts_json_embedded_in_prose() -> None:
    text = (
        "Based on my investigation, here is my verdict:\n"
        '{"diagnosis": "drift in monthly_spend", "trigger_retraining": true}\n'
        "Let me know if you need more detail."
    )
    parsed = _extract_final_json(text)
    assert parsed is not None
    assert parsed["trigger_retraining"] is True


def test_prefers_verdict_over_nested_objects() -> None:
    text = (
        '{"action": "retrain", "priority": "high"} '
        '{"diagnosis": "concept drift", "recommendations": '
        '[{"action": "retrain", "priority": "high", "reason": "f1 drop"}], '
        '"trigger_retraining": true}'
    )
    parsed = _extract_final_json(text)
    assert parsed is not None
    assert "diagnosis" in parsed


def test_returns_none_when_no_json() -> None:
    assert _extract_final_json("no structured output here") is None
    assert _extract_final_json("{broken json") is None
