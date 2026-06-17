"""Decision eval for the retrain GATE — proves the both-conditions invariant is enforced
in CODE, not just asked of the LLM in a prompt.

For each crafted drift report we know whether retraining SHOULD fire (psi_max >= alert
AND f1_drop >= alert AND enabled AND not rate-limited). We run the deterministic gate and
build a confusion matrix. A perfect diagonal is the artifact: the safety-critical rule
holds for every scenario, independent of any stochastic model output.

Usage:  python scripts/evaluate_agent.py     (writes eval/AGENT_RESULTS.md)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mlmonitor.config import settings

# Isolate the audit DB so rate-limit history starts empty and CI is reproducible.
_tmp = Path(tempfile.mkdtemp()) / "agent_eval.db"
settings.metrics_db_url = f"sqlite:///{_tmp.as_posix()}"

from mlmonitor.agent.monitor_agent import _gate_retraining  # noqa: E402
from mlmonitor.storage.db import save_retrain_audit  # noqa: E402

ALERT = settings.psi_alert_threshold
DROP = settings.perf_drop_alert


def _report(psi: float, drop: float) -> dict:
    return {"psi_max": psi, "perf_drop": drop}


SCENARIOS = [
    # (name, report, setup, expected_allowed)
    ("both conditions", _report(0.40, 0.12), None, True),
    ("at threshold", _report(ALERT, DROP), None, True),
    ("data drift only", _report(0.40, 0.00), None, False),
    ("perf drop only", _report(0.02, 0.12), None, False),
    ("neither", _report(0.02, 0.00), None, False),
    ("just below both", _report(ALERT - 0.001, DROP - 0.001), None, False),
    ("both but disabled", _report(0.40, 0.12), "disable", False),
    ("both but rate-limited", _report(0.40, 0.12), "recent_dispatch", False),
]


def main() -> None:
    rows = []
    tp = tn = fp = fn = 0
    for name, report, setup, expected in SCENARIOS:
        settings.retrain_enabled = True
        if setup == "disable":
            settings.retrain_enabled = False
        elif setup == "recent_dispatch":
            save_retrain_audit("dispatched", "prior", 0.4, 0.12, "seed for rate-limit test")

        allowed, msg = _gate_retraining(report)

        # reset rate-limit history after the dedicated test so it doesn't bleed across rows
        if setup == "recent_dispatch":
            settings.metrics_db_url = f"sqlite:///{(_tmp.parent / 'reset.db').as_posix()}"

        correct = allowed == expected
        tp += allowed and expected
        tn += (not allowed) and (not expected)
        fp += allowed and (not expected)
        fn += (not allowed) and expected
        rows.append((name, expected, allowed, correct, msg))

    settings.retrain_enabled = True
    total = len(SCENARIOS)
    accuracy = (tp + tn) / total

    out_dir = Path("eval")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "agent_results.json").write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "false_trigger_rate": fp / total,
                "missed_trigger_rate": fn / total,
                "scenarios": [
                    {"name": n, "expected_allowed": e, "gate_allowed": a, "correct": c}
                    for n, e, a, c, _ in rows
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Agent Retrain-Gate Decision Eval",
        "",
        "The both-conditions retrain rule is enforced by a **deterministic gate** "
        "(`_gate_retraining`), not by trusting the LLM. This eval feeds crafted drift "
        "reports through the gate and checks every decision against ground truth.",
        "",
        f"- Scenarios: **{total}**",
        f"- **Accuracy: {accuracy:.0%}**  ·  false-trigger rate: {fp / total:.0%}  ·  "
        f"missed-trigger rate: {fn / total:.0%}",
        f"- Thresholds in force: psi_alert={ALERT}, perf_drop_alert={DROP}",
        "",
        "| Scenario | Should retrain? | Gate allowed? | Correct |",
        "|----------|:---:|:---:|:---:|",
    ]
    for name, expected, allowed, correct, _msg in rows:
        lines.append(
            f"| {name} | {'yes' if expected else 'no'} | "
            f"{'yes' if allowed else 'no'} | {'✅' if correct else '❌'} |"
        )
    lines += [
        "",
        "A perfect diagonal means the safety invariant cannot be bypassed by a hallucinated "
        "or prompt-injected LLM trigger — the gate re-checks psi_max, f1_drop, the global "
        "kill-switch, and the rate limit before any dispatch.",
        "",
        "Regenerate with: `python scripts/evaluate_agent.py`",
    ]
    (out_dir / "AGENT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Gate decision accuracy: {accuracy:.0%} over {total} scenarios "
          f"(false-trigger {fp / total:.0%}, missed {fn / total:.0%})")
    for name, expected, allowed, correct, msg in rows:
        flag = "OK " if correct else "ERR"
        print(f"  [{flag}] {name:<22} expected={expected!s:<5} allowed={allowed!s:<5} {msg[:60]}")
    if accuracy < 1.0:
        raise SystemExit("Gate decision eval failed — invariant not enforced for all scenarios.")


if __name__ == "__main__":
    main()
