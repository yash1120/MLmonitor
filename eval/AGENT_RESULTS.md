# Agent Retrain-Gate Decision Eval

The both-conditions retrain rule is enforced by a **deterministic gate** (`_gate_retraining`), not by trusting the LLM. This eval feeds crafted drift reports through the gate and checks every decision against ground truth.

- Scenarios: **8**
- **Accuracy: 100%**  ·  false-trigger rate: 0%  ·  missed-trigger rate: 0%
- Thresholds in force: psi_alert=0.25, perf_drop_alert=0.05

| Scenario | Should retrain? | Gate allowed? | Correct |
|----------|:---:|:---:|:---:|
| both conditions | yes | yes | ✅ |
| at threshold | yes | yes | ✅ |
| data drift only | no | no | ✅ |
| perf drop only | no | no | ✅ |
| neither | no | no | ✅ |
| just below both | no | no | ✅ |
| both but disabled | no | no | ✅ |
| both but rate-limited | no | no | ✅ |

A perfect diagonal means the safety invariant cannot be bypassed by a hallucinated or prompt-injected LLM trigger — the gate re-checks psi_max, f1_drop, the global kill-switch, and the rate limit before any dispatch.

Regenerate with: `python scripts/evaluate_agent.py`