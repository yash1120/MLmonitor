# Sample agent verdict

A real run of the LangGraph diagnostic agent on a `both`-drift batch. Committed so the agent's output is visible without an API key.

## Drift report (input to the agent)
```json
{
  "status": "alert",
  "psi_max": 2.028,
  "perf_f1": 0.3209,
  "perf_drop": 0.3724,
  "prediction_drift_psi": 0.1438,
  "drift_sustained": true,
  "drifted_features": [
    "monthly_spend",
    "credit_utilisation",
    "logins_last_30d",
    "support_tickets"
  ],
  "label_coverage": 1.0
}
```

## Agent verdict (output)
```json
{
  "diagnosis": "The model is experiencing significant data drift, with a PSI max of 2.0280 and a mean of 0.7120. The features 'monthly_spend', 'credit_utilisation', 'logins_last_30d', and 'support_tickets' are drifting, with 'credit_utilisation' having the highest PSI value of 2.0279794870212617. The production F1 score has dropped by 0.3724262129434964 compared to the baseline, indicating performance degradation. The drift is sustained across multiple temporal slices and segments.",
  "recommendations": [
    {
      "action": "Collect more data to verify the drift",
      "priority": "medium",
      "reason": "The drift may be due to a temporary change in the data distribution"
    },
    {
      "action": "Update the model to account for the drift",
      "priority": "high",
      "reason": "The drift is sustained and significant, and the model's performance is degrading"
    },
    {
      "action": "Monitor the drift and performance metrics closely",
      "priority": "low",
      "reason": "The drift may continue to evolve, and the model's performance may change over time"
    }
  ],
  "trigger_retraining": true,
  "actually_triggered": false
}
```

> `trigger_retraining` is the agent's *recommendation*; `actually_triggered` reflects the **deterministic gate**, which independently re-checks both thresholds, the kill-switch, and the rate limit before any real dispatch.