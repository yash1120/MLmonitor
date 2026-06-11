# Drift-Detection Evaluation

Pipeline: PSI (alert ≥ 0.25) + KS test + F1-drop (alert ≥ 0.05) — the same code path the live monitor runs.

- Trials per scenario: **30**, batch size: **2000**
- Baseline F1 (held-out): **0.572**
- **False-positive rate on clean data: 0.0%**

| Mode | Intensity | Data-drift detection | Concept-drift detection | Any alert | Mean PSI max | Mean F1 drop |
|------|-----------|---------------------|------------------------|-----------|--------------|--------------|
| none | 0.0 | 0% | 0% | 0% | 0.008 | 0.000 |
| covariate | 0.25 | 0% | 0% | 0% | 0.093 | 0.000 |
| covariate | 0.5 | 100% | 20% | 100% | 0.347 | 0.034 |
| covariate | 1.0 | 100% | 100% | 100% | 1.417 | 0.154 |
| concept | 0.5 | 0% | 10% | 10% | 0.008 | 0.023 |
| concept | 1.0 | 0% | 97% | 97% | 0.008 | 0.105 |
| both | 0.5 | 100% | 100% | 100% | 0.347 | 0.140 |
| both | 1.0 | 100% | 100% | 100% | 1.417 | 0.180 |

Notes:
- `none` row measures false positives: any alert on clean data is a false alarm.
- Covariate intensity 0.25 is a deliberately subtle shift; partial detection there is expected
  and is exactly the regime where the warn band (PSI ≥ 0.1) plus the agent's trend check earn their keep.
- Regenerate with: `PYTHONPATH=src python scripts/evaluate_drift.py --trials 30 --n 2000`