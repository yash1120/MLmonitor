# Drift-Detection Evaluation

Pipeline: PSI (alert ≥ 0.25) + label-free prediction-PSI + F1-drop (alert ≥ 0.05) — the same code path the live monitor runs.

- Trials per scenario: **30**, batch size: **2000**
- Baseline model: **F1 0.693** / **ROC-AUC 0.841** (tuned decision threshold 0.440)
- **False-positive rate on clean data: 0.0%**

| Mode | Intensity | Data-drift detection | Concept-drift detection | Any alert | Mean PSI max | Mean F1 drop |
|------|-----------|---------------------|------------------------|-----------|--------------|--------------|
| none | 0.0 | 0% | 0% | 0% | 0.008 | 0.000 |
| covariate | 0.25 | 0% | 0% | 0% | 0.102 | 0.005 |
| covariate | 0.5 | 100% | 33% | 100% | 0.378 | 0.042 |
| covariate | 1.0 | 100% | 100% | 100% | 1.403 | 0.184 |
| concept | 0.5 | 0% | 0% | 0% | 0.008 | 0.008 |
| concept | 1.0 | 0% | 77% | 77% | 0.008 | 0.059 |
| both | 0.5 | 100% | 100% | 100% | 0.378 | 0.113 |
| both | 1.0 | 100% | 100% | 100% | 1.403 | 0.294 |

Notes:
- `none` row measures false positives: any alert on clean data is a false alarm.
- **PSI is bounded** (production is clipped into the reference support), so values are interpretable against the standard 0.10/0.25 bands rather than saturating on tail shifts.
- **KS is diagnostic only** — its p-value collapses toward 0 at large N, so it does NOT drive alerts; PSI (an effect-size measure) is the data-drift gate. KS is reported with a ≥0.10 statistic floor as supporting colour.
- Concept drift uses the labelled slice only (label latency is modelled); the label-free prediction-PSI is the online early-warning signal.
- Regenerate with: `python scripts/evaluate_drift.py --trials 30 --n 2000`