# Benchmark vs Evidently

Same simulated batches, two detectors. We hand-rolled PSI/KS to *understand* the math;
this is the deliberate-scoping comparison against the standard library.

| Scenario | Ours: PSI max | Ours: drift? | Evidently: #cols drifted | Evidently: drift? |
|----------|:---:|:---:|:---:|:---:|
| none @ 0.0 | 0.01 | no | 0 | no |
| covariate @ 0.5 | 0.388 | yes | 4 | yes |
| covariate @ 1.0 | 1.452 | yes | 4 | yes |
| concept @ 1.0 | 0.01 | no | 0 | no |
| both @ 1.0 | 1.452 | yes | 4 | yes |

**Where each wins / the scoping choice:**
- Evidently gives a polished per-column report and a large metric catalogue out of the box.
- Ours adds what neither library does here: an **LLM agent that diagnoses** the drift in plain English and a **gated, audited retrain trigger** — and a label-free prediction-PSI + label-latency model (NannyML-style performance estimation territory).
- Verdict: for pure column drift, use Evidently. The value of this project is the **agentic decision layer and the safety gate** on top of correct, understood detectors.