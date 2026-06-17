# Agentic MLOps Monitor

An agentic AI system that continuously monitors a production ML model, detects **data drift** and **concept drift**, and uses an **LLM agent (LangChain + LangGraph)** to produce natural-language diagnostic reports and remediation recommendations. When — and only when — drift criteria are met, a **deterministic safety gate** lets the agent dispatch a **GitHub Actions** retraining pipeline, whose challenger model is **promoted only if it beats the champion**. All experiments and model versions are tracked with **MLflow**, with the same code path deployable to **Azure ML**.

**Stack:** Python · FastAPI · scikit-learn · MLflow · LangChain · LangGraph · Groq (free LLM API) · pyarrow · Prometheus · Azure ML · Docker · GitHub Actions

> **Why this exists and how it actually works** → read [STORY.md](STORY.md) for the narrative version.
> **See the agent in action** → [examples/sample_verdict.md](examples/sample_verdict.md) is a real captured agent verdict (no API key needed to read it).

---

## Highlights (what makes this more than a demo)

- **Measured, not asserted** — drift detectors benchmarked over 30 batches/scenario with a **0% false-positive rate** and a baseline model of **F1 0.69 / ROC-AUC 0.84** ([eval/RESULTS.md](eval/RESULTS.md)), and validated **column-for-column against Evidently** ([eval/BENCHMARK.md](eval/BENCHMARK.md)).
- **Safety enforced in code** — the "retrain only when both conditions hold" rule is a **deterministic gate** (kill-switch + rate-limit + audit trail), not a prompt suggestion. Proven by a decision eval at **100% accuracy** ([eval/AGENT_RESULTS.md](eval/AGENT_RESULTS.md)).
- **Production-realistic** — models **label latency** (ground truth lags weeks), so concept drift is scored only on the labelled slice while a **label-free prediction-PSI** is the online early-warning signal.
- **Closed loop** — champion/challenger promotion gate; a free hourly GitHub Actions schedule; Slack/Discord webhook alerts; Prometheus metrics + an HTML dashboard.

---

## Architecture

![Agentic MLOps Monitor architecture](diagrams/architecture.png)

Production + reference data feed the drift engine (PSI, KS, F1/AUC, prediction-PSI); a LangGraph agent investigates the results and emits a verdict; a deterministic gate decides whether to dispatch GitHub Actions retraining; the challenger is promoted only if it beats the champion. FastAPI serves it; MLflow and SQLite track everything.

### How the agent decides

![How the monitoring agent decides](diagrams/agent_loop.png)

The agent uses tools to investigate *whether the drift matters* and distinguishes a transient blip from a sustained ramp. It **recommends** retraining; a deterministic gate **disposes** — re-checking that **both** data drift (`PSI ≥ 0.25`) **and** performance degradation (`F1 drop ≥ 0.05`) hold, that retraining is enabled, and that the rate limit hasn't tripped, before any real dispatch.

---

## Quick start (local)

```powershell
# 1. Install (editable, with dev tooling)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. Configure
copy .env.example .env
# Edit .env and set GROQ_API_KEY (free at https://console.groq.com)

# 3. Train baseline → seed production → run a drift check + agent pass
mlmonitor-train
mlmonitor-seed --clean 3000 --drifted 1500 --mode both
mlmonitor-monitor --window-hours 168 --force-agent

# 4. Or run the FastAPI service
uvicorn mlmonitor.main:app --reload
# → http://localhost:8000/docs   ·   dashboard at /dashboard   ·   Prometheus at /metrics
```

(`pip install -e .` puts the package on the path — no more `PYTHONPATH=src`. The `scripts/*.py` files still work too.)

## Quick start (Docker)

```bash
docker compose up --build
# API: http://localhost:8000/docs   ·   MLflow UI: http://localhost:5000
```

---

## What the agent actually does

When `/monitor/check` runs and status is `warn`/`alert`, the LangGraph ReAct agent:

1. `get_top_drifting_features` / `inspect_feature` — identify and examine the shifted features.
2. `get_performance_metrics` — confirm whether F1 actually dropped (on the labelled slice).
3. `get_drift_attribution` — read within-window temporal slices and per-segment drift to judge **transient vs sustained**.
4. `get_recent_drift_history` — cross-check the longer trend.
5. `trigger_retraining` — *requests* a retrain; the **deterministic gate** allows it only if both thresholds breach, retraining is enabled, and it isn't rate-limited.
6. Returns a **typed verdict** via structured output (`with_structured_output`), not brace-scraped free text.

---

## Drift detection methods

| Method | Detects | Role |
| ------ | ------- | ---- |
| **PSI** (bounded; production clipped into reference support) | Covariate drift | **Alert driver** (effect size) |
| KS two-sample test | Distributional difference | **Diagnostic only** — p-values collapse at large N, so KS does *not* gate alerts (reported with a ≥0.10 statistic floor) |
| Categorical / NaN-aware PSI | Mix shifts in discrete features, rising missingness | Alert driver for low-cardinality columns |
| F1 / AUC vs baseline (labelled slice) | Concept drift (confirmed) | Alert driver once labels arrive |
| Prediction-distribution PSI | Concept/covariate drift, **label-free** | Online early-warning alert driver |
| Temporal + per-segment attribution | *Where* and *whether sustained* | Agent reasoning |

Configurable thresholds & controls (`.env`):

| Setting | Default | Purpose |
|---------|---------|---------|
| `PSI_WARN_THRESHOLD` / `PSI_ALERT_THRESHOLD` | 0.10 / 0.25 | data-drift bands |
| `PRED_DRIFT_WARN_THRESHOLD` / `PRED_DRIFT_ALERT_THRESHOLD` | 0.10 / 0.25 | label-free concept-drift bands |
| `PERF_DROP_ALERT` | 0.05 | F1-drop concept-drift band |
| `MIN_SAMPLES_FOR_CHECK` | 100 | below this → `insufficient_data` |
| `LABEL_DELAY_DAYS` | 0 | simulate ground-truth latency |
| `RETRAIN_ENABLED` / `RETRAIN_DRY_RUN` / `RETRAIN_MIN_INTERVAL_MINUTES` | true / false / 60 | retrain gate controls |
| `MONITOR_API_KEY` | "" (open) | shared-secret auth on state-changing endpoints |
| `ALERT_WEBHOOK_URL` | "" (off) | Slack/Discord alert webhook |

---

## Measured detection performance

End-to-end over 30 simulated batches/scenario (same code path the live monitor runs) — [eval/RESULTS.md](eval/RESULTS.md):

- Baseline model: **F1 0.69 / ROC-AUC 0.84** (tuned decision threshold)
- **False-positive rate on clean traffic: 0%**

| Scenario | Data-drift detection | Concept-drift detection | Any alert |
|----------|:---:|:---:|:---:|
| Clean traffic | — | — | **0%** (false alarms) |
| Covariate (moderate 0.5×) | **100%** | 33% | **100%** |
| Covariate (strong 1.0×) | **100%** | 100% | **100%** |
| Concept only (strong) | 0% *(features unchanged by design)* | **77%** | **77%** |
| Both (strong) | **100%** | **100%** | **100%** |

The concept-only row is the point: labels shift while feature distributions stay identical (PSI ≈ 0.008), so PSI/KS see nothing — the F1-drop detector still catches it. That's why both detectors exist. Regenerate: `python scripts/evaluate_drift.py`.

---

## Safety: the retrain gate

The headline "only retrain when both conditions hold" is enforced by `_gate_retraining`, **not** by trusting the LLM. `scripts/evaluate_agent.py` feeds crafted reports (drift-only, perf-only, both, neither, borderline, disabled, rate-limited) through the gate and asserts every decision — **100% accuracy, 0% false triggers** ([eval/AGENT_RESULTS.md](eval/AGENT_RESULTS.md)). Every attempt (allowed, blocked, dry-run) is written to an audit trail at `GET /monitor/audits`.

## Label latency & online concept drift

In reality a churn label lands weeks after the prediction. With `LABEL_DELAY_DAYS > 0`, `run_drift_check` scores F1 only on the slice whose labels would be available by now (surfacing `label_coverage`), and leans on the **label-free prediction-PSI** as the online signal — reframing the F1-drop path as delayed *confirmation*.

## Champion / challenger promotion

Retraining trains a challenger and promotes it over the incumbent **only if it scores at least as well on the latest labelled production window** (`src/mlmonitor/models/promotion.py`, wired into `retrain.yml`); the winner is tagged `@champion` in the MLflow registry. The loop is closed and entirely free/local.

---

## Observability

- **`GET /dashboard`** — live HTML dashboard (status, PSI sparkline, recent checks, agent verdicts). All values HTML-escaped at render.
- **`GET /metrics`** — Prometheus: predictions, drift checks by status, agent outcomes, retrain dispatches, last PSI/F1 gauges, check-duration histogram.
- **`GET /monitor/audits`** — retrain-dispatch audit trail.
- Structured logs via `LOG_LEVEL`.

## Security

- **Auth**: `MONITOR_API_KEY` gates `/train`, `/predict`, `/simulate`, `/monitor/*` (open on localhost when unset).
- **No autonomous footguns**: the retrain gate re-checks thresholds in code; injection-laden agent reasons are sanitised before dispatch.
- **No injection sinks**: dashboard output is HTML-escaped (no stored XSS); `retrain.yml` passes the agent reason as an env var, never interpolated into a shell (no Actions script injection).
- **No info leaks**: GitHub/exception detail is logged server-side, not returned to clients.

---

## CI/CD

- **`ci.yml`** — `ruff` lint · `pytest` + 70% coverage gate · drift evaluation **and** agent-gate eval (uploaded as artifacts) · Docker build. Uses `pip install -e .[dev]`.
- **`monitor.yml`** — free **hourly** scheduled drift check; publishes the result to the job summary (the free equivalent of `azure/schedule.yml`).
- **`retrain.yml`** — `workflow_dispatch` from the agent: trains a challenger, **promotes only if it wins**, runs tests; the reason is passed safely as an env var.

---

## Azure ML deployment

The `azure/` directory contains job/environment/endpoint/deployment/schedule manifests for a managed-cloud deployment. They are reference manifests (the portfolio build runs on free infra); point `MLFLOW_TRACKING_URI` at an Azure ML workspace to surface runs in Azure ML Studio.

---

## Project layout

```
src/mlmonitor/
  config.py              # pydantic-settings + threshold/invariant validators
  main.py                # FastAPI app (+ auth dependency)
  monitor.py             # orchestration: drift → attribution → agent → gate → persist → alert
  cli.py                 # console entry points (mlmonitor-train/-monitor/-seed)
  notifications.py       # free Slack/Discord webhook alerts
  observability.py       # Prometheus metrics
  dashboard.py           # single-file HTML dashboard (escaped)
  models/train.py        # trainer, tuned threshold, bundle (baseline F1, ref-hash, cached probs)
  models/promotion.py    # champion/challenger promotion gate
  drift/data_drift.py    # bounded PSI + categorical/NaN + KS (diagnostic)
  drift/concept_drift.py # F1 drop + prediction-distribution PSI
  drift/attribution.py   # temporal (velocity) + per-segment drift
  agent/monitor_agent.py # LangGraph ReAct agent, contextvar state, deterministic gate, structured output
  mlflow_utils/tracking.py  # logging + champion alias
  simulator/production_stream.py  # partitioned append-only log (pyarrow pushdown)
  storage/db.py          # SQLite (WAL) + drift checks, agent reports, retrain audits
scripts/                 # train, seed, monitor, promote, evaluate_drift, evaluate_agent, benchmark_evidently
eval/                    # RESULTS.md, AGENT_RESULTS.md, BENCHMARK.md
examples/                # captured real agent verdict
.github/workflows/       # ci · monitor (cron) · retrain
tests/                   # 57 tests, all offline — no API key needed
```

---

## Tests

```powershell
pytest                   # 57 tests, all offline
ruff check src tests scripts
```

Covers drift statistics (bounded PSI, categorical/NaN, KS demotion), the simulator, the monitor flow, label latency, the retrain gate + audit, champion/challenger promotion, config validation, notifications, attribution, the API flow, agent parsing, and storage.
