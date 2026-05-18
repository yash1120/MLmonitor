# Agentic MLOps Monitor

An agentic AI system that continuously monitors a production ML model, detects **data drift** and **concept drift**, and uses an **LLM agent (LangChain + LangGraph)** to produce natural-language diagnostic reports and remediation recommendations. When drift thresholds are breached, the agent dispatches a **GitHub Actions** retraining pipeline. All experiments and model versions are tracked with **MLflow**, with the same code path deployable to **Azure ML**.

**Stack:** Python · FastAPI · scikit-learn · MLflow · LangChain · LangGraph · Groq (free LLM API) · Azure ML · Docker · GitHub Actions

> **Why this exists and how it actually works** → read [STORY.md](STORY.md) for the narrative version (why ML models silently degrade, what the agent actually does, and the design choices behind it).

---

## Architecture

```
                            ┌─────────────────────────────────────────────┐
                            │              FastAPI service                │
                            │  /predict   /monitor/check   /monitor/...   │
                            └────┬───────────────┬────────────────┬───────┘
                                 │               │                │
                                 ▼               ▼                ▼
                       ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                       │  Reference   │  │  Production  │  │  MLflow      │
                       │  data + model│  │  log         │  │  tracking    │
                       │  (joblib)    │  │  (parquet)   │  │  (sqlite or  │
                       └──────┬───────┘  └──────┬───────┘  │   Azure ML)  │
                              │                 │           └──────────────┘
                              └────────┬────────┘
                                       ▼
                       ┌─────────────────────────────────────┐
                       │  Drift engine                       │
                       │  • PSI per feature                  │
                       │  • KS two-sample test               │
                       │  • Performance F1/AUC vs baseline   │
                       │  • Prediction-distribution drift    │
                       └──────────────┬──────────────────────┘
                                      ▼
                       ┌─────────────────────────────────────┐
                       │  LangGraph agent (Groq Llama 3.3)   │
                       │  Tools:                             │
                       │   - get_top_drifting_features       │
                       │   - inspect_feature                 │
                       │   - get_performance_metrics         │
                       │   - get_recent_drift_history        │
                       │   - trigger_retraining              │
                       └──────────────┬──────────────────────┘
                                      ▼
                       ┌─────────────────────────────────────┐
                       │  Diagnosis + recommendations (JSON) │
                       │  Persisted to SQLite agent_reports  │
                       └──────────────┬──────────────────────┘
                                      ▼
                       ┌─────────────────────────────────────┐
                       │  GitHub Actions: retrain.yml        │
                       │  (workflow_dispatch from agent)     │
                       │   → trains, runs tests              │
                       │   → optionally submits Azure ML job │
                       └─────────────────────────────────────┘
```

---

## Quick start (local)

```powershell
# 1. Install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure
copy .env.example .env
# Edit .env and set GROQ_API_KEY (free at https://console.groq.com)

# 3. Train the baseline model + log to MLflow
$env:PYTHONPATH = "src"
python scripts/train_baseline.py

# 4. Seed the "production" log with clean + drifted batches
python scripts/seed_production.py --clean 3000 --drifted 1500 --mode both

# 5. Run one drift-check + agent pass
python scripts/run_monitor.py --window-hours 168

# 6. Or run the FastAPI service
uvicorn mlmonitor.main:app --reload
# → http://localhost:8000/docs
```

## Quick start (Docker)

```bash
docker compose up --build
# API:        http://localhost:8000/docs
# MLflow UI:  http://localhost:5000
```

After the API is up:

```bash
curl -X POST http://localhost:8000/train/baseline
curl -X POST http://localhost:8000/simulate/batch \
  -H "content-type: application/json" \
  -d '{"n": 1500, "drift_mode": "both", "intensity": 1.2}'
curl -X POST http://localhost:8000/monitor/check \
  -H "content-type: application/json" \
  -d '{"window_hours": 168, "run_agent": true}'
```

---

## What the agent actually does

When `/monitor/check` runs and the status is `warn` or `alert`, the LangGraph ReAct agent is invoked with the drift summary. It then:

1. Calls `get_top_drifting_features` to identify which features shifted most.
2. Calls `inspect_feature` for each suspicious feature to compare ref vs prod means/stds.
3. Calls `get_performance_metrics` to confirm whether F1 has actually dropped.
4. Optionally calls `get_recent_drift_history` to check whether this is a one-off or a trend.
5. Calls `trigger_retraining` **only when both** `psi_max >= 0.25` **and** `f1_drop >= 0.05`.
6. Emits a structured JSON verdict: `{ diagnosis, recommendations[], trigger_retraining }`.

That tool-using loop is what makes it an agent rather than a one-shot LLM call.

---

## Drift detection methods

| Method                          | Detects             | Where                                              |
| ------------------------------- | ------------------- | -------------------------------------------------- |
| PSI (Population Stability Index)| Covariate drift     | `src/mlmonitor/drift/data_drift.py`               |
| KS two-sample test              | Distributional drift| `src/mlmonitor/drift/data_drift.py`               |
| F1 / AUC vs baseline            | Concept drift       | `src/mlmonitor/drift/concept_drift.py`            |
| Prediction-distribution PSI     | Label-free proxy    | `src/mlmonitor/drift/concept_drift.py`            |

Thresholds (configurable via `.env`):
- `PSI_WARN_THRESHOLD` = 0.10
- `PSI_ALERT_THRESHOLD` = 0.25
- `PERF_DROP_ALERT` = 0.05

---

## Azure ML deployment

The `azure/` directory contains job, environment, endpoint, deployment, and schedule manifests. With an Azure ML workspace:

```bash
az login
az account set --subscription <SUB_ID>

# 1. One-off retraining job
az ml job create -f azure/train-job.yml -g <RG> -w <WS>

# 2. Hourly scheduled drift check
az ml schedule create -f azure/schedule.yml -g <RG> -w <WS>

# 3. Online endpoint + blue deployment (FastAPI as managed inference)
az ml online-endpoint create -f azure/endpoint.yml -g <RG> -w <WS>
az ml online-deployment create -f azure/deployment.yml -g <RG> -w <WS> --all-traffic
```

When `MLFLOW_TRACKING_URI` is set to the Azure ML workspace tracking URI, all runs (training + drift checks) appear in the Azure ML studio Jobs view.

---

## CI/CD

- **`.github/workflows/ci.yml`** — runs `pytest` and a Docker build on every push/PR.
- **`.github/workflows/retrain.yml`** — `workflow_dispatch` triggered by the LangChain agent. Trains the baseline, runs tests, and (if Azure credentials are present) submits an Azure ML training job.

The agent dispatches the workflow via the GitHub REST API using a fine-grained PAT (set `GITHUB_OWNER`, `GITHUB_REPO`, `GITHUB_TOKEN` in `.env`).

---

## Project layout

```
src/mlmonitor/
  config.py              # pydantic-settings, reads .env
  main.py                # FastAPI app
  monitor.py             # end-to-end orchestration (drift → agent → persist)
  models/train.py        # baseline trainer + reference data generator
  drift/data_drift.py    # PSI + KS
  drift/concept_drift.py # F1 drop + prediction-distribution PSI
  agent/monitor_agent.py # LangGraph ReAct agent + tools
  agent/prompts.py       # system + initial prompts
  mlflow_utils/tracking.py
  simulator/production_stream.py
  storage/db.py          # SQLite via SQLAlchemy
scripts/                 # CLI entry points
azure/                   # Azure ML manifests
.github/workflows/       # CI + retrain
tests/                   # PSI / drift unit tests
```

---

## Tests

```powershell
$env:PYTHONPATH = "src"
pytest tests -q
```
