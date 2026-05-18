# Contributing to Agentic MLOps Monitor

Thanks for taking the time to look at this project. The goal of this codebase is to be a small, readable, end-to-end example of an **agentic MLOps drift monitor** — not a finished platform, but a working spine you can put weight on. Contributions that keep it readable and add real production value are very welcome.

If you're not sure whether something is in scope, open an issue first and we'll figure it out together.

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Project layout](#project-layout)
- [Extension points (start here)](#extension-points-start-here)
- [Coding conventions](#coding-conventions)
- [Commit + PR conventions](#commit--pr-conventions)
- [Good first issues](#good-first-issues)
- [Code of conduct](#code-of-conduct)

---

## Ways to contribute

- **Bug reports** — open an issue with a reproducible example. Include the drift check report JSON if relevant.
- **Documentation** — anything that makes the system easier to understand or run is in scope, including diagrams, demos, and clarifying the math behind PSI/KS.
- **New drift detectors** — Jensen–Shannon divergence, Wasserstein distance, Page–Hinkley, ADWIN, etc.
- **New dataset adapters** — currently only synthetic churn data is wired up. Adapters for Kaggle credit-card-fraud, telco-churn, NYC taxi, etc. are welcome.
- **New agent tools** — anything that helps the LangChain agent investigate more accurately (e.g. correlation analysis, feature-attribution drift, recent-deploy timeline).
- **Integrations** — Slack/Teams notifiers, Feature Store adapters (Feast), feature-attribution (SHAP) drift, champion/challenger evaluation.
- **Observability / UI** — Streamlit/HTML dashboards, Grafana exporters, OpenTelemetry traces.

If you have an idea that doesn't fit these buckets, open a feature request issue and we'll talk it through.

---

## Development setup

You'll need Python 3.12.

```powershell
# Windows / PowerShell
git clone https://github.com/yash1120/MLmonitor.git
cd MLmonitor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env and set GROQ_API_KEY (free at https://console.groq.com)
```

```bash
# macOS / Linux
git clone https://github.com/yash1120/MLmonitor.git
cd MLmonitor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GROQ_API_KEY (free at https://console.groq.com)
```

Verify the install:

```bash
export PYTHONPATH=src           # PowerShell: $env:PYTHONPATH = "src"
python scripts/train_baseline.py
python scripts/seed_production.py --clean 2000 --drifted 1500 --mode both
python scripts/run_monitor.py --window-hours 168 --no-agent
```

You should see `status: alert` and a non-zero PSI/F1-drop.

---

## Running tests

```bash
export PYTHONPATH=src           # PowerShell: $env:PYTHONPATH = "src"
pytest tests -q
```

All four drift unit tests must pass. If you add a feature, add a test. If you fix a bug, write a regression test that fails on `main` and passes on your branch.

CI runs `pytest` + a Docker build on every push and PR — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Project layout

```
src/mlmonitor/
  config.py              # pydantic-settings, reads .env
  main.py                # FastAPI app + routes
  monitor.py             # end-to-end orchestration: drift → agent → persist
  models/train.py        # baseline trainer + reference data generator
  drift/data_drift.py    # PSI + KS two-sample test
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

Keep modules narrow and focused. New top-level concerns get a new sub-package.

---

## Extension points (start here)

This is the section to read carefully if you want your PR merged.

### Adding a new drift detector

1. Add the function in [`src/mlmonitor/drift/data_drift.py`](src/mlmonitor/drift/data_drift.py) (for input/covariate drift) or [`src/mlmonitor/drift/concept_drift.py`](src/mlmonitor/drift/concept_drift.py) (for performance/prediction drift).
2. Surface the result in the `report` dict produced by `run_drift_check` in [`src/mlmonitor/monitor.py`](src/mlmonitor/monitor.py).
3. Add a test in [`tests/`](tests/) covering the identical-distributions and shifted-distributions cases (see existing PSI tests as a template).
4. If the detector should affect the `status` classification, update `_classify_status` in [`monitor.py`](src/mlmonitor/monitor.py).

### Adding a new agent tool

1. Define the function in [`src/mlmonitor/agent/monitor_agent.py`](src/mlmonitor/agent/monitor_agent.py) and decorate with `@tool`.
2. Add it to the `TOOLS` list.
3. Mention it in the system prompt in [`src/mlmonitor/agent/prompts.py`](src/mlmonitor/agent/prompts.py) so the agent knows when to use it.
4. Tools should be **pure reads** by default. If a tool has side effects (like `trigger_retraining`), document the preconditions and guard with thresholds.

### Adding a new dataset adapter

1. Create `src/mlmonitor/data/<dataset_name>.py` with `load_reference()` and `load_production_window()` functions.
2. Update the config in [`src/mlmonitor/config.py`](src/mlmonitor/config.py) to allow selecting the adapter via env var.
3. The adapter must produce a DataFrame with the same schema the trained model expects — feature columns + target column. If the schema differs from the synthetic data, retrain `baseline_model.joblib`.

### Adding a new endpoint

1. Add the route in [`src/mlmonitor/main.py`](src/mlmonitor/main.py).
2. Use Pydantic models for request/response — don't pass raw dicts.
3. Update the `endpoints` list in the root handler.
4. Add the endpoint to the README's "Quick start" cURL examples.

### Adding a new notifier (Slack / Teams / email)

1. Create `src/mlmonitor/notify/<channel>.py` with a `notify(verdict: dict) -> None` function.
2. Wire it into `diagnose_with_agent` in [`monitor.py`](src/mlmonitor/monitor.py) — call after `save_agent_report`.
3. Read credentials from `config.settings`, not env vars directly.
4. Fail soft: if the notifier fails, log and continue. Drift monitoring must never break because of a downstream notification.

---

## Coding conventions

- Python 3.12. Type hints required on public functions.
- Standard library + the deps in `requirements.txt` only. Open an issue before adding a new dep.
- Format with `ruff format` (4-space indent, 100-char line length). We don't currently enforce this in CI but PRs that respect it are easier to review.
- Imports: stdlib → third-party → first-party, separated by blank lines.
- No emojis in code or commit messages.
- No comments that just restate the code. Comments explain *why*, not *what*.
- Tests live in `tests/`. Use plain `pytest` functions, no class hierarchies.
- Keep functions short. If a function is doing more than one thing, split it.

---

## Commit + PR conventions

**Commits:**
- Imperative mood: "add Jensen-Shannon drift detector", not "added" or "adds".
- One logical change per commit. Rebase to clean up before opening a PR.
- Reference the issue: `closes #12` or `refs #12` in the body.

**Pull requests:**
- Fill in the PR template. The checklist is short on purpose.
- Tests must pass locally and in CI.
- If you change a public function signature, search-replace all call sites in the same PR.
- If you change behaviour, update the README and/or [STORY.md](STORY.md) so the docs stay honest.
- Small PRs land faster than large ones. If you're touching more than ~300 lines, consider splitting.

---

## Good first issues

If you want to start small, here are concrete starter ideas — each maps to 1–3 hours of focused work:

- **Add Jensen–Shannon divergence as an alternative to PSI.** Same shape as `population_stability_index`, lives in `data_drift.py`, gets its own test.
- **Add a `/monitor/feature/{name}` endpoint** that returns the ref vs prod stats for a single feature, so dashboards can do drill-downs.
- **Add a `--seed-realistic` flag to `scripts/seed_production.py`** that simulates a slow gradual drift (linearly increasing intensity over the window) instead of a step change.
- **Add a Streamlit page** in `dashboards/streamlit_app.py` that plots `psi_max` and `perf_f1` over time from the SQLite store.
- **Write a benchmark script** in `scripts/benchmark_drift.py` that times PSI/KS over varying window sizes — gives a feel for what production load this can handle.
- **Add a `tests/test_agent.py`** that mocks Groq and verifies the agent calls at least one investigation tool before producing a verdict.
- **Document the agent's reasoning loop** with a sequence diagram in [STORY.md](STORY.md).

Open an issue if you want to claim one — saves duplication.

---

## Code of conduct

By participating in this project you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Be kind, be specific, assume good faith.

---

## Maintainer

[Yash Goyal](https://github.com/yash1120) — ML engineer, Sydney. Reach out via GitHub issues for project questions, or `yashgoyal1120 [at] gmail.com` for anything security-sensitive (see [SECURITY.md](SECURITY.md)).
