# The story behind Agentic MLOps Monitor

## Why this exists

A machine-learning model is the only piece of production software that can be wrong without throwing an exception.

A web service that breaks returns a 500. A database query that fails raises an error. A model that has quietly become useless keeps returning predictions — confidently, at full throughput, with the same latency it always had. Nothing fires. Nothing pages.

Then weeks later, someone in the business notices the churn predictions stopped matching reality. The team digs in and finds that three months ago a payments provider changed how it reports `monthly_spend`, and the model has been training-set-blind to the new distribution ever since. The fix is one retraining run. The damage is the customers who were mis-scored for ninety days.

This is the gap **Agentic MLOps Monitor** is built to close. It assumes the model *will* drift, the world *will* change, and someone needs to be watching the watchers — automatically, continuously, and with enough context to actually do something about it.

## What it does

It sits between your trained model and your production traffic and asks, every hour: *is the model still operating in the world it was trained for?*

When it sees evidence that the answer is no, it doesn't just send a Slack alert and hope someone reads it. It runs a **LangChain agent** over the drift report. The agent has tools — it can pull the top drifting features, inspect them one by one, check whether F1 has actually dropped or whether the inputs just shifted harmlessly, look at recent drift history to see whether this is a one-off blip or a trend. It then writes a short diagnostic report in plain English and, if the evidence is strong enough, dispatches the GitHub Actions retraining workflow itself.

The artefact a human eventually reads isn't `psi_max=0.42`. It's: *"Covariate drift on `monthly_spend` and `credit_utilisation` (PSI 0.42 and 0.31). Production F1 has dropped from 0.89 to 0.81 over the last 48 hours. Retraining dispatched. Recommend investigating upstream `monthly_spend` reporting changes — variance has nearly doubled."*

## How it does it

The flow is four moving parts, each doing one thing well.

**1. Drift detection — statistics, not vibes.**
Every check pulls the most recent production window and compares it against the reference distribution captured at training time. For each feature it computes the **Population Stability Index** (the industry standard — flags shifts in distribution shape) and the **two-sample Kolmogorov–Smirnov test** (sensitive to subtle distributional changes the PSI bins might miss). For the model itself, it scores any production records where the label is eventually known, computes F1 and AUC, and compares them against the baseline from training. As a label-free proxy, it also tracks the PSI of the *predicted probability* distribution — a useful early-warning even when ground truth lags by weeks.

**2. The agent — investigation, not just summary.**
The drift numbers go to a **LangGraph ReAct agent** built on Groq's Llama 3.3 70B (free tier). It's given a system prompt that explains the job and a set of tools — `get_top_drifting_features`, `inspect_feature`, `get_performance_metrics`, `get_recent_drift_history`, `trigger_retraining`. The agent decides which tools to call and in what order. A small drift in one cosmetic feature gets a one-paragraph note. A large drift in three correlated features, with F1 down 8 points, gets a full diagnosis and a retraining dispatch. The threshold for *actually* triggering retraining is conservative — both `psi_max ≥ 0.25` and `f1_drop ≥ 0.05` must hold. The agent can also choose to recommend manual review instead of acting; this is the difference between an autonomous agent and a reckless one.

**3. The retraining loop — closing the feedback cycle.**
When the agent dispatches retraining, it calls the GitHub REST API to trigger the `retrain.yml` workflow. That workflow trains a fresh baseline, runs the drift test suite as a regression check, and — if Azure credentials are configured — submits the training job to Azure ML so the run shows up in the workspace alongside historical experiments. The new model is registered with **MLflow**, versioned, and ready to deploy. The CV/CD pipeline isn't a separate process bolted on after the fact; it's woven into the monitoring loop itself.

**4. Observability — every check is an experiment.**
Every drift check is logged to MLflow with its own run, tagged with `drift_status`, and parameterised with `psi_max`, `psi_mean`, `perf_f1`, `perf_drop`. The agent's verdict — diagnosis text, recommendations, whether it triggered retraining — is persisted to SQLite alongside it. Six months from now, when someone asks "*how often did the model actually need retraining last quarter?*" the answer is a SQL query, not a guess.

## The architecture choice that matters

Most monitoring tools stop at a dashboard. The leap this project makes is treating the diagnostic step as **an agentic problem, not a templating problem**. A template can produce "drift detected on feature X." An agent can decide whether X drifting actually matters this week, given that Y also drifted last week and F1 didn't budge, and produce a recommendation grounded in the specific situation rather than a generic playbook.

That's what makes it *agentic* rather than *automated*. Automation runs the same script every time. The agent reasons about whether the script should run.

## What's been built since

Several items that were once on the "next" list are now in the codebase:

- **Champion / challenger promotion** (`models/promotion.py`) — a retrained candidate is promoted over the incumbent only if it wins on the latest labelled production window; the winner is tagged `@champion` in the MLflow registry.
- **Slack / Discord notifier** (`notifications.py`) — the verdict is pushed to an incoming webhook on warn/alert, env-gated so it no-ops when unset.
- **Label-latency model + label-free signal** — concept drift is scored only on the slice whose labels would have arrived, and the prediction-distribution PSI is wired in as the online early-warning signal.
- **A deterministic retrain gate** — the both-conditions rule is enforced in code (kill-switch + rate limit + audit trail), not just asked of the LLM, and proven by a decision eval.
- **Within-window + per-segment attribution** — so the agent can distinguish a transient blip from a sustained ramp.

## What I'd still build next

- **Real-world dataset adapter.** The synthetic generator now yields a usable model (F1 0.69 / AUC 0.84), but a swap-in adapter for a Kaggle credit-risk or IBM Telco dataset would make the demo land harder. The drift engine is already dataset-agnostic.
- **Feature-attribution drift.** PSI tells you the inputs shifted; SHAP-value drift tells you whether the *model's reasoning* shifted — the more interesting signal.
- **Multivariate / interaction drift.** A domain-classifier (reference-vs-production AUC) would catch joint-distribution shifts that per-feature PSI structurally cannot — worth adding alongside a simulator mode that actually produces one.

---

The project is small enough to read end-to-end in an hour and structured so each piece can be lifted out and dropped into a real production stack. That's the point: not a finished platform, but a working spine you could put weight on.
