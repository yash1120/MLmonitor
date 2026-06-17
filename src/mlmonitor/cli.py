"""Console entry points (installed via pyproject [project.scripts]).

These are thin wrappers over the library so `mlmonitor-train` / `mlmonitor-monitor` /
`mlmonitor-seed` work after `pip install -e .`. The scripts/ files remain as
`python scripts/<name>.py` equivalents for those who prefer them.
"""
from __future__ import annotations

import argparse
import json


def train() -> None:
    from mlmonitor.models.train import train_baseline

    result = train_baseline()
    print("Baseline training complete.")
    print(f"  MLflow run:   {result.mlflow_run_id}")
    print(f"  Model path:   {result.model_path}")
    print(f"  Reference at: {result.reference_path}")
    print(f"  Metrics:      {result.metrics}")


def seed() -> None:
    from mlmonitor.simulator.production_stream import (
        append_to_production_log,
        generate_production_batch,
    )

    parser = argparse.ArgumentParser(description="Seed the production log for demos.")
    parser.add_argument("--clean", type=int, default=3000)
    parser.add_argument("--drifted", type=int, default=1500)
    parser.add_argument("--mode", choices=["covariate", "concept", "both"], default="both")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.clean > 0:
        append_to_production_log(
            generate_production_batch(n=args.clean, drift_mode="none", seed=args.seed)
        )
        print(f"Appended {args.clean} clean rows.")
    if args.drifted > 0:
        append_to_production_log(
            generate_production_batch(
                n=args.drifted, drift_mode=args.mode, intensity=args.intensity, seed=args.seed + 1
            )
        )
        print(f"Appended {args.drifted} drifted ({args.mode}) rows.")


def monitor() -> None:
    from mlmonitor.monitor import diagnose_with_agent, run_drift_check

    parser = argparse.ArgumentParser(description="Run one drift check + optional agent diagnosis.")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--no-agent", action="store_true")
    parser.add_argument("--force-agent", action="store_true")
    args = parser.parse_args()

    report = run_drift_check(window_hours=args.window_hours)
    print(json.dumps({k: v for k, v in report.items() if k != "by_feature"}, indent=2, default=str))

    if args.no_agent or (report["status"] == "ok" and not args.force_agent):
        return
    print("\nRunning diagnostic agent...")
    print(json.dumps(diagnose_with_agent(report), indent=2, default=str))
