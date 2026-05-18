"""CLI: run one drift check + (optionally) the LangChain diagnostic agent."""
import argparse
import json

from mlmonitor.monitor import diagnose_with_agent, run_drift_check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--no-agent", action="store_true", help="skip the LangChain agent")
    parser.add_argument(
        "--force-agent",
        action="store_true",
        help="run the agent even when status is ok",
    )
    args = parser.parse_args()

    report = run_drift_check(window_hours=args.window_hours)
    print("Drift check:")
    print(
        json.dumps(
            {
                "status": report["status"],
                "psi_max": report["psi_max"],
                "psi_mean": report["psi_mean"],
                "perf_f1": report.get("perf_f1"),
                "perf_drop": report.get("perf_drop"),
                "drifted_features": report.get("drifted_features"),
                "n_samples": report.get("n_samples"),
                "mlflow_run_id": report.get("mlflow_run_id"),
            },
            indent=2,
            default=str,
        )
    )

    if args.no_agent:
        return
    if report["status"] == "ok" and not args.force_agent:
        print("\nStatus ok — skipping agent (pass --force-agent to override).")
        return

    print("\nRunning diagnostic agent...")
    verdict = diagnose_with_agent(report)
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
