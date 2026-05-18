"""CLI: train the baseline churn model and log it to MLflow."""
from mlmonitor.models.train import train_baseline


def main() -> None:
    result = train_baseline()
    print("Baseline training complete.")
    print(f"  MLflow run:    {result.mlflow_run_id}")
    print(f"  Model path:    {result.model_path}")
    print(f"  Reference at:  {result.reference_path}")
    print(f"  Metrics:       {result.metrics}")


if __name__ == "__main__":
    main()
