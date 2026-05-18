"""CLI: seed the production log with simulated batches (clean + drifted) for demos."""
import argparse

from mlmonitor.simulator.production_stream import (
    append_to_production_log,
    generate_production_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=int, default=3000, help="rows without drift")
    parser.add_argument("--drifted", type=int, default=1500, help="rows with drift")
    parser.add_argument(
        "--mode",
        choices=["covariate", "concept", "both"],
        default="both",
        help="drift type to inject for the drifted batch",
    )
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.clean > 0:
        clean = generate_production_batch(n=args.clean, drift_mode="none", seed=args.seed)
        append_to_production_log(clean)
        print(f"Appended {len(clean)} clean rows.")

    if args.drifted > 0:
        drifted = generate_production_batch(
            n=args.drifted, drift_mode=args.mode, intensity=args.intensity, seed=args.seed + 1
        )
        append_to_production_log(drifted)
        print(f"Appended {len(drifted)} drifted ({args.mode}, intensity={args.intensity}) rows.")


if __name__ == "__main__":
    main()
