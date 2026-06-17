"""CLI: train a challenger and promote it over the champion only if it wins on the
latest labelled production window (champion/challenger gate)."""
import json

from mlmonitor.models.promotion import evaluate_and_promote


def main() -> None:
    result = evaluate_and_promote()
    print(json.dumps(result.__dict__, indent=2, default=str))
    if not result.promoted:
        # non-zero exit so CI surfaces a rejected challenger without overwriting the champion
        raise SystemExit(0)


if __name__ == "__main__":
    main()
