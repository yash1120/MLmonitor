"""Central logging setup: one structured-ish formatter for app + scripts."""
from __future__ import annotations

import logging
import sys

from mlmonitor.config import settings

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel((level or settings.log_level).upper())
    # third-party chatter we never need at INFO
    for noisy in ("httpx", "httpcore", "urllib3", "mlflow", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
