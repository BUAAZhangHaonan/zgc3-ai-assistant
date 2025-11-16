from __future__ import annotations

import logging
import logging.config
from typing import Any, Mapping

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure project-wide logging with sane defaults."""
    logging_config: Mapping[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": DEFAULT_FORMAT,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": level,
            }
        },
        "root": {"level": level, "handlers": ["console"]},
    }

    logging.config.dictConfig(logging_config)
