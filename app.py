from __future__ import annotations

import logging

from zgc3_assistant.config import get_settings
from zgc3_assistant.logging_config import configure_logging
from zgc3_assistant.orchestrator import Orchestrator
from zgc3_assistant.ui.layout import build_app


def main() -> None:
    """Entry-point used by CLI, scripts, and tests."""
    settings = get_settings()
    configure_logging(settings.log_level)
    orchestrator = Orchestrator(settings=settings)
    ui = build_app(orchestrator)
    ui.queue().launch()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - startup guardrail
        logging.exception("Fatal error launching zgc3 assistant: %s", exc)
        raise
