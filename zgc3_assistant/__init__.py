"""zgc3 assistant package."""

from importlib.metadata import version

__all__ = ["__version__"]

try:  # pragma: no cover - best effort metadata
    __version__ = version("zgc3-ai-assistant")
except Exception:  # pragma: no cover - dev installs
    __version__ = "0.1.0"

