"""Console logging configured once, via rich, with a consistent format.

We keep logging deliberately simple: one ``get_logger`` entry point and a single
``configure_logging`` call made by the CLIs. Structured run data goes to the
artifact store (``metrics.jsonl`` / ``events.jsonl``), not to the console — the
console is for humans watching a run.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a single rich handler on the root logger (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    configure_logging()
    return logging.getLogger(name)
