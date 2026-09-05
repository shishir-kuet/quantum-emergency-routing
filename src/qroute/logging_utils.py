"""Logging setup shared by every entry point.

Call :func:`configure_logging` exactly once (the CLI and the numbered
pipeline scripts do this for you), then use :func:`get_logger` inside
modules. Library modules must never call ``configure_logging`` themselves --
that is the application's decision, not the library's.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

__all__ = ["configure_logging", "get_logger"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Attach a single stderr handler to the ``qroute`` logger tree.

    Parameters
    ----------
    level:
        Any name accepted by :mod:`logging` (``DEBUG``, ``INFO``, ...).
    force:
        Reconfigure even if this function has already run.
    """
    global _configured
    if _configured and not force:
        return

    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root = logging.getLogger("qroute")
    root.setLevel(numeric)
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(numeric)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # Keep third-party chatter out of the way; we surface our own progress.
    for noisy in ("matplotlib", "PIL", "urllib3", "fiona", "rasterio"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))

    root.propagate = False
    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger under the ``qroute`` namespace.

    ``get_logger(__name__)`` from inside the package yields the module's
    dotted path unchanged; anything else is prefixed with ``qroute.``.
    """
    if not name or name == "qroute":
        return logging.getLogger("qroute")
    if name.startswith("qroute."):
        return logging.getLogger(name)
    return logging.getLogger(f"qroute.{name}")
