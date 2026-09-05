"""Filesystem-root discovery.

The package is normally used from a checkout (``pip install -e .``), where
data, config and results all live beside the ``src/`` directory. This module
locates that directory robustly so relative paths in the config file mean the
same thing no matter which working directory you launch from.

Resolution order:

1. the ``QROUTE_ROOT`` environment variable, if set;
2. the nearest ancestor of this file containing a ``pyproject.toml``;
3. the current working directory (last resort, e.g. a wheel install).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["find_project_root", "ensure_dir"]

_ROOT_MARKERS = ("pyproject.toml", "requirements.txt")


def find_project_root() -> Path:
    """Return the project root directory as an absolute :class:`Path`."""
    env = os.environ.get("QROUTE_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if any((candidate / marker).is_file() for marker in _ROOT_MARKERS):
            return candidate

    return Path.cwd().resolve()


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
