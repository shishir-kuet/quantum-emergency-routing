"""``python -m qroute`` entry point.

Exists so the project is runnable straight from a clone -- ``python -m qroute
info`` works before ``pip install -e .`` has ever been run, which matters when
someone is checking whether the dependencies resolved at all.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
