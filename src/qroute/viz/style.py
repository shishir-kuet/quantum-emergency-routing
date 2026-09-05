"""Shared plotting setup.

Matplotlib is imported lazily everywhere in this package. Two reasons: importing
it costs a noticeable fraction of a second, which is annoying in a CLI that often
does not plot; and it drags in a GUI backend that fails outright in a headless
environment. :func:`use_headless_backend` forces ``Agg`` before the first import,
which is what any script that only writes PNG files should do.

Every plotting function returns the ``Figure`` it built and takes an optional
*path*. Returning the figure means callers can keep customising it; taking a path
means the common case is one line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from ..exceptions import MissingDependencyError
from ..logging_utils import get_logger

__all__ = [
    "use_headless_backend",
    "get_pyplot",
    "save_figure",
    "PALETTE",
    "annotate_source",
]

_LOG = get_logger(__name__)

#: A small, colour-blind-safe palette (Okabe-Ito), used consistently so the same
#: solver is the same colour across every chart in a report.
PALETTE = {
    "classical": "#0072B2",
    "quantum": "#D55E00",
    "annealing": "#009E73",
    "heuristic": "#CC79A7",
    "optimal": "#000000",
    "muted": "#999999",
    "road": "#D9D9D9",
    "highlight": "#E69F00",
}


def use_headless_backend() -> None:
    """Force the non-interactive ``Agg`` backend.

    Call this *before* any other matplotlib import -- once a backend is chosen,
    switching is unreliable.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("matplotlib", "plotting results") from exc
    matplotlib.use("Agg", force=True)


def get_pyplot() -> Any:
    """Import and return ``matplotlib.pyplot`` with a readable default style."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("matplotlib", "plotting results") from exc

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    return plt


def save_figure(figure: Any, path: Optional[Path | str]) -> Optional[Path]:
    """Write *figure* to *path*, creating parent directories as needed."""
    if path is None:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    _LOG.info("Wrote %s", target)
    return target


def annotate_source(axes: Any, text: str, *, coords: Tuple[float, float] = (0.0, -0.12)) -> None:
    """Put a small provenance note under an axes.

    Every figure in a report should say where its numbers came from -- which
    instance, which simulator, how many shots. Six months later that caption is
    the difference between a usable result and a pretty picture.
    """
    axes.annotate(
        text,
        xy=coords,
        xycoords="axes fraction",
        fontsize=7.5,
        color=PALETTE["muted"],
        va="top",
    )


def _equal_aspect_geographic(axes: Any, latitudes: Sequence[float]) -> None:
    """Scale a lat/lon plot so it is not visually distorted.

    A degree of longitude is :math:`\\cos(\\text{lat})` times shorter than a
    degree of latitude, so plotting raw coordinates on equal axes stretches the
    map east-west. At Dhaka's latitude (~23.75 deg) that is a 9% error -- small
    but enough to make a square block look like a rectangle.
    """
    import numpy as np

    if not len(latitudes):
        return
    mean_latitude = float(np.mean(np.asarray(latitudes, dtype=float)))
    axes.set_aspect(1.0 / max(np.cos(np.radians(mean_latitude)), 1e-6))
