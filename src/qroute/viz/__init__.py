"""Figures: maps of routes, and charts of solver behaviour.

Matplotlib is imported lazily inside each function, so importing this package (or
``qroute`` as a whole) costs nothing if you never plot. Scripts that run headless
should call :func:`use_headless_backend` first.

``style``
    Shared rcParams, a colour-blind-safe palette, ``save_figure``, and the
    latitude-aware aspect correction that stops Dhaka looking 9% too wide.

``maps``
    :func:`plot_graph`, :func:`plot_instance`, :func:`plot_route`,
    :func:`plot_node_path`, :func:`plot_assignment` -- routes drawn on real street
    geometry, not straight lines between abstract points. Use
    :func:`plot_node_path` for rung 1 (whose routes are graph node ids) and
    :func:`plot_route` for rungs 2 and 3 (whose routes are instance indices).

``charts``
    :func:`plot_convergence`, :func:`plot_energy_distribution`,
    :func:`plot_comparison`, :func:`plot_depth_scaling`.

Every function returns its ``Figure`` and accepts an optional ``path``, so the
one-line case works and the customise-further case is still available.
"""

from __future__ import annotations

from .charts import (
    plot_comparison,
    plot_convergence,
    plot_depth_scaling,
    plot_energy_distribution,
)
from .maps import plot_assignment, plot_graph, plot_instance, plot_node_path, plot_route
from .style import PALETTE, get_pyplot, save_figure, use_headless_backend

__all__ = [
    # style
    "use_headless_backend",
    "get_pyplot",
    "save_figure",
    "PALETTE",
    # maps
    "plot_graph",
    "plot_instance",
    "plot_route",
    "plot_node_path",
    "plot_assignment",
    # charts
    "plot_convergence",
    "plot_energy_distribution",
    "plot_comparison",
    "plot_depth_scaling",
]
