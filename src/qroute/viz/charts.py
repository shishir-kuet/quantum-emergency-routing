r"""Charts: convergence, sampled-energy distributions, and solver comparisons.

Three plots that each answer a question the numbers alone leave ambiguous.

**Convergence** (:func:`plot_convergence`) draws the optimiser's objective *and*
the running-best sampled energy on the same axes. That pairing is the point: a
flat mean curve next to a best-energy curve that dropped at iteration 40 means the
variational loop is not the bottleneck -- the sampling already found the answer. The
opposite shape (mean falling steadily, best energy stuck) means the distribution
is concentrating on the wrong state, which usually points at a penalty weight.

**Energy distribution** (:func:`plot_energy_distribution`) shows where the
probability mass actually sits, with the exact ground state marked. On a
constrained problem this is often the most honest single figure in a write-up: it
shows the optimum being found *and* how thin that peak is.

**Comparison** (:func:`plot_comparison`) is a grouped bar chart of route cost with
an optimum reference line. Bars are only drawn for feasible results, and an
infeasible solver gets an explicit "infeasible" label rather than a zero-height
bar, because a missing bar reads as "very good" at a glance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..evaluation.metrics import ComparisonRow, SampleStatistics
from ..logging_utils import get_logger
from ..quantum.qaoa import QAOAResult
from .style import PALETTE, annotate_source, get_pyplot, save_figure

if TYPE_CHECKING:  # pragma: no cover
    from ..qubo.matrix import QUBO

__all__ = [
    "plot_convergence",
    "plot_energy_distribution",
    "plot_comparison",
    "plot_depth_scaling",
]

_LOG = get_logger(__name__)


def plot_convergence(
    result: QAOAResult,
    *,
    path: Optional[Path | str] = None,
    reference_energy: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 4.6),
) -> Any:
    """Objective and running-best sampled energy against iteration count.

    *reference_energy* draws the exact QUBO ground energy as a horizontal line.
    Supply it when you have run brute force -- without it the reader cannot tell
    whether a plateau is convergence or a ceiling.
    """
    plt = get_pyplot()
    figure, axes = plt.subplots(figsize=figsize)

    iterations, values = result.convergence()
    if not iterations.size:
        _LOG.warning("QAOA result has no history; nothing to plot")
        return figure

    label = "mean energy" if result.cvar_alpha >= 1.0 else f"CVaR({result.cvar_alpha:g}) energy"
    axes.plot(
        iterations,
        values,
        color=PALETTE["quantum"],
        linewidth=1.5,
        label=f"objective ({label})",
    )
    axes.plot(
        iterations,
        result.best_energy_trace(),
        color=PALETTE["annealing"],
        linewidth=1.8,
        linestyle="-",
        label="best sample so far",
    )
    if reference_energy is not None:
        axes.axhline(
            reference_energy,
            color=PALETTE["optimal"],
            linewidth=1.0,
            linestyle=":",
            label=f"exact ground energy ({reference_energy:.4g})",
        )

    axes.set_xlabel("objective evaluations")
    axes.set_ylabel("QUBO energy")
    axes.set_title(
        title
        or f"QAOA convergence: p={result.reps}, {result.optimizer}, {result.expectation_mode}"
    )
    axes.legend(loc="best", fontsize=8)
    annotate_source(
        axes,
        f"{result.metadata.get('formulation', '?')} | {result.n_qubits} qubits | "
        f"{result.shots or 'exact'} shots/eval | depth {result.metadata.get('circuit_depth', '?')}",
    )
    save_figure(figure, path)
    return figure


def plot_energy_distribution(
    result: QAOAResult,
    qubo: Optional["QUBO"] = None,
    *,
    path: Optional[Path | str] = None,
    spectrum: Optional[np.ndarray] = None,
    ground_energy: Optional[float] = None,
    statistics: Optional[SampleStatistics] = None,
    bins: int = 50,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.6, 4.4),
) -> Any:
    """Where the sampled probability mass actually sits.

    Left panel: shot-weighted histogram of sampled QUBO energies. Right panel (only
    when *statistics* is supplied): the route costs of the feasible shots, which is
    the distribution a dispatcher would care about.

    Parameters
    ----------
    qubo:
        Needed to score the counts. Pass ``formulation.qubo()``. Without it the
        sampled histogram is skipped and only the reference lines are drawn.
    spectrum:
        The full energy spectrum from :func:`~qroute.qubo.matrix.all_energies`,
        drawn faintly behind the samples. This is the fairest visual baseline for
        QAOA: it is what a *uniform* sampler would produce, and beating uniform is
        the minimum bar -- one that low-depth runs on constrained problems do not
        always clear.
    """
    plt = get_pyplot()

    two_panel = statistics is not None and bool(statistics.feasible_objectives)
    if two_panel:
        figure, (axes, cost_axes) = plt.subplots(1, 2, figsize=figsize)
    else:
        figure, axes = plt.subplots(figsize=(figsize[0] * 0.62, figsize[1]))
        cost_axes = None

    if spectrum is not None and np.size(spectrum):
        axes.hist(
            np.asarray(spectrum, dtype=float),
            bins=bins,
            density=True,
            color=PALETTE["muted"],
            alpha=0.45,
            label="all states (uniform)",
        )

    if result.counts and qubo is not None:
        from ..quantum.qaoa import sample_energies

        _, sampled_energies, weights = sample_energies(dict(result.counts), qubo)
        axes.hist(
            sampled_energies,
            bins=bins,
            weights=weights,
            density=True,
            color=PALETTE["quantum"],
            alpha=0.65,
            label="QAOA samples",
        )
    elif result.counts and qubo is None:
        _LOG.warning("No QUBO supplied; skipping the sampled-energy histogram")
    else:
        _LOG.info("Result carries no counts (exact mode); drawing reference lines only")

    if ground_energy is not None:
        axes.axvline(
            ground_energy,
            color=PALETTE["optimal"],
            linewidth=1.2,
            linestyle=":",
            label=f"ground energy ({ground_energy:.4g})",
        )
    axes.axvline(
        result.best_energy,
        color=PALETTE["annealing"],
        linewidth=1.5,
        label=f"best sample ({result.best_energy:.4g})",
    )

    axes.set_xlabel("QUBO energy")
    axes.set_ylabel("density")
    axes.set_title(
        title
        or f"Sampled energies: p={result.reps}, {result.shots or 'exact'} shots"
    )
    axes.legend(loc="best", fontsize=8)

    if cost_axes is not None and statistics is not None:
        cost_axes.hist(
            statistics.feasible_objectives,
            bins=min(bins, max(len(set(statistics.feasible_objectives)), 1)),
            color=PALETTE["annealing"],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
        )
        reference = statistics.metadata.get("reference_objective")
        if reference is not None:
            cost_axes.axvline(
                float(reference),
                color=PALETTE["optimal"],
                linestyle=":",
                linewidth=1.2,
                label=f"optimum ({float(reference):,.0f})",
            )
            cost_axes.legend(loc="best", fontsize=8)
        cost_axes.set_xlabel("route cost (feasible shots)")
        cost_axes.set_ylabel("shots")
        cost_axes.set_title(f"Feasible shots: {statistics.feasibility_rate:.1%} of total")

    parts = [str(result.metadata.get("formulation", "?")), f"{result.n_qubits} qubits"]
    if statistics is not None:
        parts.append(f"{statistics.n_unique:,} unique outcomes")
        success = statistics.success_probability
        if success is not None:
            parts.append(f"p(optimal) {success:.4f}")
    annotate_source(axes, " | ".join(parts))

    figure.tight_layout()
    save_figure(figure, path)
    return figure


def plot_comparison(
    rows: Sequence[ComparisonRow],
    *,
    path: Optional[Path | str] = None,
    unit: str = "s",
    title: str = "Route cost by solver",
    figsize: Tuple[float, float] = (8.4, 4.6),
) -> Any:
    """Grouped bar chart of route cost per solver, with time as a second panel."""
    plt = get_pyplot()
    figure, (cost_axes, time_axes) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.0, 1.0]}
    )

    names = [row.solver for row in rows]
    positions = np.arange(len(rows))

    def colour_for(row: ComparisonRow) -> str:
        if row.solver.startswith("qaoa"):
            return PALETTE["quantum"]
        if "anneal" in row.solver:
            return PALETTE["annealing"]
        if row.optimal:
            return PALETTE["classical"]
        return PALETTE["heuristic"]

    costs = [0.0 if row.objective is None else row.objective for row in rows]
    cost_axes.bar(
        positions,
        costs,
        color=[colour_for(row) for row in rows],
        edgecolor="white",
        linewidth=0.8,
    )
    for position, row in zip(positions, rows):
        if row.objective is None:
            cost_axes.annotate(
                "infeasible",
                (position, 0.0),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=PALETTE["quantum"],
                rotation=90,
            )
        else:
            cost_axes.annotate(
                f"{row.objective:,.0f}",
                (position, row.objective),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
            )

    optimal = [row.objective for row in rows if row.optimal and row.objective is not None]
    if optimal:
        cost_axes.axhline(
            min(optimal),
            color=PALETTE["optimal"],
            linestyle=":",
            linewidth=1.0,
            label=f"proven optimum ({min(optimal):,.0f} {unit})",
        )
        cost_axes.legend(loc="lower right", fontsize=8)

    cost_axes.set_xticks(positions)
    cost_axes.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    cost_axes.set_ylabel(f"route cost ({unit})")
    cost_axes.set_title(title)

    time_axes.barh(
        positions,
        [max(row.seconds, 1e-6) for row in rows],
        color=[colour_for(row) for row in rows],
        edgecolor="white",
        linewidth=0.8,
    )
    time_axes.set_yticks(positions)
    time_axes.set_yticklabels(names, fontsize=8)
    time_axes.set_xscale("log")
    time_axes.set_xlabel("wall-clock (s, log scale)")
    time_axes.set_title("Runtime")
    time_axes.invert_yaxis()

    annotate_source(
        time_axes,
        "Simulator timings are not hardware timings: simulating n qubits costs O(2^n).",
        coords=(0.0, -0.22),
    )
    figure.tight_layout()
    save_figure(figure, path)
    return figure


def plot_depth_scaling(
    results: Mapping[int, QAOAResult],
    *,
    path: Optional[Path | str] = None,
    reference_energy: Optional[float] = None,
    success_probabilities: Optional[Mapping[int, Optional[float]]] = None,
    title: str = "QAOA quality against circuit depth",
    figsize: Tuple[float, float] = (8.0, 4.6),
) -> Any:
    r"""Best sampled energy and success probability against :math:`p`.

    The central empirical question of the whole project: does more depth help?
    Theory says the approximation improves monotonically with :math:`p` in the
    noiseless limit, and this plot is where that claim meets a finite shot budget.
    Two-qubit gate count grows linearly in :math:`p`, so on hardware the curve is
    expected to turn around -- finding that turning point is the interesting result.
    """
    plt = get_pyplot()
    figure, axes = plt.subplots(figsize=figsize)

    depths = sorted(results)
    energies = [results[depth].best_energy for depth in depths]
    axes.plot(
        depths,
        energies,
        marker="o",
        color=PALETTE["quantum"],
        linewidth=1.6,
        label="best sampled energy",
    )
    if reference_energy is not None:
        axes.axhline(
            reference_energy,
            color=PALETTE["optimal"],
            linestyle=":",
            linewidth=1.0,
            label=f"ground energy ({reference_energy:.4g})",
        )

    axes.set_xlabel("QAOA layers p")
    axes.set_ylabel("QUBO energy")
    axes.set_xticks(depths)
    axes.set_title(title)

    if success_probabilities:
        twin = axes.twinx()
        values = [success_probabilities.get(depth) for depth in depths]
        clean = [(depth, value) for depth, value in zip(depths, values) if value is not None]
        if clean:
            twin.plot(
                [depth for depth, _ in clean],
                [value for _, value in clean],
                marker="s",
                color=PALETTE["annealing"],
                linewidth=1.4,
                label="p(optimal)",
            )
            twin.set_ylabel("probability of an optimal shot")
            twin.grid(False)
            twin.set_ylim(bottom=0.0)

    handles, labels = axes.get_legend_handles_labels()
    if success_probabilities:
        extra = twin.get_legend_handles_labels()
        handles += extra[0]
        labels += extra[1]
    axes.legend(handles, labels, loc="best", fontsize=8)

    gates = {depth: results[depth].metadata.get("circuit_size") for depth in depths}
    annotate_source(
        axes,
        "circuit operations per depth: "
        + ", ".join(f"p={depth}:{gates[depth]}" for depth in depths),
    )
    save_figure(figure, path)
    return figure
