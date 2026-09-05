"""Aer simulator setup and transpilation.

Local simulation only, by design. Hardware and noise models come later, once the
classical and QUBO layers are proven -- chasing a hardware result before the
encoding is verified is the fastest way to waste a queue slot on a bug.

Two things here are worth more attention than they usually get:

**Transpile once, bind many times.** A QAOA run evaluates the circuit hundreds of
times with different angles. Transpiling the *parameterised* circuit once and then
binding numbers into the transpiled version does the routing and gate-decomposition
work a single time. Transpiling inside the objective function instead can easily
dominate the runtime, and it makes any timing comparison against a classical
solver meaningless.

**Seeding.** ``seed_simulator`` makes shot noise reproducible. Without it, two runs
of the same experiment give different answers and there is no way to tell a real
improvement from sampling luck.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from ..config import BackendConfig, Config
from ..exceptions import BackendError, MissingDependencyError
from ..logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit

__all__ = [
    "make_simulator",
    "transpile_circuit",
    "describe_backend",
    "run_circuit",
]

_LOG = get_logger(__name__)


def _import_aer() -> Any:
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "qiskit-aer", "simulating quantum circuits locally"
        ) from exc
    return AerSimulator


def make_simulator(
    config: Optional[Config] = None,
    *,
    backend_config: Optional[BackendConfig] = None,
    noise_model: Optional[Any] = None,
    seed: Optional[int] = None,
) -> Any:
    """Build a configured ``AerSimulator``.

    Parameters
    ----------
    config, backend_config:
        Provide either; *backend_config* wins if both are given.
    noise_model:
        An actual ``qiskit_aer.noise.NoiseModel`` instance, if you have built one.
        Config-driven noise is deliberately **not** implemented: a noise model
        assembled from a config string is a noise model nobody has inspected, and
        an unexamined error model is worse than none.
    seed:
        Overrides ``qaoa.seed_simulator`` from the config.
    """
    AerSimulator = _import_aer()

    settings = backend_config or (config.backend if config is not None else None)
    if settings is None:
        raise BackendError("make_simulator needs either config or backend_config")

    if settings.provider != "aer":
        raise BackendError(
            f"Only the 'aer' provider is supported in this build, got "
            f"{settings.provider!r}"
        )
    if settings.noise_model is not None and noise_model is None:
        raise BackendError(
            f"backend.noise_model is set to {settings.noise_model!r}, but "
            f"config-driven noise models are not implemented. Build a NoiseModel "
            f"yourself and pass it as make_simulator(noise_model=...)."
        )

    if seed is None and config is not None:
        seed = config.qaoa.seed_simulator

    options: Dict[str, Any] = {"method": settings.method}
    if seed is not None:
        options["seed_simulator"] = int(seed)
    if settings.max_parallel_threads:
        options["max_parallel_threads"] = int(settings.max_parallel_threads)
    if noise_model is not None:
        options["noise_model"] = noise_model

    simulator = AerSimulator(**options)
    _LOG.info("Simulator ready: %s", describe_backend(simulator))
    return simulator


def transpile_circuit(
    circuit: "QuantumCircuit",
    simulator: Any,
    *,
    optimization_level: int = 3,
    seed_transpiler: Optional[int] = 42,
) -> "QuantumCircuit":
    """Transpile *circuit* for *simulator*, keeping any free parameters free.

    Call this once on the parameterised ansatz, then bind angles to the result --
    see the module docstring.
    """
    try:
        from qiskit import transpile
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("qiskit", "transpiling circuits") from exc

    transpiled = transpile(
        circuit,
        backend=simulator,
        optimization_level=int(optimization_level),
        seed_transpiler=seed_transpiler,
    )
    _LOG.debug(
        "Transpiled: depth %d -> %d, size %d -> %d",
        circuit.depth(),
        transpiled.depth(),
        circuit.size(),
        transpiled.size(),
    )
    return transpiled


def run_circuit(
    circuit: "QuantumCircuit",
    simulator: Any,
    *,
    shots: int = 4096,
) -> Dict[str, int]:
    """Execute a measured circuit and return raw counts.

    Keys are Qiskit-ordered bitstrings (qubit 0 rightmost). Always convert them
    with :func:`~qroute.qubo.matrix.bitstring_to_array`.
    """
    if shots < 1:
        raise BackendError(f"shots must be at least 1, got {shots}")
    try:
        job = simulator.run(circuit, shots=int(shots))
        result = job.result()
        counts = result.get_counts()
    except Exception as exc:  # pragma: no cover - surfaced with context
        raise BackendError(f"Simulator run failed: {exc}") from exc

    if not isinstance(counts, dict):
        raise BackendError(
            "Expected a single counts dictionary; the circuit should contain one "
            "measured register"
        )
    return {str(key): int(value) for key, value in counts.items()}


def describe_backend(simulator: Any) -> str:
    """One-line summary of a simulator's configuration."""
    name = getattr(simulator, "name", None)
    if callable(name):  # older Aer exposed name() as a method
        try:
            name = name()
        except Exception:  # pragma: no cover
            name = None
    options = getattr(simulator, "options", None)
    method = getattr(options, "method", "unknown") if options is not None else "unknown"
    seed = getattr(options, "seed_simulator", None) if options is not None else None
    noise = getattr(options, "noise_model", None) if options is not None else None
    return (
        f"{name or 'AerSimulator'}(method={method}, seed={seed}, "
        f"noise={'yes' if noise is not None else 'none'})"
    )
