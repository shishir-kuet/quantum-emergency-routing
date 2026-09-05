"""Typed configuration objects and the YAML loader.

Design notes
------------
* Every field has a default, so ``config/default.yaml`` is a pure *override*
  layer -- the package is fully usable with no config file at all.
* Sections are immutable (``frozen=True``) dataclasses. Passing a config
  object around is therefore safe: no module can mutate it under another's
  feet, and ``dataclasses.replace`` gives cheap, explicit variants for
  parameter sweeps.
* Validation happens at load time, not at use time. A typo in ``optimizer``
  should fail before a 20-minute simulation, not after it.
* Relative directories in the YAML are interpreted against the project root
  (see :mod:`qroute.paths`) and materialised by :meth:`Config.build_paths`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from .exceptions import ConfigError
from .logging_utils import get_logger
from .paths import ensure_dir, find_project_root

__all__ = [
    "BBox",
    "ProjectConfig",
    "DataConfig",
    "InstanceConfig",
    "QuboConfig",
    "QaoaConfig",
    "BackendConfig",
    "OutputConfig",
    "Paths",
    "Config",
    "load_config",
]

_LOG = get_logger(__name__)

NETWORK_TYPES = frozenset(
    {"drive", "drive_service", "bike", "walk", "all", "all_private"}
)
SAMPLING_STRATEGIES = frozenset({"random", "betweenness", "degree", "farthest"})
EDGE_WEIGHTS = frozenset({"travel_time", "length"})
OPTIMIZERS = frozenset({"COBYLA", "POWELL", "NELDER-MEAD", "SPSA"})
AER_METHODS = frozenset(
    {"automatic", "statevector", "matrix_product_state", "density_matrix"}
)
PROVIDERS = frozenset({"aer"})


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{label!r} must be a number, got {value!r}") from exc


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label!r} must be an integer, got a boolean")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label!r} must be an integer, got {value!r}") from exc
    if as_float != int(as_float):
        raise ConfigError(f"{label!r} must be a whole number, got {value!r}")
    return int(as_float)


_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{label!r} must be a boolean, got {value!r}")


def _as_choice(value: Any, allowed: frozenset, label: str, *, upper: bool = False) -> str:
    text = str(value).strip()
    text = text.upper() if upper else text.lower()
    if text not in allowed:
        raise ConfigError(
            f"{label!r} must be one of {sorted(allowed)}, got {value!r}"
        )
    return text


def _section(data: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """Extract a mapping section, tolerating absent or empty (``None``) keys."""
    raw = data.get(name)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"Config section '{name}' must be a mapping, got {type(raw).__name__}"
        )
    return dict(raw)


def _drop_unknown(data: Mapping[str, Any], cls: type, label: str) -> Dict[str, Any]:
    """Filter *data* down to the declared fields of *cls*, warning about the rest."""
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - valid)
    if unknown:
        _LOG.warning(
            "Ignoring unrecognised key(s) in config section [%s]: %s",
            label,
            ", ".join(unknown),
        )
    return {key: value for key, value in data.items() if key in valid}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BBox:
    """A geographic bounding box in WGS-84 degrees."""

    north: float = 23.7580
    south: float = 23.7350
    east: float = 90.3900
    west: float = 90.3650

    def __post_init__(self) -> None:
        if not -90.0 <= self.south < self.north <= 90.0:
            raise ConfigError(
                f"bbox latitudes invalid: need -90 <= south < north <= 90, "
                f"got south={self.south}, north={self.north}"
            )
        if not -180.0 <= self.west < self.east <= 180.0:
            raise ConfigError(
                f"bbox longitudes invalid: need -180 <= west < east <= 180, "
                f"got west={self.west}, east={self.east}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BBox":
        clean = _drop_unknown(data, cls, "data.bbox")
        return cls(**{key: _as_float(val, f"data.bbox.{key}") for key, val in clean.items()})

    @property
    def as_osmnx_bbox(self) -> Tuple[float, float, float, float]:
        """``(west, south, east, north)`` -- the left/bottom/right/top ordering
        expected by OSMnx >= 2.0. The 1.x shim in
        :mod:`qroute.data._compat` re-orders this as needed."""
        return (self.west, self.south, self.east, self.north)

    @property
    def center(self) -> Tuple[float, float]:
        """``(latitude, longitude)`` of the box centre."""
        return ((self.north + self.south) / 2.0, (self.east + self.west) / 2.0)

    def approx_size_km(self) -> Tuple[float, float]:
        """Rough ``(height_km, width_km)``, good enough for a log line."""
        import math

        lat_km = (self.north - self.south) * 110.574
        mean_lat_rad = math.radians((self.north + self.south) / 2.0)
        lon_km = (self.east - self.west) * 111.320 * math.cos(mean_lat_rad)
        return (lat_km, lon_km)

    def describe(self) -> str:
        height, width = self.approx_size_km()
        lat, lon = self.center
        return (
            f"bbox N{self.north:.4f} S{self.south:.4f} E{self.east:.4f} W{self.west:.4f} "
            f"(~{height:.2f} x {width:.2f} km, centre {lat:.4f}, {lon:.4f})"
        )


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "quantum-emergency-routing"
    seed: int = 42
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        clean = _drop_unknown(data, cls, "project")
        return cls(
            name=str(clean.get("name", cls.name)),
            seed=_as_int(clean.get("seed", cls.seed), "project.seed"),
            log_level=str(clean.get("log_level", cls.log_level)).upper(),
        )


@dataclass(frozen=True)
class DataConfig:
    bbox: BBox = BBox()
    network_type: str = "drive"
    simplify: bool = True
    consolidate_tolerance_m: float = 0.0
    default_speed_kph: float = 20.0
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    graph_filename: str = "dhaka_dhanmondi_drive.graphml"

    def __post_init__(self) -> None:
        if self.consolidate_tolerance_m < 0:
            raise ConfigError("data.consolidate_tolerance_m must be >= 0")
        if self.default_speed_kph <= 0:
            raise ConfigError("data.default_speed_kph must be > 0")
        if not self.graph_filename.endswith(".graphml"):
            raise ConfigError("data.graph_filename must end with '.graphml'")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataConfig":
        clean = _drop_unknown(data, cls, "data")
        bbox_raw = clean.get("bbox")
        if bbox_raw is None:
            bbox = BBox()
        elif isinstance(bbox_raw, Mapping):
            bbox = BBox.from_dict(bbox_raw)
        else:
            raise ConfigError("data.bbox must be a mapping with north/south/east/west")
        return cls(
            bbox=bbox,
            network_type=_as_choice(
                clean.get("network_type", cls.network_type),
                NETWORK_TYPES,
                "data.network_type",
            ),
            simplify=_as_bool(clean.get("simplify", cls.simplify), "data.simplify"),
            consolidate_tolerance_m=_as_float(
                clean.get("consolidate_tolerance_m", cls.consolidate_tolerance_m),
                "data.consolidate_tolerance_m",
            ),
            default_speed_kph=_as_float(
                clean.get("default_speed_kph", cls.default_speed_kph),
                "data.default_speed_kph",
            ),
            raw_dir=str(clean.get("raw_dir", cls.raw_dir)),
            processed_dir=str(clean.get("processed_dir", cls.processed_dir)),
            graph_filename=str(clean.get("graph_filename", cls.graph_filename)),
        )


@dataclass(frozen=True)
class InstanceConfig:
    n_nodes: int = 5
    n_incidents: int = 3
    n_ambulances: int = 2
    sampling: str = "betweenness"
    weight: str = "travel_time"

    def __post_init__(self) -> None:
        if self.n_nodes < 3:
            raise ConfigError("instance.n_nodes must be >= 3 for a meaningful tour")
        if self.n_incidents < 1:
            raise ConfigError("instance.n_incidents must be >= 1")
        if self.n_ambulances < 1:
            raise ConfigError("instance.n_ambulances must be >= 1")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InstanceConfig":
        clean = _drop_unknown(data, cls, "instance")
        return cls(
            n_nodes=_as_int(clean.get("n_nodes", cls.n_nodes), "instance.n_nodes"),
            n_incidents=_as_int(
                clean.get("n_incidents", cls.n_incidents), "instance.n_incidents"
            ),
            n_ambulances=_as_int(
                clean.get("n_ambulances", cls.n_ambulances), "instance.n_ambulances"
            ),
            sampling=_as_choice(
                clean.get("sampling", cls.sampling),
                SAMPLING_STRATEGIES,
                "instance.sampling",
            ),
            weight=_as_choice(
                clean.get("weight", cls.weight), EDGE_WEIGHTS, "instance.weight"
            ),
        )


@dataclass(frozen=True)
class QuboConfig:
    penalty_scale: float = 2.0
    normalise: bool = True

    def __post_init__(self) -> None:
        if self.penalty_scale <= 0:
            raise ConfigError("qubo.penalty_scale must be > 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuboConfig":
        clean = _drop_unknown(data, cls, "qubo")
        return cls(
            penalty_scale=_as_float(
                clean.get("penalty_scale", cls.penalty_scale), "qubo.penalty_scale"
            ),
            normalise=_as_bool(clean.get("normalise", cls.normalise), "qubo.normalise"),
        )


@dataclass(frozen=True)
class QaoaConfig:
    reps: int = 2
    optimizer: str = "COBYLA"
    maxiter: int = 200
    shots: int = 4096
    initial_point: Optional[Tuple[float, ...]] = None
    seed_simulator: int = 42

    def __post_init__(self) -> None:
        if self.reps < 1:
            raise ConfigError("qaoa.reps must be >= 1")
        if self.maxiter < 1:
            raise ConfigError("qaoa.maxiter must be >= 1")
        if self.shots < 1:
            raise ConfigError("qaoa.shots must be >= 1")
        if self.initial_point is not None and len(self.initial_point) != 2 * self.reps:
            raise ConfigError(
                f"qaoa.initial_point must have exactly 2 * reps = {2 * self.reps} "
                f"entries (gammas then betas), got {len(self.initial_point)}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QaoaConfig":
        clean = _drop_unknown(data, cls, "qaoa")
        raw_point = clean.get("initial_point", None)
        if raw_point is None:
            initial_point: Optional[Tuple[float, ...]] = None
        elif isinstance(raw_point, Sequence) and not isinstance(raw_point, (str, bytes)):
            initial_point = tuple(
                _as_float(v, f"qaoa.initial_point[{i}]") for i, v in enumerate(raw_point)
            )
        else:
            raise ConfigError("qaoa.initial_point must be null or a list of numbers")
        return cls(
            reps=_as_int(clean.get("reps", cls.reps), "qaoa.reps"),
            optimizer=_as_choice(
                clean.get("optimizer", cls.optimizer),
                OPTIMIZERS,
                "qaoa.optimizer",
                upper=True,
            ),
            maxiter=_as_int(clean.get("maxiter", cls.maxiter), "qaoa.maxiter"),
            shots=_as_int(clean.get("shots", cls.shots), "qaoa.shots"),
            initial_point=initial_point,
            seed_simulator=_as_int(
                clean.get("seed_simulator", cls.seed_simulator), "qaoa.seed_simulator"
            ),
        )


@dataclass(frozen=True)
class BackendConfig:
    provider: str = "aer"
    method: str = "automatic"
    noise_model: Optional[str] = None
    optimization_level: int = 3
    max_parallel_threads: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.optimization_level <= 3:
            raise ConfigError("backend.optimization_level must be between 0 and 3")
        if self.max_parallel_threads < 0:
            raise ConfigError("backend.max_parallel_threads must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackendConfig":
        clean = _drop_unknown(data, cls, "backend")
        noise = clean.get("noise_model", None)
        return cls(
            provider=_as_choice(
                clean.get("provider", cls.provider), PROVIDERS, "backend.provider"
            ),
            method=_as_choice(
                clean.get("method", cls.method), AER_METHODS, "backend.method"
            ),
            noise_model=None if noise in (None, "", "null", "none") else str(noise),
            optimization_level=_as_int(
                clean.get("optimization_level", cls.optimization_level),
                "backend.optimization_level",
            ),
            max_parallel_threads=_as_int(
                clean.get("max_parallel_threads", cls.max_parallel_threads),
                "backend.max_parallel_threads",
            ),
        )


@dataclass(frozen=True)
class OutputConfig:
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    save_figures: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputConfig":
        clean = _drop_unknown(data, cls, "output")
        return cls(
            results_dir=str(clean.get("results_dir", cls.results_dir)),
            figures_dir=str(clean.get("figures_dir", cls.figures_dir)),
            save_figures=_as_bool(
                clean.get("save_figures", cls.save_figures), "output.save_figures"
            ),
        )


# ---------------------------------------------------------------------------
# Resolved filesystem layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    """Absolute paths derived from a :class:`Config`."""

    root: Path
    raw_dir: Path
    processed_dir: Path
    results_dir: Path
    figures_dir: Path
    graph_file: Path

    def ensure(self) -> "Paths":
        """Create every directory in this layout and return ``self``."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.results_dir,
            self.figures_dir,
        ):
            ensure_dir(directory)
        return self


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    instance: InstanceConfig = InstanceConfig()
    qubo: QuboConfig = QuboConfig()
    qaoa: QaoaConfig = QaoaConfig()
    backend: BackendConfig = BackendConfig()
    output: OutputConfig = OutputConfig()
    source_file: Optional[Path] = None

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        if not isinstance(data, Mapping):
            raise ConfigError(
                f"Top-level config must be a mapping, got {type(data).__name__}"
            )
        known = {f.name for f in fields(cls)} - {"source_file"}
        unknown = sorted(set(data) - known)
        if unknown:
            _LOG.warning(
                "Ignoring unrecognised top-level config section(s): %s",
                ", ".join(unknown),
            )
        return cls(
            project=ProjectConfig.from_dict(_section(data, "project")),
            data=DataConfig.from_dict(_section(data, "data")),
            instance=InstanceConfig.from_dict(_section(data, "instance")),
            qubo=QuboConfig.from_dict(_section(data, "qubo")),
            qaoa=QaoaConfig.from_dict(_section(data, "qaoa")),
            backend=BackendConfig.from_dict(_section(data, "backend")),
            output=OutputConfig.from_dict(_section(data, "output")),
        )

    # -- derived ------------------------------------------------------------
    def build_paths(self, *, create: bool = False) -> Paths:
        """Resolve all configured directories against the project root."""
        root = find_project_root()

        def resolve(rel: str) -> Path:
            candidate = Path(rel).expanduser()
            return candidate if candidate.is_absolute() else (root / candidate)

        raw_dir = resolve(self.data.raw_dir)
        layout = Paths(
            root=root,
            raw_dir=raw_dir,
            processed_dir=resolve(self.data.processed_dir),
            results_dir=resolve(self.output.results_dir),
            figures_dir=resolve(self.output.figures_dir),
            graph_file=raw_dir / self.data.graph_filename,
        )
        return layout.ensure() if create else layout

    def with_overrides(self, **section_kwargs: Any) -> "Config":
        """Return a copy with whole sections replaced.

        >>> cfg = Config()
        >>> deeper = cfg.with_overrides(qaoa=replace(cfg.qaoa, reps=4))
        >>> deeper.qaoa.reps
        4
        """
        return replace(self, **section_kwargs)

    def summary(self) -> str:
        """A compact multi-line description for logs and result headers."""
        lines = [
            f"project        : {self.project.name} (seed={self.project.seed})",
            f"study area     : {self.data.bbox.describe()}",
            f"network        : {self.data.network_type}, simplify={self.data.simplify}",
            f"instance       : n_nodes={self.instance.n_nodes}, "
            f"incidents={self.instance.n_incidents}, "
            f"ambulances={self.instance.n_ambulances}, "
            f"sampling={self.instance.sampling}, weight={self.instance.weight}",
            f"qubo           : penalty_scale={self.qubo.penalty_scale}, "
            f"normalise={self.qubo.normalise}",
            f"qaoa           : p={self.qaoa.reps}, optimizer={self.qaoa.optimizer}, "
            f"maxiter={self.qaoa.maxiter}, shots={self.qaoa.shots}",
            f"backend        : {self.backend.provider}/{self.backend.method}, "
            f"noise={self.backend.noise_model or 'ideal'}, "
            f"opt_level={self.backend.optimization_level}",
        ]
        if self.source_file is not None:
            lines.insert(0, f"config file    : {self.source_file}")
        return "\n".join(lines)


def load_config(path: Optional[Path | str] = None) -> Config:
    """Load a :class:`Config` from YAML.

    Parameters
    ----------
    path:
        Explicit path to a YAML file. When ``None``, ``config/default.yaml``
        under the project root is used if it exists; otherwise the built-in
        defaults are returned unchanged.
    """
    if path is None:
        candidate = find_project_root() / "config" / "default.yaml"
        if not candidate.is_file():
            _LOG.info("No config file found; using built-in defaults.")
            return Config()
        resolved = candidate
    else:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            # Try the CWD first (natural for a hand-typed CLI argument),
            # then fall back to a path relative to the project root.
            if not resolved.is_file():
                resolved = find_project_root() / resolved
        if not resolved.is_file():
            raise ConfigError(f"Config file not found: {path}")

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {resolved}: {exc}") from exc

    if raw is None:
        raw = {}

    config = Config.from_dict(raw)
    config = replace(config, source_file=resolved)
    _LOG.debug("Loaded configuration from %s", resolved)
    return config
