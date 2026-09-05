"""Configuration layer: validation, YAML loading, overrides, path resolution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from qroute.config import BBox, Config, DataConfig, InstanceConfig, QaoaConfig, load_config
from qroute.exceptions import ConfigError


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------
class TestBBox:
    def test_defaults_are_a_valid_dhaka_box(self) -> None:
        bbox = BBox()
        assert bbox.south < bbox.north
        assert bbox.west < bbox.east
        latitude, longitude = bbox.center
        # Dhanmondi. If someone edits the defaults to another city, this fails
        # loudly rather than silently downloading the wrong network.
        assert 23.7 < latitude < 23.8
        assert 90.3 < longitude < 90.4

    def test_inverted_latitudes_are_rejected(self) -> None:
        with pytest.raises(ConfigError, match="latitudes invalid"):
            BBox(north=23.70, south=23.75)

    def test_inverted_longitudes_are_rejected(self) -> None:
        with pytest.raises(ConfigError, match="longitudes invalid"):
            BBox(west=90.40, east=90.36)

    def test_equal_edges_are_rejected(self) -> None:
        # A zero-height box would download nothing and then fail much later.
        with pytest.raises(ConfigError):
            BBox(north=23.75, south=23.75)

    def test_osmnx_ordering_is_west_south_east_north(self) -> None:
        bbox = BBox(north=1.0, south=-1.0, east=2.0, west=-2.0)
        assert bbox.as_osmnx_bbox == (-2.0, -1.0, 2.0, 1.0)

    def test_size_shrinks_with_latitude(self) -> None:
        """A degree of longitude is shorter away from the equator."""
        equator = BBox(north=0.1, south=-0.1, east=0.1, west=-0.1)
        dhaka = BBox(north=23.85, south=23.65, east=90.45, west=90.25)
        _, equator_width = equator.approx_size_km()
        _, dhaka_width = dhaka.approx_size_km()
        assert dhaka_width < equator_width
        # cos(23.75 deg) ~ 0.915
        assert dhaka_width == pytest.approx(equator_width * 0.915, rel=0.01)


# ---------------------------------------------------------------------------
# Section validation
# ---------------------------------------------------------------------------
class TestSectionValidation:
    def test_graph_filename_must_be_graphml(self) -> None:
        with pytest.raises(ConfigError, match="graphml"):
            DataConfig(graph_filename="dhaka.json")

    def test_negative_speed_rejected(self) -> None:
        with pytest.raises(ConfigError):
            DataConfig(default_speed_kph=0.0)

    def test_instance_sizes_must_be_sane(self) -> None:
        with pytest.raises(ConfigError):
            InstanceConfig(n_nodes=1)

    def test_qaoa_reps_must_be_positive(self) -> None:
        with pytest.raises(ConfigError):
            QaoaConfig(reps=0)

    def test_unknown_keys_are_dropped_not_fatal(self) -> None:
        """A typo in the YAML should warn, not crash a long pipeline run."""
        config = Config.from_dict({"qaoa": {"reps": 3, "not_a_real_key": 7}})
        assert config.qaoa.reps == 3


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_round_trip_through_yaml(self, tmp_path: Path) -> None:
        target = tmp_path / "custom.yaml"
        target.write_text(
            "\n".join(
                [
                    "project:",
                    "  name: test-run",
                    "  seed: 7",
                    "data:",
                    "  bbox:",
                    "    north: 23.76",
                    "    south: 23.74",
                    "    east: 90.39",
                    "    west: 90.37",
                    "  network_type: drive",
                    "instance:",
                    "  n_nodes: 4",
                    "  weight: length",
                    "qaoa:",
                    "  reps: 3",
                    "  optimizer: SPSA",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        config = load_config(target)

        assert config.project.name == "test-run"
        assert config.project.seed == 7
        assert config.data.bbox.north == pytest.approx(23.76)
        assert config.instance.n_nodes == 4
        assert config.instance.weight == "length"
        assert config.qaoa.reps == 3
        assert config.qaoa.optimizer == "SPSA"
        assert config.source_file == target

    def test_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.yaml"
        target.write_text("", encoding="utf-8")
        assert load_config(target).qaoa.reps == Config().qaoa.reps

    def test_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_malformed_yaml_is_an_error(self, tmp_path: Path) -> None:
        target = tmp_path / "broken.yaml"
        target.write_text("qaoa:\n  reps: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Could not parse YAML"):
            load_config(target)

    def test_shipped_default_config_is_valid(self) -> None:
        """The config file in the repo must load -- it is the documented entry point."""
        root = Path(__file__).resolve().parents[1]
        candidate = root / "config" / "default.yaml"
        if not candidate.is_file():  # pragma: no cover - repo layout changed
            pytest.skip("config/default.yaml is not present")
        config = load_config(candidate)
        assert config.instance.n_nodes >= 2
        assert config.qaoa.shots > 0


# ---------------------------------------------------------------------------
# Overrides and paths
# ---------------------------------------------------------------------------
class TestOverridesAndPaths:
    def test_with_overrides_replaces_only_the_named_section(self) -> None:
        base = Config()
        deeper = base.with_overrides(qaoa=replace(base.qaoa, reps=5))
        assert deeper.qaoa.reps == 5
        assert deeper.qaoa.shots == base.qaoa.shots
        assert deeper.instance == base.instance
        assert base.qaoa.reps != 5, "the original config must not be mutated"

    def test_with_overrides_keeps_source_file(self, tmp_path: Path) -> None:
        """A CLI flag must not erase the record of which file was loaded."""
        target = tmp_path / "c.yaml"
        target.write_text("qaoa:\n  reps: 2\n", encoding="utf-8")
        config = load_config(target)
        overridden = config.with_overrides(qaoa=replace(config.qaoa, reps=4))
        assert overridden.source_file == target

    def test_absolute_directories_are_used_verbatim(self, config: Config, tmp_path: Path) -> None:
        paths = config.build_paths()
        assert paths.raw_dir == tmp_path / "raw"
        assert paths.graph_file == tmp_path / "raw" / "test_grid.graphml"
        assert not paths.raw_dir.exists(), "build_paths must not create dirs unless asked"

    def test_create_makes_every_directory(self, config: Config) -> None:
        paths = config.build_paths(create=True)
        for directory in (
            paths.raw_dir,
            paths.processed_dir,
            paths.results_dir,
            paths.figures_dir,
        ):
            assert directory.is_dir()

    def test_relative_directories_resolve_against_the_project_root(self) -> None:
        paths = Config().build_paths()
        assert paths.raw_dir.is_absolute()
        assert paths.raw_dir == paths.root / "data" / "raw"

    def test_summary_mentions_the_moving_parts(self, config: Config) -> None:
        text = config.summary()
        for expected in ("project", "study area", "instance", "qubo", "qaoa", "backend"):
            assert expected in text
