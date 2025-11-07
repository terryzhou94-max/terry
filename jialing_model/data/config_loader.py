"""Configuration loading utilities for the Jialing River basin model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


@dataclass
class DataSourceConfig:
    name: str
    options: Mapping[str, Any]

    def path(self, root: Path) -> Path:
        path_value = self.options.get("path")
        if path_value is None:
            raise ValueError(f"Data source '{self.name}' is missing the 'path' entry")
        return (root / Path(path_value)).expanduser().resolve()

    def aliases(self, key: str) -> Iterable[str]:
        value = self.options.get(key, [])
        if isinstance(value, str):
            return [value]
        return list(value)


@dataclass
class CalibrationConfig:
    objective: str
    max_iterations: int
    population_size: int
    convergence_tolerance: float
    random_seed: Optional[int]
    validation_split: float
    progress_update_interval: int
    use_multiprocessing: bool


@dataclass
class GlobalConfig:
    basin_name: str
    time_step: str
    start_date: str
    end_date: str
    output_directory: Path
    cache_directory: Path
    n_workers: int
    data_sources: Dict[str, DataSourceConfig]
    subbasin_bounds: Mapping[str, Iterable[float]]
    calibration: CalibrationConfig


def load_config(path: Path | str) -> GlobalConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    general = raw.get("general", {})
    data_sources_raw = raw.get("data_sources", {})
    bounds = raw.get("subbasin_parameter_bounds", {})
    calibration_raw = raw.get("calibration", {})

    data_sources = {
        name: DataSourceConfig(name=name, options=options)
        for name, options in data_sources_raw.items()
    }

    calibration = CalibrationConfig(
        objective=calibration_raw.get("objective", "kge").lower(),
        max_iterations=int(calibration_raw.get("max_iterations", 100)),
        population_size=int(calibration_raw.get("population_size", 20)),
        convergence_tolerance=float(calibration_raw.get("convergence_tolerance", 1e-4)),
        random_seed=calibration_raw.get("random_seed"),
        validation_split=float(calibration_raw.get("validation_split", 0.2)),
        progress_update_interval=int(calibration_raw.get("progress_update_interval", 10)),
        use_multiprocessing=bool(calibration_raw.get("use_multiprocessing", False)),
    )

    root = path.parent.parent
    output_directory = (root / general.get("output_directory", "outputs")).resolve()
    cache_directory = (root / general.get("cache_directory", "cache")).resolve()

    config = GlobalConfig(
        basin_name=str(general.get("basin_name", "Jialing River")),
        time_step=str(general.get("time_step", "D")),
        start_date=str(general.get("start_date", "2000-01-01")),
        end_date=str(general.get("end_date", "2020-12-31")),
        output_directory=output_directory,
        cache_directory=cache_directory,
        n_workers=int(general.get("n_workers", 1)),
        data_sources=data_sources,
        subbasin_bounds=bounds,
        calibration=calibration,
    )

    _ensure_directories(config)
    return config


def _ensure_directories(config: GlobalConfig) -> None:
    for directory in (config.output_directory, config.cache_directory):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = ["DataSourceConfig", "CalibrationConfig", "GlobalConfig", "load_config"]
