"""Configuration management for the Jialing River flow forecasting project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    """Describes a dataset location and optional schema hints."""

    path: Path
    time_field: Optional[str] = None
    value_fields: Optional[List[str]] = None
    id_field: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(root: Path, raw: Dict[str, str | List[str] | None]) -> "DatasetConfig":
        data = {**raw}
        path = Path(data.pop("path"))
        if not path.is_absolute():
            path = root / path
        return DatasetConfig(
            path=path,
            time_field=data.pop("time_field", None),
            value_fields=data.pop("value_fields", None),
            id_field=data.pop("id_field", None),
            metadata={k: str(v) for k, v in data.items()},
        )


@dataclass
class SubbasinConfig:
    """Configuration for a single subbasin."""

    name: str
    dataset: DatasetConfig
    area_field: Optional[str] = None
    elevation_field: Optional[str] = None
    slope_field: Optional[str] = None
    lat_field: Optional[str] = None
    lon_field: Optional[str] = None
    upstream_field: Optional[str] = None
    downstream_field: Optional[str] = None
    soil_field: Optional[str] = None
    extra_fields: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(root: Path, name: str, raw: Dict[str, str]) -> "SubbasinConfig":
        dataset = DatasetConfig.from_dict(root, raw["dataset"])
        keys = {
            "area_field",
            "elevation_field",
            "slope_field",
            "lat_field",
            "lon_field",
            "upstream_field",
            "downstream_field",
            "soil_field",
        }
        extras = {k: v for k, v in raw.items() if k not in keys and k != "dataset"}
        return SubbasinConfig(
            name=name,
            dataset=dataset,
            area_field=raw.get("area_field"),
            elevation_field=raw.get("elevation_field"),
            slope_field=raw.get("slope_field"),
            lat_field=raw.get("lat_field"),
            lon_field=raw.get("lon_field"),
            upstream_field=raw.get("upstream_field"),
            downstream_field=raw.get("downstream_field"),
            soil_field=raw.get("soil_field"),
            extra_fields=extras,
        )


@dataclass
class ProjectConfig:
    """Top level configuration."""

    dem: DatasetConfig
    slope: DatasetConfig
    soil: DatasetConfig
    subbasins: Dict[str, SubbasinConfig]
    meteorology: DatasetConfig
    discharge: DatasetConfig
    output_dir: Path

    @staticmethod
    def load(path: str | Path) -> "ProjectConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        root = config_path.parent
        dem = DatasetConfig.from_dict(root, raw["dem"])
        slope = DatasetConfig.from_dict(root, raw["slope"])
        soil = DatasetConfig.from_dict(root, raw["soil"])
        meteorology = DatasetConfig.from_dict(root, raw["meteorology"])
        discharge = DatasetConfig.from_dict(root, raw["discharge"])
        subbasins = {
            name: SubbasinConfig.from_dict(root, name, cfg)
            for name, cfg in raw["subbasins"].items()
        }
        output_dir = Path(raw.get("output_dir", root / "outputs"))
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return ProjectConfig(
            dem=dem,
            slope=slope,
            soil=soil,
            subbasins=subbasins,
            meteorology=meteorology,
            discharge=discharge,
            output_dir=output_dir,
        )

