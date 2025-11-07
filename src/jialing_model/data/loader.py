"""Data loading utilities for the Jialing River flow forecasting project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:  # Optional geopandas for vector data (soil, subbasins)
    import geopandas as gpd
except Exception:  # pragma: no cover - geopandas is optional
    gpd = None  # type: ignore

try:  # Optional rasterio for DEM and slope rasters
    import rasterio
except Exception:  # pragma: no cover - rasterio is optional
    rasterio = None  # type: ignore

from ..config import ProjectConfig, SubbasinConfig


@dataclass
class SubbasinData:
    """Holds data describing a subbasin."""

    config: SubbasinConfig
    features: pd.DataFrame

    @property
    def basin_id(self) -> str:
        return self.config.name


class DataRepository:
    """Container class bundling all datasets."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.dem = self._load_raster(config.dem.path)
        self.slope = self._load_raster(config.slope.path)
        self.soil = self._load_vector(config.soil.path)
        self.meteorology = self._load_tabular(config.meteorology.path)
        self.discharge = self._load_tabular(config.discharge.path)
        self.subbasins = {
            name: SubbasinData(cfg, self._load_subbasin(cfg))
            for name, cfg in config.subbasins.items()
        }

    def _load_tabular(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".txt"}:
            df = pd.read_csv(path)
        elif suffix in {".parquet"}:
            df = pd.read_parquet(path)
        elif suffix in {".feather"}:
            df = pd.read_feather(path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported tabular file: {path}")
        return df

    def _load_vector(self, path: Path):  # type: ignore[override]
        if gpd is None:
            raise RuntimeError(
                "geopandas is required to load vector datasets (soil/subbasin shapefiles)."
            )
        return gpd.read_file(path)

    def _load_raster(self, path: Path):
        if rasterio is None:
            raise RuntimeError("rasterio is required to load raster datasets (DEM/slope).")
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
        return data, transform

    def _load_subbasin(self, cfg: SubbasinConfig) -> pd.DataFrame:
        df = self._load_tabular(cfg.dataset.path)
        if cfg.dataset.id_field and cfg.dataset.id_field in df.columns:
            df = df.set_index(cfg.dataset.id_field)
        return df

    def iter_subbasins(self) -> Iterable[SubbasinData]:
        return self.subbasins.values()

    def get_spatial_context(self, lat: float, lon: float) -> Dict[str, float | str]:
        """Extract DEM and slope statistics around a coordinate."""

        dem, transform = self.dem
        slope, _ = self.slope
        row, col = self._coord_to_index(transform, lon, lat)
        window = 3  # simple 3x3 window for stats
        r0 = max(row - window, 0)
        r1 = min(row + window + 1, dem.shape[0])
        c0 = max(col - window, 0)
        c1 = min(col + window + 1, dem.shape[1])
        dem_patch = dem[r0:r1, c0:c1]
        slope_patch = slope[r0:r1, c0:c1]
        return {
            "dem_mean": float(np.nanmean(dem_patch)),
            "dem_std": float(np.nanstd(dem_patch)),
            "slope_mean": float(np.nanmean(slope_patch)),
            "slope_std": float(np.nanstd(slope_patch)),
        }

    @staticmethod
    def _coord_to_index(transform, lon: float, lat: float) -> Tuple[int, int]:
        col, row = ~transform * (lon, lat)
        return int(row), int(col)

