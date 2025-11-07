"""Data preprocessing utilities, including PET derivation."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from .loader import DataRepository, SubbasinData

SIGMA = 4.903e-9  # Stefan-Boltzmann constant (MJ K-4 m-2 day-1)
LAMBDA = 2.45  # latent heat of vaporization (MJ kg-1)


def compute_pet(met: pd.DataFrame, time_field: str, temp_fields: Tuple[str, str],
                radiation_field: str, wind_field: str, humidity_field: str) -> pd.Series:
    """Compute PET using the FAO-56 Penman-Monteith formulation."""

    df = met.copy()
    df["t_mean"] = df[list(temp_fields)].mean(axis=1)
    df["delta"] = 4098 * (0.6108 * np.exp((17.27 * df["t_mean"]) / (df["t_mean"] + 237.3))) / (
        (df["t_mean"] + 237.3) ** 2
    )
    df["gamma"] = 0.665e-3 * df.get("pressure", 101.3)
    df["es"] = 0.6108 * np.exp((17.27 * df[list(temp_fields)].mean(axis=1)) / (df["t_mean"] + 237.3))
    df["ea"] = df[humidity_field]
    df["net_rad"] = df[radiation_field]
    df["pet"] = (
        0.408 * df["delta"] * df["net_rad"]
        + df["gamma"] * (900 / (df["t_mean"] + 273)) * df[wind_field] * (df["es"] - df["ea"])
    ) / (df["delta"] + df["gamma"] * (1 + 0.34 * df[wind_field]))
    df = df.set_index(time_field)
    return df["pet"].rename("PET")


def harmonise_timeseries(repo: DataRepository, config: ProjectConfig) -> pd.DataFrame:
    """Merge meteorological and discharge data on a common timeline."""

    met = repo.meteorology
    discharge = repo.discharge
    time_field = config.meteorology.time_field or config.discharge.time_field
    if time_field is None:
        raise ValueError("Time field must be defined in configuration for meteorology or discharge.")
    met = met.set_index(time_field)
    discharge = discharge.set_index(config.discharge.time_field or time_field)
    df = met.join(discharge, how="inner", lsuffix="_met", rsuffix="_q")
    return df


def enrich_subbasins(repo: DataRepository) -> Dict[str, pd.DataFrame]:
    """Enrich subbasin dataframes with DEM, slope and soil statistics."""

    enriched: Dict[str, pd.DataFrame] = {}
    for sub in repo.iter_subbasins():
        df = sub.features.copy()
        cfg = sub.config
        lat = cfg.lat_field or cfg.extra_fields.get("lat_field")
        lon = cfg.lon_field or cfg.extra_fields.get("lon_field")
        if lat and lon and lat in df.columns and lon in df.columns:
            ctx = df[[lat, lon]].apply(
                lambda row: repo.get_spatial_context(float(row[lat]), float(row[lon])), axis=1
            )
            ctx_df = pd.DataFrame(list(ctx))
            df = pd.concat([df.reset_index(drop=True), ctx_df], axis=1)
        enriched[sub.basin_id] = df
    return enriched


class SubbasinTimeseriesBuilder:
    """Builds aligned timeseries for each subbasin."""

    def __init__(self, repo: DataRepository, config: ProjectConfig) -> None:
        self.repo = repo
        self.config = config
        self.combined = harmonise_timeseries(repo, config)

    def build(self) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for sub in self.repo.iter_subbasins():
            ts = self.combined.copy()
            cfg = sub.config
            if cfg.dataset.value_fields:
                ts = ts.join(
                    sub.features[cfg.dataset.value_fields],
                    how="left",
                )
            result[sub.basin_id] = ts.dropna()
        return result

