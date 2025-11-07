"""PET calculation utilities based on meteorological forcing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SOLAR_CONSTANT = 0.0820  # MJ m-2 min-1
GAMMA = 0.066  # psychrometric constant (kPa °C-1)


@dataclass
class PETConfig:
    method: str
    output_path: Path


class PETCalculator:
    def __init__(self, config: PETConfig):
        self.config = config

    def compute(self, forcing: pd.DataFrame, datetime_col: str, location_lat: float) -> pd.DataFrame:
        LOGGER.info("Computing PET using %s method", self.config.method)
        method = self.config.method.lower()
        if method == "hargreaves":
            pet = _hargreaves_pet(forcing, datetime_col, location_lat)
        elif method == "penman_monteith":
            pet = _penman_monteith_pet(forcing, datetime_col, location_lat)
        else:
            raise ValueError(f"Unsupported PET method: {self.config.method}")
        result = forcing[[datetime_col]].copy()
        result = result.rename(columns={datetime_col: "datetime"})
        result["pet_mm"] = pet
        return result

    def save(self, pet_df: pd.DataFrame) -> Path:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        pet_df.to_csv(self.config.output_path, index=False)
        return self.config.output_path


def _hargreaves_pet(forcing: pd.DataFrame, datetime_col: str, lat: float) -> np.ndarray:
    doy = forcing[datetime_col].dt.dayofyear.to_numpy()
    t_min = forcing[[col for col in forcing.columns if col.lower().startswith("tmin") or "min" in col.lower()][0]].to_numpy()
    t_max = forcing[[col for col in forcing.columns if col.lower().startswith("tmax") or "max" in col.lower()][0]].to_numpy()
    temp_mean = (t_min + t_max) / 2
    ra = _extraterrestrial_radiation(lat, doy)
    pet = 0.0023 * ra * (temp_mean + 17.8) * np.sqrt(np.maximum(t_max - t_min, 0))
    return pet


def _penman_monteith_pet(forcing: pd.DataFrame, datetime_col: str, lat: float) -> np.ndarray:
    doy = forcing[datetime_col].dt.dayofyear.to_numpy()
    t_min = forcing[[col for col in forcing.columns if "min" in col.lower()][0]].to_numpy()
    t_max = forcing[[col for col in forcing.columns if "max" in col.lower()][0]].to_numpy()
    temp_mean = (t_min + t_max) / 2
    delta = 4098 * (0.6108 * np.exp((17.27 * temp_mean) / (temp_mean + 237.3))) / (temp_mean + 237.3) ** 2
    ra = _extraterrestrial_radiation(lat, doy)
    rs = forcing[[col for col in forcing.columns if "solar" in col.lower() or "radiation" in col.lower()][0]].to_numpy()
    u2 = forcing[[col for col in forcing.columns if "wind" in col.lower()][0]].to_numpy()
    rh = forcing[[col for col in forcing.columns if "hum" in col.lower()][0]].to_numpy()
    es = 0.6108 * np.exp((17.27 * temp_mean) / (temp_mean + 237.3))
    ea = es * (rh / 100.0)
    rn = 0.77 * rs
    g = 0  # daily time step assumption
    pet = (
        0.408 * delta * (rn - g) + GAMMA * (900.0 / (temp_mean + 273.0)) * u2 * (es - ea)
    ) / (delta + GAMMA * (1 + 0.34 * u2))
    return np.maximum(pet, 0)


def _extraterrestrial_radiation(lat: float, doy: np.ndarray) -> np.ndarray:
    lat_rad = np.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    delta = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    ra = (
        24 * 60 / np.pi * SOLAR_CONSTANT * dr * (
            ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )
    )
    return ra


__all__ = ["PETConfig", "PETCalculator"]
