"""Flexible dataset loading utilities with automatic column detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

from .config_loader import DataSourceConfig

LOGGER = logging.getLogger(__name__)


class DatasetLoader:
    """Loads datasets described in the YAML configuration file."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def load_table(self, config: DataSourceConfig, parse_dates: Optional[str | Iterable[str]] = None) -> pd.DataFrame:
        path = config.path(self.data_root)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        LOGGER.info("Loading table %s", path)
        df = pd.read_csv(path)
        if parse_dates:
            df = _parse_dates(df, parse_dates)
        return df

    def map_columns(self, df: pd.DataFrame, config: DataSourceConfig, alias_mapping: Mapping[str, Iterable[str]]) -> Dict[str, str]:
        """Return actual column names based on alias definitions."""
        column_map: Dict[str, str] = {}
        for key, aliases in alias_mapping.items():
            actual = _find_column(df.columns, aliases)
            if actual is None:
                raise KeyError(
                    f"Could not identify column for '{key}' using aliases {aliases} in dataset '{config.name}'"
                )
            column_map[key] = actual
        return column_map


def _parse_dates(df: pd.DataFrame, columns: str | Iterable[str]) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        df[col] = pd.to_datetime(df[col])
    return df


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    normalized = {col.lower(): col for col in columns}
    for alias in aliases:
        alias_lower = alias.lower()
        if alias_lower in normalized:
            return normalized[alias_lower]
    # try fuzzy contains search
    for alias in aliases:
        alias_lower = alias.lower()
        for key, original in normalized.items():
            if alias_lower in key:
                return original
    return None


def infer_spatial_fields(df: pd.DataFrame) -> Dict[str, str]:
    """Try to locate latitude and longitude columns using heuristics."""
    candidates = {
        "latitude": ["lat", "latitude", "y", "ycoord"],
        "longitude": ["lon", "longitude", "x", "xcoord"],
        "elevation": ["elev", "elevation", "z", "height"],
    }
    return {
        key: column
        for key, column in (
            (key, _find_column(df.columns, aliases))
            for key, aliases in candidates.items()
        )
        if column is not None
    }


def fill_missing_with_group_mean(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    grouped = df.groupby(group_column)
    return grouped.transform(lambda x: x.fillna(x.mean()))


__all__ = [
    "DatasetLoader",
    "infer_spatial_fields",
    "fill_missing_with_group_mean",
]
