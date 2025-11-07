"""Data structures representing sub-basin metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class Subbasin:
    id: str
    area_km2: float
    mean_elevation: float
    slope: float
    latitude: float
    longitude: float
    downstream_id: Optional[str]
    soil_classes: Dict[str, float] = field(default_factory=dict)
    land_use_fractions: Dict[str, float] = field(default_factory=dict)
    arcswat_attributes: Dict[str, float] = field(default_factory=dict)


class SubbasinCollection:
    def __init__(self, subbasins: Iterable[Subbasin]):
        self._subbasins: Dict[str, Subbasin] = {sb.id: sb for sb in subbasins}
        self._validate_connectivity()

    def __len__(self) -> int:
        return len(self._subbasins)

    def __iter__(self):
        return iter(self._subbasins.values())

    def __getitem__(self, subbasin_id: str) -> Subbasin:
        return self._subbasins[subbasin_id]

    def upstream_of(self, subbasin_id: str) -> List[Subbasin]:
        return [sb for sb in self._subbasins.values() if sb.downstream_id == subbasin_id]

    def _validate_connectivity(self) -> None:
        for sb in self._subbasins.values():
            downstream = sb.downstream_id
            if downstream and downstream not in self._subbasins:
                raise ValueError(f"Subbasin {sb.id} references unknown downstream basin {downstream}")


def build_subbasins(
    subbasin_df: pd.DataFrame,
    column_map: Dict[str, str],
    soil_df: Optional[pd.DataFrame] = None,
    soil_key_column: Optional[str] = None,
    landuse_df: Optional[pd.DataFrame] = None,
    landuse_key_column: Optional[str] = None,
) -> SubbasinCollection:
    subbasins = []
    for _, row in subbasin_df.iterrows():
        sb_id = str(row[column_map["id"]])
        downstream_id = None
        downstream_key = column_map.get("downstream")
        if downstream_key is not None:
            downstream_value = row[downstream_key]
            downstream_id = str(downstream_value) if pd.notnull(downstream_value) else None
        arcswat_attributes = {}
        for key, val in column_map.items():
            if key in {"id", "downstream", "area", "elevation", "latitude", "longitude", "slope"}:
                continue
            try:
                arcswat_attributes[key] = float(row[val])
            except (TypeError, ValueError):
                continue
        soil_classes = _extract_fraction_map(soil_df, soil_key_column, sb_id) if soil_df is not None else {}
        land_use = _extract_fraction_map(landuse_df, landuse_key_column, sb_id) if landuse_df is not None else {}
        subbasins.append(
            Subbasin(
                id=sb_id,
                area_km2=float(row[column_map["area"]]),
                mean_elevation=float(row[column_map["elevation"]]),
                slope=float(row[column_map["slope"]]),
                latitude=float(row[column_map["latitude"]]),
                longitude=float(row[column_map["longitude"]]),
                downstream_id=downstream_id,
                soil_classes=soil_classes,
                land_use_fractions=land_use,
                arcswat_attributes=arcswat_attributes,
            )
        )
    return SubbasinCollection(subbasins)


def _extract_fraction_map(df: Optional[pd.DataFrame], key_column: Optional[str], key: str) -> Dict[str, float]:
    if df is None or key_column is None:
        return {}
    subset = df[df[key_column].astype(str) == str(key)]
    if subset.empty:
        return {}
    result: Dict[str, float] = {}
    for column in subset.columns:
        if column == key_column:
            continue
        value = subset.iloc[0][column]
        if pd.notnull(value):
            result[column] = float(value)
    return result


__all__ = ["Subbasin", "SubbasinCollection", "build_subbasins"]
