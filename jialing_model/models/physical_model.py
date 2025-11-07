"""Conceptual rainfall-runoff model tailored for the Jialing River basin."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from ..data.subbasin import Subbasin, SubbasinCollection

LOGGER = logging.getLogger(__name__)


@dataclass
class SubbasinParameters:
    storage_max: float
    baseflow_coefficient: float
    quickflow_coefficient: float
    percolation_rate: float
    soil_moisture_capacity: float


class PhysicalRunoffModel:
    """Implements a semi-distributed conceptual model across sub-basins."""

    def __init__(self, subbasins: SubbasinCollection, routing_order: Iterable[str]):
        self.subbasins = subbasins
        self.routing_order = list(routing_order)
        self._initial_states = {
            sb.id: {
                "soil_moisture": sb.soil_classes.get("init_soil_moisture", 0.5) * 0.8 * 100,
                "groundwater": sb.arcswat_attributes.get("gw_init", 10.0),
                "surface_storage": 0.0,
            }
            for sb in subbasins
        }
        self._states = {sb_id: state.copy() for sb_id, state in self._initial_states.items()}

    def run(
        self,
        forcing: Mapping[str, pd.DataFrame],
        params: Mapping[str, SubbasinParameters],
        datetime_col: str,
    ) -> pd.DataFrame:
        self._reset_states()
        for sb_id in self.routing_order:
            if sb_id not in params:
                raise KeyError(f"Missing parameters for subbasin {sb_id}")

        datetime_index = forcing["precipitation"][datetime_col]
        result = pd.DataFrame(index=datetime_index, columns=self.routing_order, dtype=float)

        for timestamp in datetime_index:
            contributions: Dict[str, float] = {}
            for sb_id in self.routing_order:
                contributions[sb_id] = self._simulate_subbasin_step(
                    subbasin=self.subbasins[sb_id],
                    params=params[sb_id],
                    forcing=forcing,
                    timestamp=timestamp,
                    datetime_col=datetime_col,
                )
            routed = self._route(contributions)
            for sb_id, value in routed.items():
                result.loc[timestamp, sb_id] = value
        return result

    def _simulate_subbasin_step(
        self,
        subbasin: Subbasin,
        params: SubbasinParameters,
        forcing: Mapping[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        datetime_col: str,
    ) -> float:
        state = self._states[subbasin.id]
        precipitation = _get_value(forcing["precipitation"], datetime_col, timestamp, subbasin.id, fallback=0.0)
        pet = _get_value(forcing["pet"], datetime_col, timestamp, subbasin.id, fallback=0.0)
        area_factor = subbasin.area_km2 * 1e6  # convert to m^2

        soil_storage = state["soil_moisture"]
        groundwater = state["groundwater"]
        surface_storage = state["surface_storage"]

        net_precip = max(precipitation - pet, 0)
        infiltration = min(net_precip, params.soil_moisture_capacity - soil_storage)
        runoff_excess = max(net_precip - infiltration, 0)
        soil_storage += infiltration

        percolation = min(soil_storage, params.percolation_rate)
        soil_storage -= percolation
        groundwater += percolation

        quickflow = params.quickflow_coefficient * runoff_excess
        surface_storage += runoff_excess - quickflow
        surface_storage = min(surface_storage, params.storage_max)
        baseflow = params.baseflow_coefficient * groundwater
        groundwater = max(groundwater - baseflow, 0)
        surface_release = params.quickflow_coefficient * surface_storage
        surface_storage = max(surface_storage - surface_release, 0)

        total_outflow_mm = quickflow + baseflow + surface_release
        discharge_m3 = total_outflow_mm / 1000.0 * area_factor

        state["soil_moisture"] = np.clip(soil_storage, 0, params.soil_moisture_capacity)
        state["groundwater"] = max(groundwater, 0)
        state["surface_storage"] = max(surface_storage, 0)

        return discharge_m3

    def _route(self, contributions: Mapping[str, float]) -> Dict[str, float]:
        routed = dict(contributions)
        for sb_id in reversed(self.routing_order):
            downstream = self.subbasins[sb_id].downstream_id
            if downstream is None:
                continue
            routed[downstream] = routed.get(downstream, 0.0) + routed[sb_id]
        return routed

    def _reset_states(self) -> None:
        self._states = {sb_id: state.copy() for sb_id, state in self._initial_states.items()}


def _get_value(
    df: pd.DataFrame,
    datetime_col: str,
    timestamp: pd.Timestamp,
    subbasin_id: Optional[str] = None,
    fallback: float = 0.0,
) -> float:
    rows = df[df[datetime_col] == timestamp]
    if rows.empty:
        return fallback
    if subbasin_id and subbasin_id in rows.columns:
        value = rows.iloc[0][subbasin_id]
    else:
        candidates = [col for col in rows.columns if col != datetime_col]
        if not candidates:
            return fallback
        value = rows.iloc[0][candidates[0]]
    return float(value)


__all__ = ["SubbasinParameters", "PhysicalRunoffModel"]
