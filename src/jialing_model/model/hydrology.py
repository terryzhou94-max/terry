"""Hydrological model implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .metrics import kge, nse


@dataclass
class LinearReservoirParams:
    """Parameter set for the linear reservoir model."""

    runoff_coeff: float
    baseflow_coeff: float
    percolation_rate: float
    routing_coefficient: float

    @staticmethod
    def bounds() -> Dict[str, Tuple[float, float]]:
        return {
            "runoff_coeff": (0.1, 1.0),
            "baseflow_coeff": (0.01, 0.5),
            "percolation_rate": (0.0, 2.0),
            "routing_coefficient": (0.1, 5.0),
        }

    @staticmethod
    def from_vector(vec: np.ndarray) -> "LinearReservoirParams":
        keys = list(LinearReservoirParams.bounds().keys())
        return LinearReservoirParams(**{k: float(v) for k, v in zip(keys, vec)})


class LinearReservoirModel:
    """Simple conceptual rainfall-runoff model."""

    def __init__(self, params: LinearReservoirParams) -> None:
        self.params = params

    def run(self, forcings: pd.DataFrame, precipitation_field: str, pet_field: str) -> pd.Series:
        soil_moisture = 0.0
        groundwater = 0.0
        discharge = []
        for row in forcings.itertuples():
            precip = getattr(row, precipitation_field)
            pet = getattr(row, pet_field)
            net_precip = max(precip - pet, 0.0)
            infiltration = self.params.runoff_coeff * net_precip
            recharge = self.params.percolation_rate * soil_moisture
            baseflow = self.params.baseflow_coeff * groundwater
            quickflow = self.params.routing_coefficient * infiltration
            groundwater = groundwater + recharge - baseflow
            soil_moisture = soil_moisture + net_precip - infiltration - recharge
            soil_moisture = max(soil_moisture, 0.0)
            total_flow = quickflow + baseflow
            discharge.append(total_flow)
        return pd.Series(discharge, index=forcings.index, name="simulated_discharge")

    def evaluate(self, forcings: pd.DataFrame, observation: pd.Series,
                 precipitation_field: str, pet_field: str) -> Dict[str, float]:
        sim = self.run(forcings, precipitation_field, pet_field)
        obs = observation.loc[sim.index]
        return {
            "NSE": nse(sim.to_numpy(), obs.to_numpy()),
            "KGE": kge(sim.to_numpy(), obs.to_numpy()),
        }


class RandomSearchCalibrator:
    """Random search calibration for the reservoir model."""

    def __init__(
        self,
        iterations: int = 5000,
        seed: int | None = None,
        bounds: Dict[str, Tuple[float, float]] | None = None,
    ) -> None:
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)
        self.bounds = bounds or LinearReservoirParams.bounds()

    def sample_params(self) -> np.ndarray:
        vec = []
        for low, high in self.bounds.values():
            vec.append(self.rng.uniform(low, high))
        return np.array(vec)

    def calibrate(
        self,
        forcings: pd.DataFrame,
        observation: pd.Series,
        precipitation_field: str,
        pet_field: str,
        progress_cb=None,
    ) -> Tuple[LinearReservoirParams, Dict[str, float]]:
        best_score = -np.inf
        best_params = None
        best_metrics = {}
        bounds = list(self.bounds.items())
        keys = [key for key, _ in bounds]
        for i in range(1, self.iterations + 1):
            vec = np.array([self.rng.uniform(low, high) for _, (low, high) in bounds])
            params = LinearReservoirParams(**{k: float(v) for k, v in zip(keys, vec)})
            model = LinearReservoirModel(params)
            metrics = model.evaluate(forcings, observation, precipitation_field, pet_field)
            score = 0.5 * (metrics["NSE"] + metrics["KGE"])
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
            if progress_cb:
                progress_cb(i, self.iterations, best_metrics)
        assert best_params is not None
        return best_params, best_metrics

