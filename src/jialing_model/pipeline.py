"""End-to-end pipeline for the Jialing River forecasting project."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import ProjectConfig
from .data.loader import DataRepository
from .data.preprocess import (
    SubbasinTimeseriesBuilder,
    compute_pet,
    enrich_subbasins,
)
from .model.hydrology import LinearReservoirModel, LinearReservoirParams, RandomSearchCalibrator
from .model.parameterization import derive_parameter_bounds
from .utils.progress import ProgressPrinter


class ForecastPipeline:
    """Coordinates data preparation, model calibration and simulation."""

    def __init__(self, config_path: str | Path) -> None:
        self.config = ProjectConfig.load(config_path)
        self.repo = DataRepository(self.config)
        self.pet_series = self._compute_pet()
        self.subbasin_timeseries = SubbasinTimeseriesBuilder(self.repo, self.config).build()
        self.enriched_subbasins = enrich_subbasins(self.repo)

    def _compute_pet(self) -> pd.Series:
        meta = self.config.meteorology.metadata
        required = [
            "time_field",
            "temp_max_field",
            "temp_min_field",
            "radiation_field",
            "wind_field",
            "humidity_field",
        ]
        for key in required:
            if key not in meta and key not in {"time_field"}:
                raise ValueError(f"Meteorology metadata must define '{key}'.")
        time_field = meta.get("time_field", self.config.meteorology.time_field)
        if time_field is None:
            raise ValueError("time_field must be provided for meteorology dataset.")
        pet = compute_pet(
            self.repo.meteorology,
            time_field,
            (meta["temp_max_field"], meta["temp_min_field"]),
            meta["radiation_field"],
            meta["wind_field"],
            meta["humidity_field"],
        )
        self.repo.meteorology = self.repo.meteorology.set_index(time_field)
        self.repo.meteorology["PET"] = pet
        self.repo.meteorology.reset_index(inplace=True)
        return pet

    def calibrate_subbasins(self, iterations: int = 2000) -> Dict[str, Dict[str, float]]:
        meta = self.config.meteorology.metadata
        precip_field = meta.get("precip_field")
        if precip_field is None:
            raise ValueError("Meteorology metadata must define 'precip_field'.")
        discharge_fields = self.config.discharge.metadata
        results: Dict[str, Dict[str, float]] = {}
        printer = ProgressPrinter(len(self.subbasin_timeseries), prefix="Calibrating ")
        for idx, (subbasin, ts) in enumerate(self.subbasin_timeseries.items(), 1):
            discharge_field = discharge_fields.get(subbasin)
            if discharge_field is None:
                raise ValueError(f"Discharge metadata must map subbasin '{subbasin}' to a column name.")
            forcings = ts[[precip_field, "PET"]].rename(columns={precip_field: "precip"})
            observation = ts[discharge_field]
            bounds = derive_parameter_bounds(
                self.repo.subbasins[subbasin].config,
                self.enriched_subbasins.get(subbasin, self.repo.subbasins[subbasin].features),
            )
            calibrator = RandomSearchCalibrator(iterations=iterations, bounds=bounds)
            metrics_holder: Dict[str, float] = {}

            def progress_cb(current: int, total: int, metrics: Dict[str, float]) -> None:
                metrics_holder.update(metrics)
                printer.update((idx - 1) + current / total, metrics_holder)

            params, metrics = calibrator.calibrate(
                forcings,
                observation,
                precipitation_field="precip",
                pet_field="PET",
                progress_cb=progress_cb,
            )
            results[subbasin] = {
                **metrics,
                **params.__dict__,
            }
            printer.update(idx, metrics)
        return results

    def simulate(self, params: Dict[str, LinearReservoirParams]) -> Dict[str, pd.Series]:
        meta = self.config.meteorology.metadata
        precip_field = meta.get("precip_field")
        simulations: Dict[str, pd.Series] = {}
        for subbasin, ts in self.subbasin_timeseries.items():
            forcings = ts[[precip_field, "PET"]].rename(columns={precip_field: "precip"})
            model = LinearReservoirModel(params[subbasin])
            simulations[subbasin] = model.run(forcings, "precip", "PET")
        return simulations

    def export_results(
        self,
        metrics: Dict[str, Dict[str, float]],
        simulations: Dict[str, pd.Series],
    ) -> None:
        metrics_df = pd.DataFrame(metrics).T
        metrics_path = Path(self.config.output_dir) / "metrics.csv"
        metrics_df.to_csv(metrics_path)
        for subbasin, series in simulations.items():
            path = Path(self.config.output_dir) / f"{subbasin}_simulation.csv"
            series.to_csv(path, header=True)

    def run(self, iterations: int = 2000) -> Dict[str, Dict[str, float]]:
        metrics = self.calibrate_subbasins(iterations=iterations)
        params = {
            subbasin: LinearReservoirParams(
                runoff_coeff=val["runoff_coeff"],
                baseflow_coeff=val["baseflow_coeff"],
                percolation_rate=val["percolation_rate"],
                routing_coefficient=val["routing_coefficient"],
            )
            for subbasin, val in metrics.items()
        }
        simulations = self.simulate(params)
        self.export_results(metrics, simulations)
        return metrics


__all__ = ["ForecastPipeline"]

