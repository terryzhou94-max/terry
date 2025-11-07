"""Entry point for the Jialing River basin physical flow forecasting pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from .data.config_loader import GlobalConfig, load_config
from .data.dataset_loader import DatasetLoader
from .data.pet import PETCalculator, PETConfig
from .data.subbasin import SubbasinCollection, build_subbasins
from .models.calibration import CalibrationBounds, EvolutionaryCalibrator
from .models.physical_model import PhysicalRunoffModel, SubbasinParameters
from .utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def main(config_path: str | Path = "config/config.yaml") -> None:
    configure_logging()
    config = load_config(Path(config_path))
    LOGGER.info("Loaded configuration for %s", config.basin_name)

    data_root = Path(config_path).resolve().parent.parent
    loader = DatasetLoader(data_root)

    subbasins = _load_subbasins(config, loader)
    routing = _derive_routing_order(subbasins)
    LOGGER.info("Detected %d subbasins", len(subbasins))

    forcing = _load_forcing(config, loader, subbasins)
    observed = _load_discharge(config, loader, subbasins)

    model = PhysicalRunoffModel(subbasins=subbasins, routing_order=routing)

    bounds = CalibrationBounds(
        storage_max=tuple(config.subbasin_bounds.get("storage_max", [10.0, 500.0])),
        baseflow_coefficient=tuple(config.subbasin_bounds.get("baseflow_coefficient", [0.01, 0.8])),
        quickflow_coefficient=tuple(config.subbasin_bounds.get("quickflow_coefficient", [0.1, 2.0])),
        percolation_rate=tuple(config.subbasin_bounds.get("percolation_rate", [0.001, 5.0])),
        soil_moisture_capacity=tuple(config.subbasin_bounds.get("soil_moisture_capacity", [5.0, 300.0])),
    )

    calibrator = EvolutionaryCalibrator(
        model=model,
        bounds=bounds,
        objective=config.calibration.objective,
        population_size=config.calibration.population_size,
        max_iterations=config.calibration.max_iterations,
        convergence_tolerance=config.calibration.convergence_tolerance,
        random_seed=config.calibration.random_seed,
        progress_interval=config.calibration.progress_update_interval,
    )

    best_parameters = calibrator.calibrate(
        subbasins=subbasins,
        observed=observed,
        forcing=forcing,
        datetime_col="datetime",
    )

    _export_results(best_parameters, config, subbasins)


def _load_subbasins(config: GlobalConfig, loader: DatasetLoader) -> SubbasinCollection:
    sb_config = config.data_sources["subbasins"]
    sb_df = loader.load_table(sb_config)
    required_aliases = {
        "id": sb_config.aliases("id_aliases"),
        "area": sb_config.aliases("area_aliases"),
        "elevation": sb_config.aliases("elevation_aliases"),
        "latitude": sb_config.aliases("latitude_aliases"),
        "longitude": sb_config.aliases("longitude_aliases"),
        "slope": sb_config.aliases("slope_aliases"),
    }
    column_map = loader.map_columns(sb_df, sb_config, required_aliases)
    downstream_aliases = sb_config.aliases("downstream_aliases")
    if downstream_aliases:
        try:
            column_map["downstream"] = loader.map_columns(sb_df, sb_config, {"downstream": downstream_aliases})["downstream"]
        except KeyError:
            LOGGER.warning("Could not identify downstream column in subbasin dataset")

    soil_df = None
    soil_key_column = None
    if "soil" in config.data_sources:
        soil_cfg = config.data_sources["soil"]
        soil_df = loader.load_table(soil_cfg)
        soil_key_column = loader.map_columns(
            soil_df,
            soil_cfg,
            {"key": soil_cfg.aliases("key_column_aliases")},
        )["key"]

    land_df = None
    land_key_column = None
    if "land_use" in config.data_sources:
        land_cfg = config.data_sources["land_use"]
        land_df = loader.load_table(land_cfg)
        land_key_column = loader.map_columns(
            land_df,
            land_cfg,
            {"key": land_cfg.aliases("key_column_aliases")},
        )["key"]

    subbasins = build_subbasins(
        subbasin_df=sb_df,
        column_map=column_map,
        soil_df=soil_df,
        soil_key_column=soil_key_column,
        landuse_df=land_df,
        landuse_key_column=land_key_column,
    )
    return subbasins


def _derive_routing_order(subbasins: SubbasinCollection) -> List[str]:
    visited = set()
    order: List[str] = []

    def visit(sb_id: str) -> None:
        if sb_id in visited:
            return
        visited.add(sb_id)
        downstream = subbasins[sb_id].downstream_id
        if downstream:
            visit(downstream)
        order.append(sb_id)

    for sb in subbasins:
        visit(sb.id)
    order.reverse()
    return order


def _load_forcing(config: GlobalConfig, loader: DatasetLoader, subbasins: SubbasinCollection) -> Mapping[str, pd.DataFrame]:
    forcing_cfg = config.data_sources["meteorological_forcing"]
    forcing_df = loader.load_table(forcing_cfg, parse_dates=forcing_cfg.aliases("datetime_aliases"))
    dt_column = loader.map_columns(
        forcing_df,
        forcing_cfg,
        {"datetime": forcing_cfg.aliases("datetime_aliases")},
    )["datetime"]

    precip_column = loader.map_columns(
        forcing_df,
        forcing_cfg,
        {"precip": forcing_cfg.aliases("precipitation_aliases")},
    )["precip"]

    forcing_df = forcing_df.rename(columns={dt_column: "datetime", precip_column: "precipitation"})

    pet_cfg = config.data_sources.get("pet")
    if pet_cfg:
        output_setting = pet_cfg.options.get("output_path", config.cache_directory / "pet.csv")
        output_path = Path(output_setting)
        if not output_path.is_absolute():
            output_path = config.cache_directory / output_path
        pet_config = PETConfig(
            method=str(pet_cfg.options.get("method", "hargreaves")),
            output_path=output_path,
        )
    else:
        pet_config = PETConfig(method="hargreaves", output_path=config.cache_directory / "pet.csv")

    representative_lat = np.mean([sb.latitude for sb in subbasins])
    calculator = PETCalculator(pet_config)
    pet_df = calculator.compute(forcing_df.rename(columns={"datetime": dt_column}), dt_column, representative_lat)
    calculator.save(pet_df)

    forcing_df["pet"] = pet_df["pet_mm"].values

    return {
        "precipitation": forcing_df[["datetime", "precipitation"]],
        "pet": forcing_df[["datetime", "pet"]],
    }


def _load_discharge(
    config: GlobalConfig,
    loader: DatasetLoader,
    subbasins: SubbasinCollection,
) -> pd.DataFrame:
    discharge_cfg = config.data_sources["discharge"]
    discharge_df = loader.load_table(discharge_cfg, parse_dates=discharge_cfg.aliases("datetime_aliases"))
    column_map = {}
    for field, alias_key in (
        ("datetime", "datetime_aliases"),
        ("flow", "flow_aliases"),
        ("station", "station_aliases"),
    ):
        column_map[field] = loader.map_columns(discharge_df, discharge_cfg, {field: discharge_cfg.aliases(alias_key)})[field]

    optional_fields = {"latitude": "latitude_aliases", "longitude": "longitude_aliases"}
    for field, alias_key in optional_fields.items():
        aliases = discharge_cfg.aliases(alias_key)
        if aliases:
            try:
                column_map[field] = loader.map_columns(discharge_df, discharge_cfg, {field: aliases})[field]
            except KeyError:
                LOGGER.warning("Could not map optional column %s for discharge dataset", field)

    discharge_df = discharge_df.rename(columns={column_map["datetime"]: "datetime", column_map["flow"]: "flow", column_map["station"]: "station"})
    if "latitude" in column_map and "longitude" in column_map:
        discharge_df = discharge_df.rename(columns={column_map["latitude"]: "latitude", column_map["longitude"]: "longitude"})
        discharge_df["subbasin"] = discharge_df.apply(
            lambda row: _nearest_subbasin(row.get("latitude"), row.get("longitude"), subbasins, row["station"]),
            axis=1,
        )
    else:
        discharge_df["subbasin"] = discharge_df["station"]

    pivot = discharge_df.pivot_table(index="datetime", columns="subbasin", values="flow")
    pivot.sort_index(inplace=True)
    return pivot


def _export_results(
    best_parameters: Mapping[str, SubbasinParameters],
    config: GlobalConfig,
    subbasins: SubbasinCollection,
) -> None:
    output_path = config.output_directory / "calibrated_parameters.csv"
    records = []
    for sb in subbasins:
        params = best_parameters[sb.id]
        records.append(
            {
                "subbasin": sb.id,
                "area_km2": sb.area_km2,
                "mean_elevation": sb.mean_elevation,
                "slope": sb.slope,
                "latitude": sb.latitude,
                "longitude": sb.longitude,
                "storage_max": params.storage_max,
                "baseflow_coefficient": params.baseflow_coefficient,
                "quickflow_coefficient": params.quickflow_coefficient,
                "percolation_rate": params.percolation_rate,
                "soil_moisture_capacity": params.soil_moisture_capacity,
            }
        )
    df = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved calibrated parameters to %s", output_path)


def _nearest_subbasin(
    lat: float | None,
    lon: float | None,
    subbasins: SubbasinCollection,
    fallback_station: str,
) -> str:
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return fallback_station
    distances = []
    for sb in subbasins:
        distance = (sb.latitude - lat) ** 2 + (sb.longitude - lon) ** 2
        distances.append((distance, sb.id))
    distances.sort()
    nearest = distances[0][1]
    return nearest


if __name__ == "__main__":
    main()
