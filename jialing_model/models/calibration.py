"""Parameter calibration for the Jialing River physical model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

from .metrics import kge, nse
from .physical_model import PhysicalRunoffModel, SubbasinParameters
from ..data.subbasin import SubbasinCollection
from ..utils.progress import ProgressBar

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationBounds:
    storage_max: Tuple[float, float]
    baseflow_coefficient: Tuple[float, float]
    quickflow_coefficient: Tuple[float, float]
    percolation_rate: Tuple[float, float]
    soil_moisture_capacity: Tuple[float, float]

    def lower_array(self, n_subbasins: int) -> np.ndarray:
        return np.tile(
            np.array(
                [
                    self.storage_max[0],
                    self.baseflow_coefficient[0],
                    self.quickflow_coefficient[0],
                    self.percolation_rate[0],
                    self.soil_moisture_capacity[0],
                ]
            ),
            n_subbasins,
        )

    def upper_array(self, n_subbasins: int) -> np.ndarray:
        return np.tile(
            np.array(
                [
                    self.storage_max[1],
                    self.baseflow_coefficient[1],
                    self.quickflow_coefficient[1],
                    self.percolation_rate[1],
                    self.soil_moisture_capacity[1],
                ]
            ),
            n_subbasins,
        )


class EvolutionaryCalibrator:
    def __init__(
        self,
        model: PhysicalRunoffModel,
        bounds: CalibrationBounds,
        objective: str = "kge",
        population_size: int = 20,
        max_iterations: int = 100,
        convergence_tolerance: float = 1e-4,
        random_seed: int | None = None,
        progress_interval: int = 10,
    ) -> None:
        self.model = model
        self.bounds = bounds
        self.objective = objective
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.random_state = np.random.default_rng(random_seed)
        self.progress_interval = progress_interval

    def calibrate(
        self,
        subbasins: SubbasinCollection,
        observed: pd.DataFrame,
        forcing: Mapping[str, pd.DataFrame],
        datetime_col: str,
    ) -> Dict[str, SubbasinParameters]:
        population = self._initialize_population(subbasins)
        best_score = -np.inf
        best_params: Dict[str, SubbasinParameters] = {}
        progress = ProgressBar(self.max_iterations, description="Calibrating")

        for iteration in range(1, self.max_iterations + 1):
            scores, individuals = [], []
            for candidate in population:
                params = self._decode_parameters(candidate, subbasins)
                simulation = self.model.run(forcing=forcing, params=params, datetime_col=datetime_col)
                score = self._evaluate(simulation, observed)
                if np.isnan(score):
                    score = -np.inf
                scores.append(score)
                individuals.append(params)
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_params = individuals[best_idx]

            if iteration % self.progress_interval == 0 or iteration == self.max_iterations:
                remaining = max(progress.state.total - progress.state.current, 0)
                step = self.progress_interval if remaining >= self.progress_interval else remaining
                progress.update(step if step > 0 else 0, extra_message=f"Best {self.objective.upper()}: {best_score:0.4f}")

            if self._has_converged(scores):
                LOGGER.info("Calibration converged at iteration %d", iteration)
                break

            population = self._evolve(population, scores, len(subbasins))

        return best_params

    def _initialize_population(self, subbasins: SubbasinCollection) -> np.ndarray:
        pop = []
        for _ in range(self.population_size):
            individual = []
            for sb in subbasins:
                slope_factor = np.clip(sb.slope / 30.0, 0.2, 3.0)
                area_factor = np.clip(sb.area_km2 / 500.0, 0.1, 2.5)
                elevation_factor = np.clip(sb.mean_elevation / 1500.0, 0.2, 3.0)
                storage = self._sample_within_bounds(self.bounds.storage_max, base=150.0 * area_factor)
                baseflow = self._sample_within_bounds(self.bounds.baseflow_coefficient, base=0.2 / slope_factor)
                quickflow = self._sample_within_bounds(self.bounds.quickflow_coefficient, base=0.8 * slope_factor)
                percolation = self._sample_within_bounds(self.bounds.percolation_rate, base=0.5 * elevation_factor)
                soil_capacity = self._sample_within_bounds(self.bounds.soil_moisture_capacity, base=100.0 / slope_factor)
                individual.extend([storage, baseflow, quickflow, percolation, soil_capacity])
            pop.append(individual)
        return np.array(pop)

    def _decode_parameters(
        self, vector: Iterable[float], subbasins: SubbasinCollection
    ) -> Dict[str, SubbasinParameters]:
        params: Dict[str, SubbasinParameters] = {}
        iterator = iter(vector)
        for sb in subbasins:
            params[sb.id] = SubbasinParameters(
                storage_max=next(iterator),
                baseflow_coefficient=next(iterator),
                quickflow_coefficient=next(iterator),
                percolation_rate=next(iterator),
                soil_moisture_capacity=next(iterator),
            )
        return params

    def _evaluate(self, simulation: pd.DataFrame, observed: pd.DataFrame) -> float:
        score_per_station = []
        objective_func = kge if self.objective == "kge" else nse
        common_times = simulation.index.intersection(observed.index)
        for column in observed.columns:
            if column not in simulation.columns:
                continue
            score_per_station.append(
                objective_func(simulation.loc[common_times, column].to_numpy(), observed.loc[common_times, column].to_numpy())
            )
        if not score_per_station:
            return float("nan")
        return float(np.nanmean(score_per_station))

    def _has_converged(self, scores: Iterable[float]) -> bool:
        recent = list(scores)[-max(5, self.population_size // 2):]
        if len(recent) < 2:
            return False
        return float(np.std(recent)) < self.convergence_tolerance

    def _evolve(self, population: np.ndarray, scores: Iterable[float], n_subbasins: int) -> np.ndarray:
        scores = np.asarray(scores)
        ranks = np.argsort(scores)[::-1]
        elite = population[ranks[: max(2, self.population_size // 5)]]
        lower = self.bounds.lower_array(n_subbasins)
        upper = self.bounds.upper_array(n_subbasins)
        offspring = []
        while len(offspring) + len(elite) < self.population_size:
            parents = self.random_state.choice(elite, size=2, replace=True)
            crossover_point = self.random_state.integers(1, parents.shape[1] - 1)
            child = np.concatenate([
                parents[0, :crossover_point],
                parents[1, crossover_point:],
            ])
            mutation_mask = self.random_state.random(child.size) < 0.1
            mutation = self.random_state.normal(scale=0.05, size=child.size)
            child = child * (1 + mutation * mutation_mask)
            child = np.clip(child, lower, upper)
            offspring.append(child)
        new_population = np.vstack([elite, np.array(offspring)])
        return new_population

    def _sample_within_bounds(self, bounds: Tuple[float, float], base: float) -> float:
        low, high = bounds
        mean = np.clip(base, low, high)
        span = (high - low) * 0.25
        sample = self.random_state.normal(loc=mean, scale=max(span, 1e-3))
        return float(np.clip(sample, low, high))


__all__ = ["CalibrationBounds", "EvolutionaryCalibrator"]
