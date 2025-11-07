"""Performance metrics for hydrological model evaluation."""

from __future__ import annotations

import numpy as np


def nse(simulated: np.ndarray, observed: np.ndarray) -> float:
    mask = _valid_mask(simulated, observed)
    sim = simulated[mask]
    obs = observed[mask]
    if obs.size == 0:
        return float("nan")
    denominator = np.sum((obs - obs.mean()) ** 2)
    if denominator == 0:
        return float("nan")
    value = 1 - np.sum((obs - sim) ** 2) / denominator
    return float(np.clip(value, -np.inf, 1.0))


def kge(simulated: np.ndarray, observed: np.ndarray) -> float:
    mask = _valid_mask(simulated, observed)
    sim = simulated[mask]
    obs = observed[mask]
    if obs.size == 0:
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-12)
    beta = sim.mean() / (obs.mean() + 1e-12)
    value = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(np.clip(value, -np.inf, 1.0))


def _valid_mask(simulated: np.ndarray, observed: np.ndarray) -> np.ndarray:
    mask = (~np.isnan(simulated)) & (~np.isnan(observed))
    mask &= np.isfinite(simulated) & np.isfinite(observed)
    return mask


__all__ = ["nse", "kge"]
