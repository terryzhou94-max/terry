"""Hydrological performance metrics."""
from __future__ import annotations

import numpy as np


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[mask]
    obs = obs[mask]
    if obs.size == 0:
        return float("nan")
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return float("nan")
    return 1 - np.sum((sim - obs) ** 2) / denom


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta efficiency."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[mask]
    obs = obs[mask]
    if obs.size == 0:
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-6)
    beta = (np.mean(sim) + 1e-6) / (np.mean(obs) + 1e-6)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

