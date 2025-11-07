"""Parameter derivation utilities per subbasin."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..config import SubbasinConfig
from .hydrology import LinearReservoirParams


def derive_parameter_bounds(config: SubbasinConfig, features: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    base = LinearReservoirParams.bounds()
    bounds = dict(base)
    if config.area_field and config.area_field in features.columns:
        area = float(np.nanmean(features[config.area_field]))
        scale = min(max(area / 1000.0, 0.5), 2.0)
        low, high = base["runoff_coeff"]
        bounds["runoff_coeff"] = (max(low / scale, 0.05), min(high * scale, 1.5))
    if config.slope_field and config.slope_field in features.columns:
        slope = float(np.nanmean(features[config.slope_field]))
        low, high = base["routing_coefficient"]
        factor = min(max(slope / 30.0, 0.3), 3.0)
        bounds["routing_coefficient"] = (low * factor, high * factor)
    if "soil" in features.columns:
        soil_types = features["soil"].astype(str).unique()
        if any("clay" in s.lower() for s in soil_types):
            low, high = base["percolation_rate"]
            bounds["percolation_rate"] = (low * 0.5, high * 0.8)
    if config.upstream_field and config.upstream_field in features.columns:
        connectivity = features[config.upstream_field].notna().sum()
        low, high = base["baseflow_coeff"]
        factor = min(max(connectivity / max(len(features), 1), 0.5), 1.5)
        bounds["baseflow_coeff"] = (low * factor, min(high * factor, 1.0))
    return bounds

