# Jialing River Physical Flow Forecasting

This repository provides a modular Python implementation of a physical flow forecasting
pipeline for the Jialing River basin. The project reads configuration-driven inputs
(Digital Elevation Model, slope, soil, land use, meteorological forcing, ARCSWAT
sub-basin delineations, and observed discharge at four gauges) and calibrates a
semi-distributed conceptual model with the goal of maximizing the NSE or KGE metrics.

## Directory Structure

```
config/                 # YAML configuration describing data sources and calibration options
jialing_model/
  data/                 # Data loading, sub-basin assembly, PET utilities
  models/               # Physical model, calibration algorithm, evaluation metrics
  utils/                # Logging helpers and progress reporting
outputs/                # Calibration artifacts (created at runtime)
cache/                  # Derived datasets such as PET (created at runtime)
```

## Key Features

- Configuration-driven data ingestion with automatic column detection for heterogeneous CSV files.
- PET generation from meteorological forcing using Hargreaves or Penman-Monteith methods.
- Sub-basin metadata integration (area, DEM-derived slope, soil, land use, ARCSWAT routing).
- Evolutionary calibration per sub-basin with progress bar feedback and configurable objectives (NSE/KGE).
- Modular components organized for PyCharm execution with a concise `main.py` orchestration entry point.

## Running the Pipeline

1. Populate the data directories referenced in `config/config.yaml` with the
   corresponding DEM, slope, soil, land-use, ARCSWAT, discharge, and meteorological
   datasets. Update aliases if the CSV headers differ.
2. (Optional) Adjust calibration bounds, population size, or objective in the
   configuration file.
3. Execute the pipeline from the repository root:

   ```bash
   python -m jialing_model.main
   ```

   The script prints a textual progress bar showing calibration status and
   writes calibrated parameters to `outputs/calibrated_parameters.csv`.

## Extensibility

The project is structured to ease substitution of alternative routing schemes,
objective functions, or optimization strategies. Each module is unit-test friendly
and relies on `pandas`, `numpy`, and the Python standard library.
