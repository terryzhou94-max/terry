"""Command line entry point for the Jialing River forecasting workflow."""
from __future__ import annotations

import argparse
from pathlib import Path

from jialing_model.pipeline import ForecastPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jialing River flow forecasting")
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument(
        "--iterations", type=int, default=2000, help="Calibration iterations per subbasin"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = ForecastPipeline(args.config)
    metrics = pipeline.run(iterations=args.iterations)
    for basin, stats in metrics.items():
        print(f"{basin}: " + ", ".join(f"{k}={v:.3f}" for k, v in stats.items()))


if __name__ == "__main__":
    main()

