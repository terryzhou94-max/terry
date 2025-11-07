"""Logging utilities for the Jialing River hydrological model."""

from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure logging with a sensible default format.

    Parameters
    ----------
    level:
        Logging level for the root logger.
    log_file:
        Optional path to a log file. When provided, logs will be duplicated to the
        file in addition to the console.
    """

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(level=level, format=log_format, datefmt=date_format, handlers=handlers)


__all__ = ["configure_logging"]
