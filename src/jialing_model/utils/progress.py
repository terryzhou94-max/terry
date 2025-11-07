"""Console progress utilities."""
from __future__ import annotations

import sys
import time
from typing import Dict


class ProgressPrinter:
    """Lightweight progress bar compatible with PyCharm console."""

    def __init__(self, total: int, prefix: str = "") -> None:
        self.total = total
        self.prefix = prefix
        self.start = time.time()
        self.last_len = 0

    def update(self, current: float, metrics: Dict[str, float] | None = None) -> None:
        elapsed = time.time() - self.start
        rate = current / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - current) / rate if rate > 0 else float("inf")
        metrics_str = " " + " ".join(f"{k}:{v:.3f}" for k, v in (metrics or {}).items()) if metrics else ""
        message = (
            f"\r{self.prefix}{current}/{self.total} "
            f"({current / max(self.total, 1):.1%}) Elapsed:{elapsed/60:.1f}m "
            f"ETA:{remaining/60:.1f}m{metrics_str}"
        )
        padding = " " * max(self.last_len - len(message), 0)
        sys.stdout.write(message + padding)
        sys.stdout.flush()
        self.last_len = len(message)
        if current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

