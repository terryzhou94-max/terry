"""Simple textual progress bar with elapsed time and ETA reporting."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgressState:
    total: int
    current: int = 0
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    bar_length: int = 40


class ProgressBar:
    """A minimal progress bar for console environments such as PyCharm."""

    def __init__(self, total: int, description: str = "", bar_length: int = 40) -> None:
        if total <= 0:
            raise ValueError("total must be positive")
        self.state = ProgressState(total=total, bar_length=bar_length)
        self.description = description

    def update(self, step: int = 1, extra_message: Optional[str] = None) -> None:
        state = self.state
        state.current += step
        now = time.time()
        state.last_update_time = now
        elapsed = now - state.start_time
        progress = min(state.current / state.total, 1.0)
        filled_length = int(state.bar_length * progress)
        bar = "#" * filled_length + "-" * (state.bar_length - filled_length)
        eta = (elapsed / progress - elapsed) if progress > 0 else float("nan")
        eta_str = _format_duration(eta) if progress > 0 else "NA"
        elapsed_str = _format_duration(elapsed)
        message = extra_message or ""
        output = (
            f"\r{self.description} |{bar}| "
            f"{progress * 100:6.2f}% Elapsed: {elapsed_str} ETA: {eta_str} {message}"
        )
        sys.stdout.write(output)
        sys.stdout.flush()
        if state.current >= state.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _format_duration(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "NA"
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:04.1f}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:04.1f}s"
    return f"{secs:0.2f}s"


__all__ = ["ProgressBar"]
