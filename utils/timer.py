"""
Timing utilities for the DSTNet project.

Provides a lightweight timer that can be used either as a context manager
or with explicit start()/stop() calls.

Example
-------
from utils.timer import Timer

with Timer("Preprocessing"):
    preprocess_dataset()
"""

from __future__ import annotations

import time
from typing import Optional

from utils.logger import get_logger

LOGGER = get_logger(__name__)


class Timer:
    """
    High-resolution execution timer.

    Parameters
    ----------
    name : str
        Human-readable name for the timed operation.

    logger : optional
        Logger instance. If None, the default project logger is used.

    enabled : bool
        If False, timing is disabled.
    """

    def __init__(
        self,
        name: str = "Operation",
        *,
        logger=None,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.logger = logger or LOGGER
        self.enabled = enabled

        self._start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def start(self) -> None:
        """
        Start the timer.
        """
        if not self.enabled:
            return

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Stop the timer.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        if not self.enabled:
            return 0.0

        if self._start_time is None:
            raise RuntimeError(
                "Timer.stop() called before Timer.start()."
            )

        self.elapsed = time.perf_counter() - self._start_time

        self.logger.info(
            "%s completed in %.3f seconds.",
            self.name,
            self.elapsed,
        )

        return self.elapsed

    def reset(self) -> None:
        """
        Reset timer state.
        """
        self._start_time = None
        self.elapsed = None

    @property
    def running(self) -> bool:
        """
        Whether the timer is currently running.
        """
        return self._start_time is not None

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.stop()
