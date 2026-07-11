"""
Centralized logging utilities for the DSTNet project.

All production modules should obtain loggers through this module instead of
creating their own logging configuration.

Features
--------
- Console logging
- File logging
- Automatic log directory creation
- Rich console support (optional)
- Duplicate handler protection
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "dstnet.log"


def _create_formatter() -> logging.Formatter:
    """
    Returns the standard formatter used throughout the project.
    """
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _create_console_handler() -> logging.Handler:
    """
    Create console handler.

    Uses RichHandler when available, otherwise falls back to StreamHandler.
    """
    if _RICH_AVAILABLE:
        handler = RichHandler(
            rich_tracebacks=True,
            markup=False,
            show_time=False,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(_create_formatter())

    return handler


def _create_file_handler(log_file: Path) -> logging.FileHandler:
    """
    Create file handler.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(
        filename=log_file,
        mode="a",
        encoding="utf-8",
    )

    handler.setFormatter(_create_formatter())

    return handler


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name:
        Logger name.

    level:
        Logging level.

    log_file:
        Optional custom log file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    console_handler = _create_console_handler()
    logger.addHandler(console_handler)

    file_handler = _create_file_handler(
        log_file or DEFAULT_LOG_FILE
    )
    logger.addHandler(file_handler)

    return logger
