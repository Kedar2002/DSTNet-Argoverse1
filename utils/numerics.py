"""Runtime switches for expensive numerical sanity checks."""

from __future__ import annotations

import os


FINITE_CHECKS_ENABLED = os.environ.get(
    "DSTNET_VALIDATE_FINITE",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}


__all__ = ["FINITE_CHECKS_ENABLED"]
