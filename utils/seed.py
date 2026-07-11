"""
Utilities for deterministic execution and experiment reproducibility.

Every executable script should call `seed_everything()` before creating
datasets, models, or dataloaders.

Example
-------
from utils.seed import seed_everything

seed_everything(42)
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from utils.logger import get_logger

LOGGER = get_logger(__name__)


def seed_everything(
    seed: int = 42,
    deterministic: bool = True,
) -> None:
    """
    Seed all supported random number generators.

    Parameters
    ----------
    seed : int
        Global random seed.

    deterministic : bool
        Enable deterministic PyTorch execution.
        May reduce performance slightly.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            LOGGER.warning(
                "Deterministic algorithms are not fully supported "
                "for the current PyTorch installation."
            )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    LOGGER.info(
        "Global random seed set to %d (deterministic=%s)",
        seed,
        deterministic,
    )


def seed_worker(worker_id: int) -> None:
    """
    Initialize DataLoader worker seed.

    This function should be passed to the DataLoader as
    worker_init_fn.

    Parameters
    ----------
    worker_id : int
        Worker identifier assigned by PyTorch.
    """

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)

    random.seed(worker_seed)


def create_generator(seed: int = 42) -> torch.Generator:
    """
    Create a seeded PyTorch Generator.

    Returns
    -------
    torch.Generator
        Deterministically seeded generator.
    """

    generator = torch.Generator()

    generator.manual_seed(seed)

    return generator
