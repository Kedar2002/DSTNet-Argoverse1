"""
scripts.train_mini

Mini training script for DSTNet.

This script reuses the production training pipeline while training on
a small subset of the Argoverse 1 dataset for rapid verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import Subset

###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################
# Import Production Training Pipeline
###############################################################################

from scripts.train import (
    TRAIN_ROOT,
    VAL_ROOT,
    build_dataset,
    print_header,
    print_section,
    run_training,
)

###############################################################################
# Mini Configuration
###############################################################################

TRAIN_SCENES = 2048

VAL_SCENES = 512

EPOCHS = 10

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "mini"
)

LOG_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "mini"
)

###############################################################################
# Dataset Builders
###############################################################################

def build_train_subset():

    print_section(
        "Building Mini Training Dataset"
    )

    dataset = build_dataset(
        TRAIN_ROOT,
        train=True,
    )

    subset_size = min(
        TRAIN_SCENES,
        len(dataset),
    )

    subset = Subset(
        dataset,
        range(subset_size),
    )

    print(
        f"Original Training Scenes : {len(dataset):,}"
    )

    print(
        f"Mini Training Scenes     : {subset_size}"
    )

    return subset


def build_validation_subset():

    print_section(
        "Building Mini Validation Dataset"
    )

    dataset = build_dataset(
        VAL_ROOT,
        train=False,
    )

    subset_size = min(
        VAL_SCENES,
        len(dataset),
    )

    subset = Subset(
        dataset,
        range(subset_size),
    )

    print(
        f"Original Validation Scenes : {len(dataset):,}"
    )

    print(
        f"Mini Validation Scenes     : {subset_size}"
    )

    return subset

###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet Mini Training"
    )

    ###########################################################################
    # Build Mini Datasets
    ###########################################################################

    train_dataset = build_train_subset()

    val_dataset = build_validation_subset()

    ###########################################################################
    # Run Production Training Pipeline
    ###########################################################################

    run_training(

        train_dataset=train_dataset,

        val_dataset=val_dataset,

        epochs=EPOCHS,

        checkpoint_root=CHECKPOINT_ROOT,

        log_root=LOG_ROOT,

    )


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()
