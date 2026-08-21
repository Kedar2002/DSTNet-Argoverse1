"""
scripts.train_mini

Mini training script for DSTNet.

This script reuses the current production training pipeline while
training on a small subset of the Argoverse-1 dataset.

Purpose
-------
This is a framework-validation run, NOT a final training run.

It verifies that:

    ArgoverseDataset
        ->
    DataLoader
        ->
    DSTNet
        ->
    TrainStep
        ->
    TotalLoss
        ->
    Optimizer
        ->
    Scheduler
        ->
    Validation
        ->
    Checkpointing

works correctly over a complete epoch-level training cycle.

The production training implementation remains in:

    scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import Subset


###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


###############################################################################
# Production Training Pipeline
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
# Mini-Training Configuration
###############################################################################

# Number of scenes used for this framework test.
#
# Keep this deliberately small. The purpose is to verify the complete
# training pipeline rather than obtain meaningful model performance.
TRAIN_SCENES = 8

VAL_SCENES = 4


# One complete epoch is sufficient to verify:
#
#   forward
#   loss
#   backward
#   optimizer
#   scheduler
#   validation
#   checkpoint
#
EPOCHS = 1


###############################################################################
# Mini Output Directories
###############################################################################

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "mini_test"
)

LOG_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "mini_test"
)


###############################################################################
# Dataset Helpers
###############################################################################

def build_train_subset():
    """
    Build a small training subset using the production dataset builder.
    """

    print_section(
        "Building Mini Training Dataset"
    )

    ###########################################################################
    # Build the real production dataset.
    ###########################################################################

    dataset = build_dataset(
        TRAIN_ROOT,
        train=True,
    )

    ###########################################################################
    # Restrict it to the requested number of scenes.
    ###########################################################################

    subset_size = min(
        TRAIN_SCENES,
        len(dataset),
    )

    subset = Subset(
        dataset,
        range(subset_size),
    )

    ###########################################################################
    # Information
    ###########################################################################

    print(
        f"Original Training Scenes : "
        f"{len(dataset):,}"
    )

    print(
        f"Mini Training Scenes     : "
        f"{subset_size:,}"
    )

    return subset


def build_validation_subset():
    """
    Build a small validation subset using the production dataset builder.
    """

    print_section(
        "Building Mini Validation Dataset"
    )

    ###########################################################################
    # Build the real production validation dataset.
    ###########################################################################

    dataset = build_dataset(
        VAL_ROOT,
        train=False,
    )

    ###########################################################################
    # Restrict it to the requested number of scenes.
    ###########################################################################

    subset_size = min(
        VAL_SCENES,
        len(dataset),
    )

    subset = Subset(
        dataset,
        range(subset_size),
    )

    ###########################################################################
    # Information
    ###########################################################################

    print(
        f"Original Validation Scenes : "
        f"{len(dataset):,}"
    )

    print(
        f"Mini Validation Scenes     : "
        f"{subset_size:,}"
    )

    return subset


###############################################################################
# Main
###############################################################################

def main() -> None:
    """
    Execute one mini end-to-end training run.
    """

    print_header(
        "DSTNet Mini Training Framework Test"
    )

    ###########################################################################
    # Configuration Summary
    ###########################################################################

    print_section(
        "Mini Training Configuration"
    )

    print(
        f"Training Scenes     : "
        f"{TRAIN_SCENES}"
    )

    print(
        f"Validation Scenes   : "
        f"{VAL_SCENES}"
    )

    print(
        f"Epochs              : "
        f"{EPOCHS}"
    )

    print(
        f"Checkpoint Root     : "
        f"{CHECKPOINT_ROOT}"
    )

    print(
        f"Log Root            : "
        f"{LOG_ROOT}"
    )

    ###########################################################################
    # Build Datasets
    ###########################################################################

    train_dataset = build_train_subset()

    val_dataset = build_validation_subset()

    ###########################################################################
    # Run Production Training Pipeline
    ###########################################################################

    print_section(
        "Running Production Training Pipeline"
    )

    run_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        checkpoint_root=CHECKPOINT_ROOT,
        log_root=LOG_ROOT,
    )

    ###########################################################################
    # Completion
    ###########################################################################

    print()
    print("=" * 80)
    print("DSTNet Mini Training Framework Test Complete")
    print("=" * 80)

    print()
    print(
        f"Checkpoints : "
        f"{CHECKPOINT_ROOT}"
    )

    print(
        f"Logs        : "
        f"{LOG_ROOT}"
    )

    print()


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":
    main()
