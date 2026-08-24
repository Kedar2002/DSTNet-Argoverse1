"""
scripts.train_mini

Mini end-to-end framework-validation run for DSTNet.

Purpose
-------
This script reuses the current production training components and runs
a deliberately small real-data experiment to verify:

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

This is NOT a performance experiment.

The production training entry point remains:

    scripts/train.py

The mini runner intentionally does not use an obsolete ``run_training()``
wrapper. It constructs the current Trainer interface directly from the
same production builders used by scripts.train.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
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
# Production Training Components
###############################################################################

from scripts.train import (
    CFG,
    _optional_attribute,
    build_criterion,
    build_dataloader,
    build_dataset,
    build_dataset_roots,
    build_model,
    build_training_optimizer,
    build_training_scheduler,
    count_parameters,
    resolve_device,
    set_random_seed,
)

from engine.trainer import Trainer


###############################################################################
# Mini-Training Configuration
###############################################################################

TRAIN_SCENES = 8
VAL_SCENES = 4
EPOCHS = 1


###############################################################################
# Mini Output Directory
###############################################################################

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "mini_test"
)


###############################################################################
# Runtime Helpers
###############################################################################


def build_runtime_settings() -> tuple[
    torch.device,
    int,
    bool,
    int,
    bool,
    float | None,
]:
    """
    Resolve the runtime/training settings required by the mini run.
    """

    requested_device = str(
        _optional_attribute(
            CFG,
            (
                "runtime.device",
            ),
            "auto",
        )
    )

    device = resolve_device(
        requested_device
    )

    num_workers = int(
        _optional_attribute(
            CFG,
            (
                "runtime.num_workers",
            ),
            0,
        )
    )

    pin_memory = bool(
        _optional_attribute(
            CFG,
            (
                "runtime.pin_memory",
            ),
            device.type == "cuda",
        )
    )

    seed = int(
        _optional_attribute(
            CFG,
            (
                "runtime.seed",
            ),
            42,
        )
    )

    deterministic = bool(
        _optional_attribute(
            CFG,
            (
                "runtime.deterministic",
            ),
            True,
        )
    )

    gradient_clip = _optional_attribute(
        CFG,
        (
            "training.gradient_clip",
            "training.gradient_clip_norm",
        ),
        1.0,
    )

    if gradient_clip is not None:
        gradient_clip = float(
            gradient_clip
        )

    return (
        device,
        num_workers,
        pin_memory,
        seed,
        deterministic,
        gradient_clip,
    )


###############################################################################
# Dataset Helpers
###############################################################################


def build_train_subset() -> Subset:
    """
    Build the first TRAIN_SCENES real training scenes.
    """

    train_root, _ = build_dataset_roots()

    if not train_root.exists():
        raise FileNotFoundError(
            "Training directory does not exist: "
            f"{train_root}"
        )

    print()
    print("=" * 80)
    print("BUILDING MINI TRAINING DATASET")
    print("=" * 80)

    dataset = build_dataset(
        train_root,
        train=True,
    )

    subset_size = min(
        TRAIN_SCENES,
        len(dataset),
    )

    if subset_size == 0:
        raise RuntimeError(
            "The training dataset is empty."
        )

    subset = Subset(
        dataset,
        range(subset_size),
    )

    print(
        f"Original Training Scenes : "
        f"{len(dataset):,}"
    )

    print(
        f"Mini Training Scenes     : "
        f"{subset_size:,}"
    )

    return subset


def build_validation_subset() -> Subset:
    """
    Build the first VAL_SCENES real validation scenes.
    """

    _, val_root = build_dataset_roots()

    if not val_root.exists():
        raise FileNotFoundError(
            "Validation directory does not exist: "
            f"{val_root}"
        )

    print()
    print("=" * 80)
    print("BUILDING MINI VALIDATION DATASET")
    print("=" * 80)

    dataset = build_dataset(
        val_root,
        train=False,
    )

    subset_size = min(
        VAL_SCENES,
        len(dataset),
    )

    if subset_size == 0:
        raise RuntimeError(
            "The validation dataset is empty."
        )

    subset = Subset(
        dataset,
        range(subset_size),
    )

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
    Execute one clean mini end-to-end training run.
    """

    print()
    print("=" * 80)
    print("DSTNet MINI TRAINING FRAMEWORK TEST")
    print("=" * 80)

    ###########################################################################
    # Runtime
    ###########################################################################

    (
        device,
        num_workers,
        pin_memory,
        seed,
        deterministic,
        gradient_clip,
    ) = build_runtime_settings()

    set_random_seed(
        seed=seed,
        deterministic=deterministic,
    )

    batch_size = int(
        _optional_attribute(
            CFG,
            (
                "training.batch_size",
            ),
            2,
        )
    )

    ###########################################################################
    # Configuration Summary
    ###########################################################################

    print()
    print("Mini Training Configuration")
    print("-" * 80)

    print(
        f"Training Scenes     : {TRAIN_SCENES}"
    )

    print(
        f"Validation Scenes   : {VAL_SCENES}"
    )

    print(
        f"Epochs              : {EPOCHS}"
    )

    print(
        f"Batch size          : {batch_size}"
    )

    print(
        f"Device              : {device}"
    )

    print(
        f"Num workers         : {num_workers}"
    )

    print(
        f"Pin memory          : {pin_memory}"
    )

    print(
        f"Seed                : {seed}"
    )

    print(
        f"Gradient clip       : {gradient_clip}"
    )

    print(
        f"Checkpoint root     : {CHECKPOINT_ROOT}"
    )

    ###########################################################################
    # Clean Mini Checkpoint Directory
    ###########################################################################

    if CHECKPOINT_ROOT.exists():
        shutil.rmtree(
            CHECKPOINT_ROOT
        )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Datasets
    ###########################################################################

    train_dataset = build_train_subset()

    val_dataset = build_validation_subset()

    ###########################################################################
    # DataLoaders
    ###########################################################################

    print()
    print("=" * 80)
    print("BUILDING MINI DATALOADERS")
    print("=" * 80)

    train_loader = build_dataloader(
        train_dataset,
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = build_dataloader(
        val_dataset,
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"Training batches   : {len(train_loader):,}"
    )

    print(
        f"Validation batches : {len(val_loader):,}"
    )

    ###########################################################################
    # Model
    ###########################################################################

    print()
    print("=" * 80)
    print("BUILDING MINI MODEL")
    print("=" * 80)

    model = build_model()

    model.to(
        device
    )

    total_parameters, trainable_parameters = (
        count_parameters(
            model
        )
    )

    print(
        f"Device               : {device}"
    )

    print(
        f"Total Parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable Parameters : "
        f"{trainable_parameters:,}"
    )

    ###########################################################################
    # Loss
    ###########################################################################

    criterion = build_criterion()

    print()
    print(
        f"Loss : {criterion}"
    )

    ###########################################################################
    # Optimizer
    ###########################################################################

    optimizer = build_training_optimizer(
        model
    )

    print()
    print(
        f"Optimizer : "
        f"{optimizer.__class__.__name__}"
    )

    print(
        f"Learning rate : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    print(
        f"Weight decay  : "
        f"{optimizer.param_groups[0]['weight_decay']:.8f}"
    )

    ###########################################################################
    # Scheduler
    ###########################################################################

    total_training_steps = (
        len(train_loader)
        * EPOCHS
    )

    scheduler = build_training_scheduler(
        optimizer,
        total_steps=total_training_steps,
    )

    print(
        f"Scheduler : "
        f"{scheduler.__class__.__name__}"
    )

    print(
        f"Total optimizer steps : "
        f"{total_training_steps:,}"
    )

    ###########################################################################
    # Trainer
    ###########################################################################

    print()
    print("=" * 80)
    print("BUILDING MINI TRAINER")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=str(
            CHECKPOINT_ROOT
        ),
        gradient_clip=gradient_clip,
    )

    print(
        "Trainer initialized successfully."
    )

    ###########################################################################
    # End-to-End Training
    ###########################################################################

    print()
    print("=" * 80)
    print("STARTING MINI END-TO-END TRAINING")
    print("=" * 80)

    trainer.fit(
        epochs=EPOCHS
    )

    ###########################################################################
    # Checkpoint Verification
    ###########################################################################

    latest_checkpoint = (
        CHECKPOINT_ROOT
        / "latest.pth"
    )

    best_checkpoint = (
        CHECKPOINT_ROOT
        / "best_model.pth"
    )

    if not latest_checkpoint.exists():
        raise RuntimeError(
            "Mini training completed but latest.pth "
            "was not created."
        )

    if not best_checkpoint.exists():
        raise RuntimeError(
            "Mini training completed but best_model.pth "
            "was not created."
        )

    print()
    print("=" * 80)
    print("DSTNet MINI TRAINING FRAMEWORK TEST COMPLETE")
    print("=" * 80)

    print(
        f"Final Trainer epoch : "
        f"{trainer.epoch}"
    )

    print(
        f"Global steps        : "
        f"{trainer.global_step}"
    )

    print(
        f"Latest checkpoint   : "
        f"{latest_checkpoint}"
    )

    print(
        f"Best checkpoint     : "
        f"{best_checkpoint}"
    )

    print()


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":
    main()
