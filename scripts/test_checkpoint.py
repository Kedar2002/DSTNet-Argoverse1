"""
scripts.test_checkpoint

Checkpoint verification for DSTNet.

Verifies

    ✓ Model checkpoint saving
    ✓ Model checkpoint loading
    ✓ Optimizer state restoration
    ✓ Scheduler state restoration
    ✓ Parameter equality
    ✓ Resume metadata

Pipeline
--------
Dataset
    ↓
DSTNet
    ↓
Forward
    ↓
Loss
    ↓
Backward
    ↓
Optimizer
    ↓
Save Checkpoint
    ↓
Create New Model
    ↓
Load Checkpoint
    ↓
Compare Everything
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import (
    DataLoader,
    Subset,
)

###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################
# Dataset
###############################################################################

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.cache_manager import CacheManager
from datasets.collate import collate_fn
from datasets.map_loader import MapLoader
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser
from datasets.transforms import Identity

###############################################################################
# Model
###############################################################################

from models.dstnet import DSTNet

###############################################################################
# Training Components
###############################################################################

from losses.total_loss import TotalLoss

from engine.optimizer import build_optimizer
from engine.scheduler import build_scheduler

###############################################################################
# Configuration
###############################################################################

MAP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
    / "hd_maps"
    / "map_files"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
    / "train"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "cache"
    / "test_checkpoint"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "checkpoint_test"
)

CHECKPOINT_FILE = (
    CHECKPOINT_DIR
    / "checkpoint_test.pth"
)

DEVICE = torch.device("cpu")

###############################################################################
# Test Configuration
###############################################################################

NUM_SCENES = 8

BATCH_SIZE = 2

NUM_WORKERS = 0

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-2

###############################################################################
# Printing Utilities
###############################################################################

def print_header(title: str) -> None:

    print()

    print("=" * 80)

    print(title)

    print("=" * 80)


def print_section(title: str) -> None:

    print()

    print(title)

    print("-" * 80)


###############################################################################
# Parameter Counter
###############################################################################

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


###############################################################################
# Dataset Builder
###############################################################################

def build_dataset():

    print_section(
        "Building Dataset"
    )

    map_loader = MapLoader(
        map_root=MAP_ROOT,
    )

    parser = SceneParser(
        map_loader,
    )

    preprocessor = ScenePreprocessor(

        observation_steps=20,

        prediction_steps=30,

        lane_sample_points=20,

        agent_radius=30.0,

        lane_radius=30.0,

    )

    cache = CacheManager(
        CACHE_ROOT,
    )

    dataset = ArgoverseDataset(

        root=DATASET_ROOT,

        parser=parser,

        preprocessor=preprocessor,

        transform=Identity(),

        cache=cache,

    )

    subset = Subset(
        dataset,
        list(range(NUM_SCENES)),
    )

    print(
        f"Dataset Size : {len(subset)}"
    )

    return subset


###############################################################################
# DataLoader Builder
###############################################################################

def build_dataloader(dataset):

    print_section(
        "Building DataLoader"
    )

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        collate_fn=collate_fn,

        pin_memory=False,

    )

    print(
        f"Batch Size : {BATCH_SIZE}"
    )

    return loader


###############################################################################
# Model Builder
###############################################################################

def build_model():

    print_section(
        "Building DSTNet"
    )

    model = DSTNet()

    model.to(
        DEVICE,
    )

    total, trainable = count_parameters(
        model,
    )

    print(
        f"Device               : {DEVICE}"
    )

    print(
        f"Total Parameters     : {total:,}"
    )

    print(
        f"Trainable Parameters : {trainable:,}"
    )

    return model


###############################################################################
# Training Components
###############################################################################

def build_training(model, total_steps: int):

    print_section(
        "Building Training Components"
    )

    optimizer = build_optimizer(

        model=model,

        optimizer="adamw",

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )

    scheduler = build_scheduler(

        optimizer,

        scheduler="cosine",

        total_steps=total_steps,

    )

    criterion = TotalLoss()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        optimizer,
        scheduler,
        criterion,
    )

###############################################################################
# Checkpoint Utilities
###############################################################################

def save_checkpoint(
    *,
    epoch: int,
    model: DSTNet,
    optimizer,
    scheduler,
    loss: float,
) -> None:
    """
    Save a training checkpoint.
    """

    print_section(
        "Saving Checkpoint"
    )

    checkpoint = {

        "epoch": epoch,

        "loss": loss,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

    }

    torch.save(
        checkpoint,
        CHECKPOINT_FILE,
    )

    print(
        f"Checkpoint : {CHECKPOINT_FILE}"
    )

    print("✓ Saved successfully")


###############################################################################
# Load Checkpoint
###############################################################################

def load_checkpoint(
    *,
    model: DSTNet,
    optimizer,
    scheduler,
):

    print_section(
        "Loading Checkpoint"
    )

    checkpoint = torch.load(

        CHECKPOINT_FILE,

        map_location=DEVICE,

    )

    ###############################################################
    # Restore Model
    ###############################################################

    model.load_state_dict(
        checkpoint["model_state_dict"],
    )

    ###############################################################
    # Restore Optimizer
    ###############################################################

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"],
    )

    ###############################################################
    # Restore Scheduler
    ###############################################################

    if (

        scheduler is not None

        and

        checkpoint["scheduler_state_dict"] is not None

    ):

        scheduler.load_state_dict(

            checkpoint["scheduler_state_dict"]

        )

    print("✓ Loaded successfully")

    return checkpoint


###############################################################################
# Build Fresh Training Objects
###############################################################################

def build_fresh_training_objects(
    total_steps: int,
):
    """
    Build a completely new model, optimizer,
    scheduler and criterion.

    This simulates resuming training in a
    brand-new Python process.
    """

    print_section(
        "Building Fresh Objects"
    )

    model = build_model()

    optimizer, scheduler, criterion = build_training(

        model,

        total_steps=total_steps,

    )

    return (

        model,

        optimizer,

        scheduler,

        criterion,

    )


###############################################################################
# Remove Old Checkpoint
###############################################################################

def remove_old_checkpoint():
    """
    Delete any previous checkpoint so that
    this test always starts clean.
    """

    if CHECKPOINT_FILE.exists():

        CHECKPOINT_FILE.unlink()

        print_section(
            "Removing Old Checkpoint"
        )

        print(
            f"Deleted : {CHECKPOINT_FILE}"
        )

###############################################################################
# Model Verification
###############################################################################

def verify_model_parameters(
    original_model: DSTNet,
    restored_model: DSTNet,
) -> bool:
    """
    Verify that every parameter tensor matches exactly.
    """

    print_section(
        "Model Parameter Verification"
    )

    original_state = original_model.state_dict()

    restored_state = restored_model.state_dict()

    all_match = True

    for name in original_state:

        if not torch.equal(

            original_state[name],

            restored_state[name],

        ):

            print(f"Mismatch : {name}")

            all_match = False

            break

    if all_match:

        print("✓ All model parameters match")

    else:

        print("✗ Model parameters do not match")

    return all_match


###############################################################################
# Optimizer Verification
###############################################################################

def verify_optimizer_state(
    original_optimizer,
    restored_optimizer,
) -> bool:
    """
    Verify optimizer state.

    Compares parameter groups and every tensor in the optimizer
    state dictionary individually.
    """

    print_section(
        "Optimizer Verification"
    )

    original = original_optimizer.state_dict()
    restored = restored_optimizer.state_dict()

    #######################################################################
    # Parameter groups
    #######################################################################

    if original["param_groups"] != restored["param_groups"]:

        print("✗ Optimizer parameter groups differ")

        return False

    #######################################################################
    # State keys
    #######################################################################

    if original["state"].keys() != restored["state"].keys():

        print("✗ Optimizer state keys differ")

        return False

    #######################################################################
    # Compare every optimizer tensor
    #######################################################################

    for parameter_id in original["state"]:

        original_state = original["state"][parameter_id]

        restored_state = restored["state"][parameter_id]

        if original_state.keys() != restored_state.keys():

            print(
                f"✗ State mismatch for parameter {parameter_id}"
            )

            return False

        for key in original_state:

            original_value = original_state[key]

            restored_value = restored_state[key]

            if isinstance(original_value, torch.Tensor):

                if not torch.equal(
                    original_value,
                    restored_value,
                ):

                    print(
                        f"✗ Tensor mismatch ({key}) "
                        f"for parameter {parameter_id}"
                    )

                    return False

            else:

                if original_value != restored_value:

                    print(
                        f"✗ Value mismatch ({key}) "
                        f"for parameter {parameter_id}"
                    )

                    return False

    print("✓ Optimizer state matches")

    return True


###############################################################################
# Scheduler Verification
###############################################################################

def verify_scheduler_state(
    original_scheduler,
    restored_scheduler,
) -> bool:
    """
    Verify scheduler state.
    """

    print_section(
        "Scheduler Verification"
    )

    if (

        original_scheduler is None

        and

        restored_scheduler is None

    ):

        print("✓ No scheduler")

        return True

    if (

        original_scheduler is None

        or

        restored_scheduler is None

    ):

        print("✗ Scheduler mismatch")

        return False

    original_state = original_scheduler.state_dict()

    restored_state = restored_scheduler.state_dict()

    if original_state.keys() != restored_state.keys():
        print("✗ Scheduler keys differ")
        return False

    for key in original_state:

        original_value = original_state[key]
        restored_value = restored_state[key]

        if isinstance(original_value, torch.Tensor):

            if not torch.equal(original_value, restored_value):

                print(f"✗ Scheduler mismatch ({key})")
                return False

        else:

            if original_value != restored_value:

                print(f"✗ Scheduler mismatch ({key})")
                return False

    print("✓ Scheduler state matches")

    return True


###############################################################################
# Metadata Verification
###############################################################################

def verify_metadata(
    checkpoint: dict,
    expected_epoch: int,
    expected_loss: float,
) -> bool:
    """
    Verify checkpoint metadata.
    """

    print_section(
        "Checkpoint Metadata"
    )

    epoch_ok = (

        checkpoint["epoch"]

        ==

        expected_epoch

    )

    loss_ok = abs(

        checkpoint["loss"]

        -

        expected_loss

    ) < 1e-8

    print(
        f"Epoch : {checkpoint['epoch']}"
    )

    print(
        f"Loss  : {checkpoint['loss']:.6f}"
    )

    if epoch_ok:

        print("✓ Epoch correct")

    else:

        print("✗ Epoch incorrect")

    if loss_ok:

        print("✓ Loss correct")

    else:

        print("✗ Loss incorrect")

    return epoch_ok and loss_ok


###############################################################################
# Complete Verification
###############################################################################

def verify_checkpoint(
    *,
    original_model,
    restored_model,
    original_optimizer,
    restored_optimizer,
    original_scheduler,
    restored_scheduler,
    checkpoint,
    epoch,
    loss,
) -> bool:
    """
    Execute every checkpoint verification.
    """

    model_ok = verify_model_parameters(

        original_model,

        restored_model,

    )

    optimizer_ok = verify_optimizer_state(

        original_optimizer,

        restored_optimizer,

    )

    scheduler_ok = verify_scheduler_state(

        original_scheduler,

        restored_scheduler,

    )

    metadata_ok = verify_metadata(

        checkpoint,

        expected_epoch=epoch,

        expected_loss=loss,

    )

    return (

        model_ok

        and

        optimizer_ok

        and

        scheduler_ok

        and

        metadata_ok

    )

###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet Checkpoint Verification"
    )

    ###########################################################################
    # Remove old checkpoint
    ###########################################################################

    remove_old_checkpoint()

    ###########################################################################
    # Dataset
    ###########################################################################

    dataset = build_dataset()

    loader = build_dataloader(
        dataset,
    )

    total_steps = len(loader)

    ###########################################################################
    # Original training objects
    ###########################################################################

    model = build_model()

    optimizer, scheduler, criterion = build_training(

        model,

        total_steps=total_steps,

    )

    ###########################################################################
    # Load one batch
    ###########################################################################

    print_section(
        "Loading Batch"
    )

    batch = next(iter(loader))

    agent_trajectories = batch[
        "agent_trajectories"
    ].to(DEVICE)

    future_trajectories = batch[
        "future_trajectories"
    ].to(DEVICE)

    lane_centerlines = batch[
        "lane_centerlines"
    ].to(DEVICE)

    positions = batch[
        "positions"
    ].to(DEVICE)

    headings = batch[
        "headings"
    ].to(DEVICE)

    agent_mask = batch[
        "agent_mask"
    ].to(DEVICE)

    lane_mask = batch[
        "lane_mask"
    ].to(DEVICE)

    graph = batch[
        "graph"
    ]

    ###########################################################################
    # One training step
    ###########################################################################

    print_section(
        "Running One Training Step"
    )

    model.train()

    optimizer.zero_grad(
        set_to_none=True,
    )

    coarse_prediction, refined_prediction = model(

        agent_trajectories=agent_trajectories,

        lane_centerlines=lane_centerlines,

        positions=positions,

        headings=headings,

        graph=graph,

        agent_mask=agent_mask,

        lane_mask=lane_mask,

    )

    loss_dict = criterion(

        prediction=coarse_prediction,

        refined_prediction=refined_prediction,

        ground_truth=future_trajectories,

    )

    loss = loss_dict["loss"]

    loss.backward()

    optimizer.step()

    scheduler.step()

    print(
        f"Loss : {loss.item():.6f}"
    )

    ###########################################################################
    # Save checkpoint
    ###########################################################################

    save_checkpoint(

        epoch=1,

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        loss=loss.item(),

    )

    ###########################################################################
    # Build fresh objects
    ###########################################################################

    restored_model, restored_optimizer, restored_scheduler, _ = (

        build_fresh_training_objects(

            total_steps=total_steps,

        )

    )

    ###########################################################################
    # Load checkpoint
    ###########################################################################

    checkpoint = load_checkpoint(

        model=restored_model,

        optimizer=restored_optimizer,

        scheduler=restored_scheduler,

    )

    ###########################################################################
    # Verify
    ###########################################################################

    passed = verify_checkpoint(

        original_model=model,

        restored_model=restored_model,

        original_optimizer=optimizer,

        restored_optimizer=restored_optimizer,

        original_scheduler=scheduler,

        restored_scheduler=restored_scheduler,

        checkpoint=checkpoint,

        epoch=1,

        loss=loss.item(),

    )

    ###########################################################################
    # Final
    ###########################################################################

    print()

    print("=" * 80)

    if passed:

        print("✓ CHECKPOINT TEST PASSED")

    else:

        print("✗ CHECKPOINT TEST FAILED")

    print("=" * 80)


###############################################################################

if __name__ == "__main__":

    main()

