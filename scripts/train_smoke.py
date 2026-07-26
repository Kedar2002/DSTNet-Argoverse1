"""
scripts.train_smoke

CPU smoke-training script for DSTNet.

Purpose
-------
Verifies that the complete training pipeline is stable before
running long training sessions.

Pipeline
--------
CSV
    ↓
SceneParser
    ↓
ScenePreprocessor
    ↓
ArgoverseDataset
    ↓
Subset (64 scenes)
    ↓
DataLoader
    ↓
DSTNet
    ↓
TotalLoss
    ↓
Backward
    ↓
AdamW
    ↓
Checkpoint

Configuration
-------------
Device      : CPU
Scenes      : 64
Epochs      : 1
Batch Size  : 2
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
# Training
###############################################################################

from losses.total_loss import TotalLoss

from engine.optimizer import (
    build_optimizer,
)

from engine.scheduler import (
    build_scheduler,
)

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
    / "train_smoke"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "smoke"
)

DEVICE = torch.device("cpu")

###############################################################################
# Smoke Training Configuration
###############################################################################

NUM_SCENES = 64

BATCH_SIZE = 2

NUM_EPOCHS = 1

NUM_WORKERS = 0

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-2

GRADIENT_CLIP = 5.0

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

        list(
            range(
                NUM_SCENES,
            )
        ),

    )

    print(
        f"Original Dataset : {len(dataset)} scenes"
    )

    print(
        f"Smoke Dataset    : {len(subset)} scenes"
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

    print(
        f"Workers    : {NUM_WORKERS}"
    )

    return loader


###############################################################################
# Model Builder
###############################################################################

def build_model():

    print_section(
        "Building DSTNet"
    )

    model = DSTNet(

        observation_steps=20,

        prediction_steps=30,

        lane_points=20,

        hidden_dim=256,

        num_heads=8,

        num_encoder_layers=3,

        num_modes=6,

        refinement_iterations=2,

    )

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

def build_training(
        model,
        total_steps: int,
    ):

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

    print("Optimizer : AdamW")

    print("Scheduler : Cosine")

    print("Loss      : TotalLoss")

    return (

        optimizer,

        scheduler,

        criterion,

    )

###############################################################################
# Average Meter
###############################################################################

class AverageMeter:
    """
    Tracks running averages.
    """

    def __init__(self) -> None:

        self.reset()

    def reset(self) -> None:

        self.value = 0.0

        self.total = 0.0

        self.count = 0

        self.average = 0.0

    def update(
        self,
        value: float,
        n: int = 1,
    ) -> None:

        self.value = float(value)

        self.total += float(value) * n

        self.count += n

        self.average = self.total / self.count


###############################################################################
# Gradient Norm
###############################################################################

def compute_gradient_norm(
    model: torch.nn.Module,
) -> float:
    """
    Compute global L2 gradient norm.
    """

    total_norm_sq = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        grad_norm = parameter.grad.detach().norm(2)

        total_norm_sq += grad_norm.item() ** 2

    return total_norm_sq ** 0.5


###############################################################################
# Checkpoint
###############################################################################

def save_checkpoint(
    *,
    epoch: int,
    model: DSTNet,
    optimizer,
    scheduler,
    loss: float,
) -> Path:

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"smoke_epoch_{epoch}.pth"
    )

    torch.save(
        {

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "scheduler_state_dict": scheduler.state_dict(),

            "loss": loss,

        },
        checkpoint_path,
    )

    print(
        f"\nCheckpoint saved : {checkpoint_path}"
    )

    return checkpoint_path


###############################################################################
# Batch Logger
###############################################################################

def log_batch(
    *,
    epoch: int,
    batch_index: int,
    total_batches: int,
    elapsed: float,
    loss_dict: dict,
    grad_norm: float,
) -> None:

    print(

        f"[Epoch {epoch:02d}] "

        f"Batch "

        f"{batch_index:03d}/{total_batches:03d} "

        f"| "

        f"Loss "

        f"{loss_dict['loss'].item():8.4f} "

        f"| "

        f"Proposal "

        f"{loss_dict['proposal_loss'].item():7.4f} "

        f"| "

        f"Cls "

        f"{loss_dict['classification_loss'].item():7.4f} "

        f"| "

        f"Score "

        f"{loss_dict['score_loss'].item():7.4f} "

        f"| "

        f"Ref "

        f"{loss_dict['refinement_loss'].item():7.4f} "

        f"| "

        f"Grad "

        f"{grad_norm:8.4f} "

        f"| "

        f"{elapsed:6.2f} ms"

    )


###############################################################################
# Epoch Summary
###############################################################################

def print_epoch_summary(
    *,
    epoch: int,
    loss_meter: AverageMeter,
    epoch_time: float,
) -> None:

    print()

    print("=" * 80)

    print(f"Epoch {epoch} Summary")

    print("=" * 80)

    print(
        f"Average Loss : {loss_meter.average:.6f}"
    )

    print(
        f"Epoch Time   : {epoch_time:.2f} s"
    )


###############################################################################
# Interrupt Handler
###############################################################################

def training_interrupted() -> None:

    print()

    print("=" * 80)

    print("Training interrupted by user.")

    print("=" * 80)

###############################################################################
# Train One Epoch
###############################################################################

def train_one_epoch(
    *,
    epoch: int,
    model: DSTNet,
    loader: DataLoader,
    criterion: TotalLoss,
    optimizer,
    scheduler,
) -> float:

    print_header(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    model.train()

    loss_meter = AverageMeter()

    epoch_start = time.perf_counter()

    total_batches = len(loader)

    ###########################################################################
    # Training Loop
    ###########################################################################

    for batch_index, batch in enumerate(loader, start=1):

        iteration_start = time.perf_counter()

        #######################################################################
        # Move batch to device
        #######################################################################

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

        #######################################################################
        # Zero Gradients
        #######################################################################

        optimizer.zero_grad(
            set_to_none=True,
        )

        #######################################################################
        # Forward
        #######################################################################

        coarse_prediction, refined_prediction = model(

            agent_trajectories=agent_trajectories,

            lane_centerlines=lane_centerlines,

            positions=positions,

            headings=headings,

            graph=graph,

            agent_mask=agent_mask,

            lane_mask=lane_mask,

        )

        #######################################################################
        # Loss
        #######################################################################

        loss_dict = criterion(

            prediction=coarse_prediction,

            refined_prediction=refined_prediction,

            ground_truth=future_trajectories,

        )

        total_loss = loss_dict["loss"]

        #######################################################################
        # Backward
        #######################################################################

        total_loss.backward()

        #######################################################################
        # Gradient Clipping
        #######################################################################

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=GRADIENT_CLIP,

        )

        #######################################################################
        # Gradient Statistics
        #######################################################################

        grad_norm = compute_gradient_norm(
            model,
        )

        #######################################################################
        # Optimizer
        #######################################################################

        optimizer.step()

        scheduler.step()

        #######################################################################
        # Statistics
        #######################################################################

        loss_meter.update(

            total_loss.item(),

            n=agent_trajectories.shape[0],

        )

        elapsed = (

            time.perf_counter()

            - iteration_start

        ) * 1000.0

        #######################################################################
        # Logging
        #######################################################################

        log_batch(

            epoch=epoch,

            batch_index=batch_index,

            total_batches=total_batches,

            elapsed=elapsed,

            loss_dict=loss_dict,

            grad_norm=grad_norm,

        )

    ###########################################################################
    # Epoch Summary
    ###########################################################################

    epoch_time = (

        time.perf_counter()

        - epoch_start

    )

    print_epoch_summary(

        epoch=epoch,

        loss_meter=loss_meter,

        epoch_time=epoch_time,

    )

    ###########################################################################
    # Return average loss
    #######################################################################

    return loss_meter.average

###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet CPU Smoke Training"
    )

    ###########################################################################
    # Dataset
    ###########################################################################

    dataset = build_dataset()

    loader = build_dataloader(
        dataset,
    )

    ###########################################################################
    # Model
    ###########################################################################

    model = build_model()

    ###########################################################################
    # Training Components
    ###########################################################################

    total_steps = len(loader) * NUM_EPOCHS

    optimizer, scheduler, criterion = build_training(

        model,

        total_steps=total_steps,

    )

    ###########################################################################
    # Training
    ###########################################################################

    best_loss = float("inf")

    overall_start = time.perf_counter()

    try:

        for epoch in range(
            1,
            NUM_EPOCHS + 1,
        ):

            average_loss = train_one_epoch(

                epoch=epoch,

                model=model,

                loader=loader,

                criterion=criterion,

                optimizer=optimizer,

                scheduler=scheduler,

            )

            ###################################################################
            # Save Best Checkpoint
            ###################################################################

            if average_loss < best_loss:

                best_loss = average_loss

                save_checkpoint(

                    epoch=epoch,

                    model=model,

                    optimizer=optimizer,

                    scheduler=scheduler,

                    loss=average_loss,

                )

        #######################################################################
        # Final Summary
        #######################################################################

        total_time = (
            time.perf_counter()
            - overall_start
        )

        print()

        print("=" * 80)

        print("CPU Smoke Training Complete")

        print("=" * 80)

        print(
            f"Best Loss     : {best_loss:.6f}"
        )

        print(
            f"Total Time    : {total_time:.2f} s"
        )

        print(
            f"Checkpoint    : {CHECKPOINT_DIR}"
        )

        print()

        print("✓ Smoke training successful.")

    ###########################################################################
    # Graceful Exit
    ###########################################################################

    except KeyboardInterrupt:

        training_interrupted()


###############################################################################

if __name__ == "__main__":

    main()
