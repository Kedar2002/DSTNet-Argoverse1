"""
scripts.train_smoke

CPU smoke-training script for the current DSTNet implementation.

Purpose
-------
Runs a small real Argoverse-1 subset through the current dataset, model,
loss, optimizer, scheduler, backward pass, and checkpoint path.

This is a framework smoke test, not a performance experiment.

The script deliberately uses the current production builders from
``scripts.train`` where possible, so model/configuration changes do not
need to be duplicated here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


###############################################################################
# Current Production Builders
###############################################################################

from datasets.collate import collate_fn

from scripts.train import (
    build_criterion,
    build_dataset,
    build_dataset_roots,
    build_model,
    build_training_optimizer,
    build_training_scheduler,
)


###############################################################################
# Configuration
###############################################################################

DEVICE = torch.device("cpu")

NUM_SCENES = 64

BATCH_SIZE = 2

NUM_EPOCHS = 1

NUM_WORKERS = 0

GRADIENT_CLIP = 5.0

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "smoke"
)


###############################################################################
# Printing
###############################################################################


def print_header(
    title: str,
) -> None:

    print()

    print("=" * 80)

    print(title)

    print("=" * 80)


def print_section(
    title: str,
) -> None:

    print()

    print(title)

    print("-" * 80)


###############################################################################
# Parameter Counter
###############################################################################


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total,
        trainable,
    )


###############################################################################
# Dataset
###############################################################################


def build_smoke_dataset():
    """
    Build the real training dataset through the current production
    dataset builder and restrict it to NUM_SCENES.
    """

    print_section(
        "Building Dataset"
    )

    train_root, _ = build_dataset_roots()

    if not train_root.exists():

        raise FileNotFoundError(
            f"Training directory does not exist: "
            f"{train_root}"
        )

    dataset = build_dataset(
        train_root,
        train=True,
    )

    subset_size = min(
        NUM_SCENES,
        len(dataset),
    )

    if subset_size <= 0:

        raise RuntimeError(
            "Training dataset contains no scenes."
        )

    subset = Subset(
        dataset,
        range(subset_size),
    )

    print(
        f"Original Dataset : "
        f"{len(dataset):,} scenes"
    )

    print(
        f"Smoke Dataset    : "
        f"{len(subset):,} scenes"
    )

    return subset


###############################################################################
# DataLoader
###############################################################################


def build_smoke_dataloader(
    dataset,
) -> DataLoader:

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
        drop_last=False,
    )

    print(
        f"Batch Size : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Workers    : "
        f"{NUM_WORKERS}"
    )

    print(
        f"Batches    : "
        f"{len(loader):,}"
    )

    return loader


###############################################################################
# Gradient Norm
###############################################################################


def compute_gradient_norm(
    model: torch.nn.Module,
) -> float:

    total_norm_sq = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        gradient_norm = (
            parameter.grad.detach().norm(2)
        )

        total_norm_sq += (
            gradient_norm.item() ** 2
        )

    return total_norm_sq ** 0.5


###############################################################################
# Checkpoint
###############################################################################


def save_checkpoint(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    loss: float,
) -> Path:

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"smoke_epoch_{epoch}.pth"
    )

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

            "loss": loss,
        },
        checkpoint_path,
    )

    print(
        f"Checkpoint saved : "
        f"{checkpoint_path}"
    )

    return checkpoint_path


###############################################################################
# Batch Logging
###############################################################################


def log_batch(
    *,
    epoch: int,
    batch_index: int,
    total_batches: int,
    elapsed_ms: float,
    losses: dict[str, torch.Tensor],
    gradient_norm: float,
) -> None:

    print(
        f"[Epoch {epoch:02d}] "
        f"Batch "
        f"{batch_index:03d}/"
        f"{total_batches:03d} "
        f"| Loss "
        f"{losses['loss'].item():10.4f} "
        f"| Proposal "
        f"{losses['proposal_loss'].item():9.4f} "
        f"| Cls "
        f"{losses['classification_loss'].item():9.4f} "
        f"| Score "
        f"{losses['score_loss'].item():9.4f} "
        f"| Ref "
        f"{losses['refinement_loss'].item():9.4f} "
        f"| Grad "
        f"{gradient_norm:9.4f} "
        f"| {elapsed_ms:8.2f} ms"
    )


###############################################################################
# Main
###############################################################################


def main() -> None:

    print_header(
        "DSTNet CPU Smoke Training"
    )

    ###########################################################################
    # Configuration
    ###########################################################################

    print_section(
        "Smoke Configuration"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Scenes       : {NUM_SCENES}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Epochs       : {NUM_EPOCHS}"
    )

    print(
        f"Workers      : {NUM_WORKERS}"
    )

    print(
        f"Grad clip    : {GRADIENT_CLIP}"
    )

    print(
        f"Checkpoint   : {CHECKPOINT_DIR}"
    )

    ###########################################################################
    # Dataset / DataLoader
    ###########################################################################

    dataset = build_smoke_dataset()

    loader = build_smoke_dataloader(
        dataset,
    )

    ###########################################################################
    # Model
    ###########################################################################

    print_section(
        "Building DSTNet"
    )

    model = build_model()

    model.to(
        DEVICE
    )

    total_parameters, trainable_parameters = (
        count_parameters(
            model
        )
    )

    print(
        f"Device               : "
        f"{DEVICE}"
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
    # Training Components
    ###########################################################################

    print_section(
        "Building Training Components"
    )

    criterion = build_criterion()

    optimizer = build_training_optimizer(
        model,
    )

    scheduler = build_training_scheduler(
        optimizer,
        total_steps=(
            len(loader)
            * NUM_EPOCHS
        ),
    )

    print(
        f"Optimizer : "
        f"{optimizer.__class__.__name__}"
    )

    print(
        f"Scheduler : "
        f"{scheduler.__class__.__name__}"
    )

    print(
        f"Loss      : "
        f"{criterion}"
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

            print_header(
                f"Epoch {epoch}/{NUM_EPOCHS}"
            )

            model.train()

            epoch_start = (
                time.perf_counter()
            )

            running_loss = 0.0

            batch_count = 0

            ###################################################################
            # Batch Loop
            ###################################################################

            for batch_index, batch in enumerate(
                loader,
                start=1,
            ):

                iteration_start = (
                    time.perf_counter()
                )

                ################################################################
                # Current batch contract
                ################################################################

                agent_trajectories = (
                    batch[
                        "agent_trajectories"
                    ].to(DEVICE)
                )

                future_trajectories = (
                    batch[
                        "future_trajectories"
                    ].to(DEVICE)
                )

                map_centerlines = (
                    batch[
                        "map_centerlines"
                    ].to(DEVICE)
                )

                positions = (
                    batch[
                        "positions"
                    ].to(DEVICE)
                )

                agent_mask = batch.get(
                    "agent_mask"
                )

                if agent_mask is not None:

                    agent_mask = (
                        agent_mask.to(
                            DEVICE
                        )
                    )

                map_mask = batch.get(
                    "map_mask"
                )

                if map_mask is not None:

                    map_mask = (
                        map_mask.to(
                            DEVICE
                        )
                    )

                graph = batch[
                    "graph"
                ]

                ################################################################
                # Optimizer
                ################################################################

                optimizer.zero_grad(
                    set_to_none=True
                )

                ################################################################
                # Forward
                ################################################################

                coarse_prediction, refined_prediction = model(

                    agent_trajectories=(
                        agent_trajectories
                    ),

                    map_centerlines=(
                        map_centerlines
                    ),

                    positions=(
                        positions
                    ),

                    graph=graph,

                    agent_mask=(
                        agent_mask
                    ),

                    map_mask=(
                        map_mask
                    ),
                )

                ################################################################
                # Loss
                ################################################################

                losses = criterion(

                    prediction=(
                        coarse_prediction
                    ),

                    refined_prediction=(
                        refined_prediction
                    ),

                    ground_truth=(
                        future_trajectories
                    ),
                )

                loss = losses[
                    "loss"
                ]

                if not torch.isfinite(
                    loss
                ).all():

                    raise RuntimeError(
                        f"Non-finite loss at "
                        f"epoch {epoch}, "
                        f"batch {batch_index}."
                    )

                ################################################################
                # Backward
                ################################################################

                loss.backward()

                ################################################################
                # Gradient clipping
                ################################################################

                gradient_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=GRADIENT_CLIP,
                        error_if_nonfinite=True,
                    )
                )

                ################################################################
                # Optimizer / Scheduler
                ################################################################

                optimizer.step()

                if scheduler is not None:

                    scheduler.step()

                ################################################################
                # Statistics
                ################################################################

                running_loss += (
                    loss.item()
                )

                batch_count += 1

                elapsed_ms = (
                    time.perf_counter()
                    - iteration_start
                ) * 1000.0

                log_batch(

                    epoch=epoch,

                    batch_index=batch_index,

                    total_batches=len(
                        loader
                    ),

                    elapsed_ms=elapsed_ms,

                    losses=losses,

                    gradient_norm=float(
                        gradient_norm
                    ),
                )

            ###################################################################
            # Epoch Summary
            ###################################################################

            if batch_count == 0:

                raise RuntimeError(
                    "Smoke training produced "
                    "zero batches."
                )

            average_loss = (
                running_loss
                / batch_count
            )

            epoch_time = (
                time.perf_counter()
                - epoch_start
            )

            print()

            print(
                f"Epoch {epoch} Summary"
            )

            print(
                f"Average Loss : "
                f"{average_loss:.6f}"
            )

            print(
                f"Epoch Time   : "
                f"{epoch_time:.2f} s"
            )

            ###################################################################
            # Checkpoint
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

        print_header(
            "CPU Smoke Training Complete"
        )

        print(
            f"Best Loss  : "
            f"{best_loss:.6f}"
        )

        print(
            f"Total Time : "
            f"{total_time:.2f} s"
        )

        print(
            f"Checkpoint : "
            f"{CHECKPOINT_DIR}"
        )

        print()

        print(
            "✓ Smoke training successful."
        )

    except KeyboardInterrupt:

        print()

        print("=" * 80)

        print(
            "Training interrupted by user."
        )

        print("=" * 80)


###############################################################################
# Entry Point
###############################################################################


if __name__ == "__main__":

    main()
