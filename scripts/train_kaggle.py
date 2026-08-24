"""
scripts.train_kaggle

Kaggle training entry point for the current DSTNet implementation.

Purpose
-------
Runs production-style training on the Kaggle Argoverse-1 dataset while
keeping Kaggle-specific paths in this file.

Current model/data terminology
------------------------------
The training batch uses:

    agent_trajectories
    future_trajectories
    map_centerlines
    positions
    agent_mask
    map_mask
    graph

``headings`` remains available in the dataset/scene representation but is
not passed as a separate argument to the current DSTNet forward interface.

Checkpointing
-------------
A full local checkpoint is written every epoch.

An optional external backup path is provided below. Once that path is
changed to a persistent mounted location, the script copies a complete
checkpoint there every ``EXTERNAL_SAVE_EVERY`` epochs.

The external backup is intentionally disabled until the path is configured.

Cache
-----
The dataset cache is stored under ``CACHE_ROOT`` by default. The
``CacheManager`` stores each processed ``SceneData`` as a pickle file, so
the cache directory can also be pointed at a persistent mounted path or
copied/downloaded as an artifact after training.
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


###############################################################################
# Current Repository Imports
###############################################################################

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.cache_manager import CacheManager
from datasets.collate import collate_fn
from datasets.map_loader import MapLoader
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser
from datasets.transforms import (
    build_eval_transform,
    build_train_transform,
)

from engine.evaluator import Evaluator
from engine.optimizer import build_optimizer
from engine.scheduler import build_scheduler
from engine.utils import move_to_device

from losses.total_loss import TotalLoss

from models.dstnet import DSTNet


###############################################################################
# Kaggle Dataset Paths
###############################################################################

TRAIN_ROOT = Path(
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_train_v1.1/train/data"
)

VAL_ROOT = Path(
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_val_v1.1/val/data"
)

TEST_ROOT = Path(
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_test_v1.1/test_obs/data"
)

MAP_ROOT = Path(
    "/kaggle/input/datasets/kedaradhikari/"
    "argoverse1-hd-mapss/"
    "hd_maps/map_files"
)


###############################################################################
# Kaggle Working Directories
###############################################################################

CHECKPOINT_ROOT = (
    Path(
        "/kaggle/working/checkpoints"
    )
)

LOG_ROOT = (
    Path(
        "/kaggle/working/logs"
    )
)

RESULTS_ROOT = (
    Path(
        "/kaggle/working/results"
    )
)

CACHE_ROOT = (
    Path(
        "/kaggle/working/cache"
    )
)


###############################################################################
# External Checkpoint Backup
###############################################################################
#
# IMPORTANT:
#
# Replace this placeholder with a persistent mounted path later.
#
# For example, after you configure a persistent external mount:
#
#     EXTERNAL_CHECKPOINT_ROOT = Path(
#         "/path/to/mounted/storage/DSTNet/checkpoints"
#     )
#
# Keep EXTERNAL_CHECKPOINT_ENABLED=False until the path is valid.
#
###############################################################################

EXTERNAL_CHECKPOINT_ROOT = Path(
    "/YOUR/EXTERNAL/MOUNT/DSTNet/checkpoints"
)

EXTERNAL_CHECKPOINT_ENABLED = False


# The local checkpoint is written every epoch.
#
# An external copy is made every five completed epochs.
#
# Change this to 3 or 4 if desired.

EXTERNAL_SAVE_EVERY = 5


###############################################################################
# Directories
###############################################################################

CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


###############################################################################
# Training Configuration
###############################################################################

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if torch.cuda.is_available():

    torch.backends.cudnn.benchmark = True


BATCH_SIZE = 8

NUM_WORKERS = 2

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-2

SAVE_EVERY = 1

GRADIENT_CLIP = 2.0


###############################################################################
# Early Stopping
###############################################################################

EARLY_STOPPING = True

PATIENCE = 10


###############################################################################
# Checkpoint Paths
###############################################################################

LATEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "latest.pth"
)

BEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "best_model.pth"
)

CSV_LOG = (
    LOG_ROOT
    / "training_log.csv"
)


###############################################################################
# Printing Utilities
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
# Scene Preprocessor
###############################################################################


def build_preprocessor() -> ScenePreprocessor:
    """
    Build the current ScenePreprocessor API.

    Current parameters:

        observation_steps
        prediction_steps
        map_sample_points
        spatial_radius
        map_radius
    """

    return ScenePreprocessor(

        observation_steps=20,

        prediction_steps=30,

        map_sample_points=20,

        spatial_radius=30.0,

        map_radius=30.0,
    )


###############################################################################
# Dataset
###############################################################################


def build_dataset(
    root: Path,
    *,
    train: bool,
) -> ArgoverseDataset:
    """
    Build one Argoverse-1 split using the current dataset API.
    """

    if not root.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{root}"
        )

    if not MAP_ROOT.exists():

        raise FileNotFoundError(
            f"HD map directory does not exist: "
            f"{MAP_ROOT}"
        )

    map_loader = MapLoader(
        map_root=MAP_ROOT,
    )

    parser = SceneParser(
        map_loader,
    )

    preprocessor = build_preprocessor()

    cache = CacheManager(
        CACHE_ROOT,
    )

    transform = (

        build_train_transform()

        if train

        else build_eval_transform()
    )

    dataset = ArgoverseDataset(

        root=root,

        parser=parser,

        preprocessor=preprocessor,

        transform=transform,

        cache=cache,
    )

    return dataset


###############################################################################
# DataLoader
###############################################################################


def build_dataloader(
    dataset: ArgoverseDataset,
    *,
    train: bool,
) -> DataLoader:

    return DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=train,

        num_workers=NUM_WORKERS,

        collate_fn=collate_fn,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

        drop_last=False,
    )


###############################################################################
# Model
###############################################################################


def build_model() -> DSTNet:
    """
    Build the current DSTNet.

    The model constructor defaults are used intentionally here so this
    Kaggle-specific runner does not duplicate an older constructor
    signature.
    """

    model = DSTNet()

    model.to(
        DEVICE,
    )

    total, trainable = count_parameters(
        model,
    )

    print_section(
        "Model"
    )

    print(
        f"Device               : "
        f"{DEVICE}"
    )

    print(
        f"Total Parameters     : "
        f"{total:,}"
    )

    print(
        f"Trainable Parameters : "
        f"{trainable:,}"
    )

    return model


###############################################################################
# Training Components
###############################################################################


def build_training_components(
    model: DSTNet,
    total_steps: int,
) -> tuple:

    optimizer = build_optimizer(

        model=model,

        optimizer="adamw",

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(

        optimizer,

        scheduler="cosine",

        total_steps=(
            total_steps
            * EPOCHS
        ),
    )

    criterion = TotalLoss()

    return (
        optimizer,
        scheduler,
        criterion,
    )


###############################################################################
# Directory Creation
###############################################################################


def create_directories() -> None:

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


###############################################################################
# Checkpoint State
###############################################################################


def build_checkpoint_state(
    *,
    epoch: int,
    model: DSTNet,
    optimizer,
    scheduler,
    train_loss: float,
    val_metrics: dict[str, float],
) -> dict:

    return {

        "epoch": epoch,

        "train_loss": train_loss,

        "val_metrics": val_metrics,

        "best_metric": val_metrics.get(
            "minADE",
            float("inf"),
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (

            scheduler.state_dict()

            if scheduler is not None

            else None
        ),

        "torch_rng_state": (
            torch.get_rng_state()
        ),

        "cuda_rng_state_all": (

            torch.cuda.get_rng_state_all()

            if torch.cuda.is_available()

            else None
        ),
    }


###############################################################################
# Save Local Checkpoint
###############################################################################


def save_checkpoint(
    *,
    epoch: int,
    model: DSTNet,
    optimizer,
    scheduler,
    train_loss: float,
    val_metrics: dict[str, float],
    best: bool = False,
) -> Path:
    """
    Save the complete local checkpoint.

    The checkpoint contains enough training state for model/optimizer/
    scheduler restoration.
    """

    checkpoint = build_checkpoint_state(

        epoch=epoch,

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        train_loss=train_loss,

        val_metrics=val_metrics,
    )

    torch.save(

        checkpoint,

        LATEST_CHECKPOINT,
    )

    if best:

        torch.save(

            checkpoint,

            BEST_CHECKPOINT,
        )

    print()

    print(
        f"Checkpoint saved : "
        f"{LATEST_CHECKPOINT}"
    )

    if best:

        print(
            f"Best model saved : "
            f"{BEST_CHECKPOINT}"
        )

    return LATEST_CHECKPOINT


###############################################################################
# External Checkpoint Backup
###############################################################################


def backup_checkpoint_externally(
    *,
    epoch: int,
    best: bool,
) -> None:
    """
    Copy the latest complete checkpoint to persistent external storage.

    This is intentionally opt-in.

    When enabled, an epoch-specific checkpoint and a rolling
    ``latest_external.pth`` are written.
    """

    if not EXTERNAL_CHECKPOINT_ENABLED:

        return

    if (
        epoch
        % EXTERNAL_SAVE_EVERY
        != 0
    ):

        return

    if str(
        EXTERNAL_CHECKPOINT_ROOT
    ).startswith(
        "/YOUR/EXTERNAL/"
    ):

        raise RuntimeError(

            "External checkpointing is enabled, "
            "but EXTERNAL_CHECKPOINT_ROOT still "
            "contains the placeholder path."
        )

    EXTERNAL_CHECKPOINT_ROOT.mkdir(

        parents=True,

        exist_ok=True,
    )

    external_epoch_path = (

        EXTERNAL_CHECKPOINT_ROOT

        / f"epoch_{epoch:04d}.pth"
    )

    shutil.copy2(

        LATEST_CHECKPOINT,

        external_epoch_path,
    )

    external_latest_path = (

        EXTERNAL_CHECKPOINT_ROOT

        / "latest_external.pth"
    )

    shutil.copy2(

        LATEST_CHECKPOINT,

        external_latest_path,
    )

    if (
        best
        and
        BEST_CHECKPOINT.exists()
    ):

        external_best_path = (

            EXTERNAL_CHECKPOINT_ROOT

            / "best_model.pth"
        )

        shutil.copy2(

            BEST_CHECKPOINT,

            external_best_path,
        )

    print()

    print(
        f"External checkpoint backup : "
        f"{external_epoch_path}"
    )


###############################################################################
# Resume
###############################################################################


def load_checkpoint(
    model: DSTNet,
    optimizer,
    scheduler,
) -> tuple[int, float]:
    """
    Resume from the local latest checkpoint if one exists.
    """

    if not LATEST_CHECKPOINT.exists():

        print_section(
            "Checkpoint"
        )

        print(
            "No checkpoint found."
        )

        return (
            0,
            float("inf"),
        )

    print_section(
        "Resuming Training"
    )

    checkpoint = torch.load(

        LATEST_CHECKPOINT,

        map_location=DEVICE,
    )

    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(

        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler_state = checkpoint.get(

        "scheduler_state_dict",

        None,
    )

    if (

        scheduler is not None

        and scheduler_state is not None
    ):

        scheduler.load_state_dict(

            scheduler_state
        )

    if "torch_rng_state" in checkpoint:

        torch.set_rng_state(

            checkpoint[
                "torch_rng_state"
            ]
        )

    if (

        torch.cuda.is_available()

        and checkpoint.get(
            "cuda_rng_state_all"
        ) is not None
    ):

        torch.cuda.set_rng_state_all(

            checkpoint[
                "cuda_rng_state_all"
            ]
        )

    epoch = int(

        checkpoint.get(
            "epoch",
            0,
        )
    )

    val_metrics = checkpoint.get(

        "val_metrics",

        {},
    )

    best_metric = float(

        checkpoint.get(

            "best_metric",

            val_metrics.get(

                "minADE",

                float("inf"),
            ),
        )
    )

    print(
        f"Resumed from epoch "
        f"{epoch}"
    )

    print(
        f"Best minADE : "
        f"{best_metric:.6f}"
    )

    return (
        epoch,
        best_metric,
    )


###############################################################################
# CSV Logging
###############################################################################

CSV_HEADER = [

    "epoch",

    "train_loss",

    "minADE",

    "minFDE",

    "MissRate",

    "learning_rate",

    "epoch_time",
]


def initialize_csv() -> None:

    if CSV_LOG.exists():

        return

    with open(

        CSV_LOG,

        "w",

        newline="",

        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            CSV_HEADER
        )


def append_csv(
    *,
    epoch: int,
    train_loss: float,
    metrics: dict[str, float],
    learning_rate: float,
    epoch_time: float,
) -> None:

    with open(

        CSV_LOG,

        "a",

        newline="",

        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(

            [

                epoch,

                train_loss,

                metrics.get(
                    "minADE",
                    0.0,
                ),

                metrics.get(
                    "minFDE",
                    0.0,
                ),

                metrics.get(
                    "MissRate",
                    0.0,
                ),

                learning_rate,

                epoch_time,
            ]
        )


###############################################################################
# Training
###############################################################################


def train_one_epoch(
    *,
    epoch: int,
    model: DSTNet,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    criterion: TotalLoss,
) -> float:
    """
    Train DSTNet for one complete epoch.
    """

    model.train()

    running_loss = 0.0

    num_batches = len(
        dataloader
    )

    epoch_start = (
        time.perf_counter()
    )

    for batch_index, batch in enumerate(

        dataloader,

        start=1,
    ):

        batch_start = (
            time.perf_counter()
        )

        #######################################################################
        # Move complete batch to device
        #######################################################################

        batch = move_to_device(

            batch,

            DEVICE,
        )

        #######################################################################
        # Optimizer
        #######################################################################

        optimizer.zero_grad(

            set_to_none=True,
        )

        #######################################################################
        # Current DSTNet forward interface
        #######################################################################

        (
            coarse_prediction,
            refined_prediction,
        ) = model(

            agent_trajectories=(
                batch[
                    "agent_trajectories"
                ]
            ),

            map_centerlines=(
                batch[
                    "map_centerlines"
                ]
            ),

            positions=(
                batch[
                    "positions"
                ]
            ),

            graph=(
                batch[
                    "graph"
                ]
            ),

            agent_mask=batch.get(
                "agent_mask"
            ),

            map_mask=batch.get(
                "map_mask"
            ),
        )

        #######################################################################
        # Loss
        #######################################################################

        losses = criterion(

            prediction=(
                coarse_prediction
            ),

            refined_prediction=(
                refined_prediction
            ),

            ground_truth=(
                batch[
                    "future_trajectories"
                ]
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

        #######################################################################
        # Backward
        #######################################################################

        loss.backward()

        #######################################################################
        # Gradient clipping
        #######################################################################

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=GRADIENT_CLIP,

                error_if_nonfinite=True,
            )
        )

        #######################################################################
        # Optimizer / Scheduler
        #######################################################################

        optimizer.step()

        if scheduler is not None:

            scheduler.step()

        #######################################################################
        # Statistics
        #######################################################################

        running_loss += (
            loss.item()
        )

        batch_time = (

            time.perf_counter()

            - batch_start
        )

        if (

            batch_index == 1

            or batch_index % 10 == 0

            or batch_index == num_batches
        ):

            print(

                f"Epoch {epoch:03d} "

                f"[{batch_index:04d}/"
                f"{num_batches:04d}] "

                f"loss={loss.item():.6f} "

                f"grad={float(gradient_norm):.4f} "

                f"lr="
                f"{optimizer.param_groups[0]['lr']:.8e} "

                f"time={batch_time:.2f}s"
            )

    if num_batches == 0:

        raise RuntimeError(

            "Training DataLoader contains "
            "zero batches."
        )

    epoch_time = (

        time.perf_counter()

        - epoch_start
    )

    average_loss = (

        running_loss

        / num_batches
    )

    print()

    print(

        f"Training Epoch {epoch} "
        f"Loss : {average_loss:.6f}"
    )

    print(

        f"Training Epoch Time : "
        f"{epoch_time:.2f} s"
    )

    return average_loss


###############################################################################
# Validation
###############################################################################


def validate_one_epoch(
    *,
    model: DSTNet,
    dataloader: DataLoader,
) -> dict[str, float]:
    """
    Evaluate using the current Evaluator.
    """

    print()

    print("=" * 80)

    print(
        "Validation"
    )

    print("=" * 80)

    evaluator = Evaluator(

        model=model,

        dataloader=dataloader,

        device=DEVICE,
    )

    start_time = (
        time.perf_counter()
    )

    metrics = evaluator.evaluate()

    validation_time = (

        time.perf_counter()

        - start_time
    )

    print()

    print("-" * 80)

    print(
        "Validation Metrics"
    )

    print("-" * 80)

    for key in sorted(
        metrics.keys()
    ):

        print(

            f"{key:<15}"

            f"{metrics[key]:.6f}"
        )

    print()

    print(

        f"Validation Time : "
        f"{validation_time:.2f} s"
    )

    required_metrics = (

        "minADE",

        "minFDE",

        "MissRate",
    )

    for metric_name in (
        required_metrics
    ):

        if metric_name not in metrics:

            raise RuntimeError(

                f"Missing validation metric "
                f"'{metric_name}'."
            )

        value = metrics[
            metric_name
        ]

        if not torch.isfinite(

            torch.tensor(

                value,

                dtype=torch.float64,
            )
        ):

            raise RuntimeError(

                f"Metric '{metric_name}' "
                "is NaN or Inf."
            )

    return metrics


###############################################################################
# Epoch Summary
###############################################################################


def print_epoch_summary(
    *,
    epoch: int,
    train_loss: float,
    metrics: dict[str, float],
    learning_rate: float,
    epoch_time: float,
) -> None:

    print()

    print("=" * 80)

    print(
        f"Epoch {epoch} Summary"
    )

    print("=" * 80)

    print(

        f"Train Loss : "
        f"{train_loss:.6f}"
    )

    print(

        f"minADE     : "
        f"{metrics.get('minADE', float('nan')):.6f}"
    )

    print(

        f"minFDE     : "
        f"{metrics.get('minFDE', float('nan')):.6f}"
    )

    print(

        f"MissRate   : "
        f"{metrics.get('MissRate', float('nan')):.6f}"
    )

    print(

        f"Learning Rate : "
        f"{learning_rate:.8e}"
    )

    print(

        f"Epoch Time   : "
        f"{epoch_time:.2f} s"
    )


###############################################################################
# Main Training Pipeline
###############################################################################


def run_training() -> None:

    create_directories()

    initialize_csv()

    ###########################################################################
    # Datasets
    ###########################################################################

    print_section(
        "Building Datasets"
    )

    train_dataset = build_dataset(

        TRAIN_ROOT,

        train=True,
    )

    val_dataset = build_dataset(

        VAL_ROOT,

        train=False,
    )

    print(

        f"Training Scenes   : "
        f"{len(train_dataset):,}"
    )

    print(

        f"Validation Scenes : "
        f"{len(val_dataset):,}"
    )

    ###########################################################################
    # DataLoaders
    ###########################################################################

    print_section(
        "Building DataLoaders"
    )

    train_loader = build_dataloader(

        train_dataset,

        train=True,
    )

    val_loader = build_dataloader(

        val_dataset,

        train=False,
    )

    print(

        f"Training Batches   : "
        f"{len(train_loader):,}"
    )

    print(

        f"Validation Batches : "
        f"{len(val_loader):,}"
    )

    ###########################################################################
    # Model
    ###########################################################################

    model = build_model()

    ###########################################################################
    # Training Components
    ###########################################################################

    (
        optimizer,
        scheduler,
        criterion,
    ) = build_training_components(

        model,

        total_steps=len(
            train_loader
        ),
    )

    ###########################################################################
    # Resume
    ###########################################################################

    (
        start_epoch,
        best_metric,
    ) = load_checkpoint(

        model,

        optimizer,

        scheduler,
    )

    epochs_without_improvement = 0

    training_start = (
        time.perf_counter()
    )

    ###########################################################################
    # Epoch Loop
    ###########################################################################

    try:

        for epoch in range(

            start_epoch + 1,

            EPOCHS + 1,
        ):

            print_header(

                f"Epoch "
                f"{epoch}/{EPOCHS}"
            )

            epoch_start = (
                time.perf_counter()
            )

            train_loss = train_one_epoch(

                epoch=epoch,

                model=model,

                dataloader=train_loader,

                optimizer=optimizer,

                scheduler=scheduler,

                criterion=criterion,
            )

            val_metrics = (
                validate_one_epoch(

                    model=model,

                    dataloader=val_loader,
                )
            )

            epoch_time = (

                time.perf_counter()

                - epoch_start
            )

            learning_rate = (

                optimizer.param_groups[0][
                    "lr"
                ]
            )

            append_csv(

                epoch=epoch,

                train_loss=train_loss,

                metrics=val_metrics,

                learning_rate=learning_rate,

                epoch_time=epoch_time,
            )

            ###################################################################
            # Best Model
            ###################################################################

            current_metric = (
                val_metrics.get(

                    "minADE",

                    float("inf"),
                )
            )

            is_best = (

                current_metric

                < best_metric
            )

            if is_best:

                best_metric = (
                    current_metric
                )

                epochs_without_improvement = 0

                print()

                print(

                    f"✓ New Best Model "

                    f"(minADE="
                    f"{best_metric:.6f})"
                )

            else:

                epochs_without_improvement += 1

                print(

                    f"No improvement "

                    f"({epochs_without_improvement}/"
                    f"{PATIENCE})"
                )

            ###################################################################
            # Local Checkpoint
            ###################################################################

            if (

                epoch
                % SAVE_EVERY
                == 0
            ):

                save_checkpoint(

                    epoch=epoch,

                    model=model,

                    optimizer=optimizer,

                    scheduler=scheduler,

                    train_loss=train_loss,

                    val_metrics=val_metrics,

                    best=is_best,
                )

                ################################################################
                # Optional external backup
                ################################################################

                backup_checkpoint_externally(

                    epoch=epoch,

                    best=is_best,
                )

            ###################################################################
            # Epoch Summary
            ###################################################################

            print_epoch_summary(

                epoch=epoch,

                train_loss=train_loss,

                metrics=val_metrics,

                learning_rate=learning_rate,

                epoch_time=epoch_time,
            )

            ###################################################################
            # Early Stopping
            ###################################################################

            if (

                EARLY_STOPPING

                and

                epochs_without_improvement
                >= PATIENCE
            ):

                print()

                print("=" * 80)

                print(
                    "Early stopping triggered."
                )

                print(

                    f"No improvement for "
                    f"{PATIENCE} epochs."
                )

                print("=" * 80)

                break

    except KeyboardInterrupt:

        print()

        print("=" * 80)

        print(
            "Training Interrupted"
        )

        print("=" * 80)

        if "epoch" in locals():

            save_checkpoint(

                epoch=epoch,

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

                train_loss=(

                    train_loss

                    if "train_loss" in locals()

                    else float("inf")
                ),

                val_metrics=(

                    val_metrics

                    if "val_metrics" in locals()

                    else {}
                ),

                best=False,
            )

            backup_checkpoint_externally(

                epoch=epoch,

                best=False,
            )

        raise

    total_training_time = (

        time.perf_counter()

        - training_start
    )

    print_header(
        "Training Complete"
    )

    print(

        f"Best minADE : "
        f"{best_metric:.6f}"
    )

    print(

        f"Total Time  : "
        f"{total_training_time / 3600:.2f} hours"
    )

    print(

        f"Checkpoints : "
        f"{CHECKPOINT_ROOT}"
    )

    print(

        f"Logs        : "
        f"{CSV_LOG}"
    )

    print(

        f"Cache       : "
        f"{CACHE_ROOT}"
    )

    if EXTERNAL_CHECKPOINT_ENABLED:

        print(

            f"External backups : "
            f"{EXTERNAL_CHECKPOINT_ROOT}"
        )

    else:

        print(

            "External backups : DISABLED "
            "(configure "
            "EXTERNAL_CHECKPOINT_ROOT "
            "and enable "
            "EXTERNAL_CHECKPOINT_ENABLED)"
        )


###############################################################################
# Entry Point
###############################################################################


def main() -> None:

    print_header(
        "DSTNet Kaggle Training"
    )

    print_section(
        "Kaggle Configuration"
    )

    print(
        f"Train root  : "
        f"{TRAIN_ROOT}"
    )

    print(
        f"Val root    : "
        f"{VAL_ROOT}"
    )

    print(
        f"Map root    : "
        f"{MAP_ROOT}"
    )

    print(
        f"Cache root  : "
        f"{CACHE_ROOT}"
    )

    print(
        f"Device      : "
        f"{DEVICE}"
    )

    print(
        f"Batch size  : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs      : "
        f"{EPOCHS}"
    )

    print(

        f"External checkpointing : "
        f"{EXTERNAL_CHECKPOINT_ENABLED}"
    )

    run_training()


###############################################################################
# Entry Point
###############################################################################


if __name__ == "__main__":

    main()
