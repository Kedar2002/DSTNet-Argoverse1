"""
scripts.train

Production training script for DSTNet.

Responsibilities
----------------
- Training
- Validation
- Checkpointing
- Resume Training
- CSV Logging
- Learning Rate Scheduling
- Best Model Saving

Pipeline

Training Dataset
        │
        ▼
Validation Dataset
        │
        ▼
DataLoader
        │
        ▼
DSTNet
        │
        ▼
Forward
        │
        ▼
Loss
        │
        ▼
Backward
        │
        ▼
Optimizer
        │
        ▼
Scheduler
        │
        ▼
Validation
        │
        ▼
Checkpoint
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
from datasets.transforms import (
    build_train_transform,
    build_eval_transform,
)

###############################################################################
# Model
###############################################################################

from models.dstnet import DSTNet

###############################################################################
# Training
###############################################################################

from losses.total_loss import TotalLoss

from engine.optimizer import build_optimizer
from engine.scheduler import build_scheduler
from engine.evaluator import Evaluator
from engine.utils import move_to_device

###############################################################################
# Configuration
###############################################################################

TRAIN_ROOT = (
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_train_v1.1/train/data"
)

VAL_ROOT = (
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_val_v1.1/val/data"
)

TEST_ROOT = (
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_test_v1.1/test_obs/data"
)

MAP_ROOT = (
    "/kaggle/input/datasets/kedaradhikari/argoverse1-hdmaps/"
    "hd_maps/map_files"
)

CHECKPOINT_ROOT = Path("/kaggle/working/checkpoints")

LOG_ROOT = Path("/kaggle/working/logs")

RESULTS_ROOT = Path("/kaggle/working/results")

CACHE_ROOT = Path("/kaggle/working/cache")

CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

###############################################################################
# Hyperparameters
###############################################################################

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

BATCH_SIZE = 8

NUM_WORKERS = 2

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-2

SAVE_EVERY = 1

GRADIENT_CLIP = 5.0

###############################################################################
# Early Stopping
###############################################################################

EARLY_STOPPING = True

PATIENCE = 10

###############################################################################
# Utilities
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
    model,
):

    total = sum(

        parameter.numel()

        for parameter in model.parameters()

    )

    trainable = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )

    return total, trainable


###############################################################################
# Scene Preprocessor Builder
###############################################################################

def build_preprocessor():

    return ScenePreprocessor(

        observation_steps=20,

        prediction_steps=30,

        lane_sample_points=20,

        agent_radius=30.0,

        lane_radius=30.0,

    )


###############################################################################
# Dataset Builder
###############################################################################

def build_dataset(
    root: Path,
    train: bool,
):

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
# DataLoader Builder
###############################################################################

def build_dataloader(
    dataset,
    train: bool,
):

    return DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=train,

        num_workers=NUM_WORKERS,

        collate_fn=collate_fn,

        pin_memory=True,

        drop_last=False,

    )


###############################################################################
# Model Builder
###############################################################################

def build_model():

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
# Training Component Builder
###############################################################################

def build_training_components(
    model,
    total_steps: int,
):

    optimizer = build_optimizer(

        model=model,

        optimizer="adamw",

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )

    scheduler = build_scheduler(

        optimizer,

        scheduler="cosine",

        total_steps=total_steps * EPOCHS,

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

def create_directories():

    CHECKPOINT_ROOT.mkdir(

        parents=True,

        exist_ok=True,

    )

    LOG_ROOT.mkdir(

        parents=True,

        exist_ok=True,

    )

###############################################################################
# Checkpoint Utilities
###############################################################################

LATEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "latest.pth"
)

BEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "best.pth"
)

CSV_LOG = (
    LOG_ROOT
    / "training_log.csv"
)


###############################################################################
# Save Checkpoint
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
) -> None:
    """
    Save training checkpoint.
    """

    checkpoint = {

        "epoch": epoch,

        "train_loss": train_loss,

        "val_metrics": val_metrics,

        "best_metric": val_metrics.get(
            "minADE",
            float("inf"),
        ),

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "torch_rng_state": torch.get_rng_state(),
    }

    ###############################################################
    # Latest checkpoint
    ###############################################################

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT,
    )

    ###############################################################
    # Best checkpoint
    ###############################################################

    if best:

        torch.save(
            checkpoint,
            BEST_CHECKPOINT,
        )

    print()

    print(
        f"Checkpoint saved : {LATEST_CHECKPOINT.name}"
    )

    if best:

        print(
            f"Best model saved : {BEST_CHECKPOINT.name}"
        )


###############################################################################
# Resume Training
###############################################################################

def load_checkpoint(
    model: DSTNet,
    optimizer,
    scheduler,
):
    """
    Resume training if a checkpoint exists.

    Returns
    -------
    start_epoch
    best_metric
    """

    if not LATEST_CHECKPOINT.exists():

        print_section(
            "Checkpoint"
        )

        print("No checkpoint found.")

        return 0, float("inf")

    print_section(
        "Resuming Training"
    )

    checkpoint = torch.load(

        LATEST_CHECKPOINT,

        map_location=DEVICE,

    )

    if "torch_rng_state" in checkpoint:

        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    optimizer.load_state_dict(

        checkpoint["optimizer_state_dict"]

    )

    scheduler_state = checkpoint.get(
        "scheduler_state_dict",
        None,
    )

    if scheduler is not None and scheduler_state is not None:

        scheduler.load_state_dict(
            scheduler_state
        )

    epoch = checkpoint["epoch"]

    best_metric = checkpoint.get(
        "best_metric",
        checkpoint["val_metrics"].get(
            "minADE",
            float("inf"),
        ),
    )

    print(
        f"Resumed from epoch {epoch}"
    )

    return epoch, best_metric


###############################################################################
# CSV Logger
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


def initialize_csv():

    if CSV_LOG.exists():

        return

    with open(

        CSV_LOG,

        "w",

        newline="",

    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            CSV_HEADER
        )


def append_csv(

    epoch: int,

    train_loss: float,

    metrics: dict[str, float],

    learning_rate: float,

    epoch_time: float,

):

    with open(

        CSV_LOG,

        "a",

        newline="",

    ) as file:

        writer = csv.writer(file)

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
# Learning Rate
###############################################################################

def get_learning_rate(
    optimizer,
) -> float:

    return optimizer.param_groups[0]["lr"]


###############################################################################
# Epoch Summary
###############################################################################

def print_epoch_summary(

    epoch: int,

    train_loss: float,

    metrics: dict[str, float],

    learning_rate: float,

    epoch_time: float,

):

    print()

    print("=" * 80)

    print(
        f"Epoch {epoch} Summary"
    )

    print("=" * 80)

    print(
        f"Train Loss : {train_loss:.6f}"
    )

    print(
        f"minADE     : {metrics.get('minADE',0.0):.6f}"
    )

    print(
        f"minFDE     : {metrics.get('minFDE',0.0):.6f}"
    )

    print(
        f"MissRate   : {metrics.get('MissRate',0.0):.6f}"
    )

    print(
        f"Learning Rate : {learning_rate:.8f}"
    )

    print(
        f"Epoch Time : {epoch_time:.2f} s"
    )

###############################################################################
# Training Loop
###############################################################################

def train_one_epoch(
    *,
    epoch: int,
    model: DSTNet,
    dataloader,
    optimizer,
    scheduler,
    criterion,
):
    """
    Train DSTNet for one epoch.

    Returns
    -------
    average_loss
    """

    model.train()

    running_loss = 0.0

    num_batches = len(dataloader)

    epoch_start = time.perf_counter()

    ###########################################################################
    # Iterate over batches
    ###########################################################################

    for batch_index, batch in enumerate(

        dataloader,

        start=1,

    ):

        batch_start = time.perf_counter()

        ###############################################################
        # Move batch
        ###############################################################

        batch = move_to_device(
            batch,
            DEVICE,
        )

        ###############################################################
        # Zero gradients
        ###############################################################

        optimizer.zero_grad(
            set_to_none=True,
        )

        ###############################################################
        # Forward
        ###############################################################

        coarse_prediction, refined_prediction = model(

            agent_trajectories=batch[
                "agent_trajectories"
            ],

            lane_centerlines=batch[
                "lane_centerlines"
            ],

            positions=batch[
                "positions"
            ],

            headings=batch[
                "headings"
            ],

            graph=batch[
                "graph"
            ],

            agent_mask=batch.get(
                "agent_mask",
            ),

            lane_mask=batch.get(
                "lane_mask",
            ),
        )

        ###############################################################
        # Loss
        ###############################################################

        losses = criterion(

            prediction=coarse_prediction,

            refined_prediction=refined_prediction,

            ground_truth=batch[
                "future_trajectories"
            ],
        )

        loss = losses["loss"]

        ###############################################################
        # Backward
        ###############################################################

        loss.backward()

        ###############################################################
        # Gradient clipping
        ###############################################################

        gradient_norm = torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=GRADIENT_CLIP,

        )

        ###########################################################################
        # Gradient Monitoring
        ###########################################################################

        if torch.isnan(gradient_norm):

            raise RuntimeError(

                "Gradient norm became NaN."

            )

        if gradient_norm > 100:

            print(

                f"\n[Warning] Large gradient norm: "

                f"{gradient_norm:.2f}"

            )

        ###############################################################
        # Optimizer
        ###############################################################

        optimizer.step()

        ###############################################################
        # Scheduler
        ###############################################################

        if scheduler is not None:

            scheduler.step()

        ###############################################################
        # Statistics
        ###############################################################

        running_loss += loss.item()

        ###############################################################
        # Progress
        ###############################################################

        if (

            batch_index == 1

            or

            batch_index % 50 == 0

            or

            batch_index == num_batches

        ):

            elapsed = (

                time.perf_counter()

                - batch_start

            )

            print(

                f"[Epoch {epoch:03d}] "

                f"Batch {batch_index:05d}/{num_batches:05d} | "

                f"Loss {loss.item():8.4f} | "

                f"Proposal {losses['proposal_loss']:.4f} | "

                f"Cls {losses['classification_loss']:.4f} | "

                f"Score {losses['score_loss']:.4f} | "

                f"Ref {losses['refinement_loss']:.4f} | "

                f"Grad {float(gradient_norm):7.4f} | "

                f"LR {get_learning_rate(optimizer):.6f} | "

                f"{elapsed:.1f}s"

            )

    ###########################################################################
    # Epoch statistics
    ###########################################################################

    average_loss = (

        running_loss

        / max(

            1,

            num_batches,

        )

    )

    return average_loss

###############################################################################
# Validation Loop
###############################################################################

def validate_one_epoch(
    *,
    model: DSTNet,
    dataloader,
):
    """
    Evaluate the model on the validation dataset.

    Returns
    -------
    dict[str, float]
        {
            "minADE": ...,
            "minFDE": ...,
            "MissRate": ...
        }
    """

    print()

    print("=" * 80)

    print("Validation")

    print("=" * 80)

    ###########################################################################
    # Build evaluator
    ###########################################################################

    evaluator = Evaluator(

        model=model,

        dataloader=dataloader,

        device=DEVICE,

    )

    ###########################################################################
    # Run evaluation
    ###########################################################################

    start_time = time.perf_counter()

    metrics = evaluator.evaluate()

    validation_time = (

        time.perf_counter()

        - start_time

    )

    ###########################################################################
    # Print metrics
    ###########################################################################

    print()

    print("-" * 80)

    print("Validation Metrics")

    print("-" * 80)

    if len(metrics) == 0:

        print("No metrics returned.")

        return {}

    for key in sorted(metrics.keys()):

        print(

            f"{key:<15}"

            f"{metrics[key]:.6f}"

        )

    print()

    print(

        f"Validation Time : "

        f"{validation_time:.2f} s"

    )

    ###########################################################################
    # Sanity Checks
    ###########################################################################

    required_metrics = (

        "minADE",

        "minFDE",

        "MissRate",

    )

    for metric_name in required_metrics:

        if metric_name not in metrics:

            raise RuntimeError(

                f"Missing validation metric "

                f"'{metric_name}'."

            )

        value = metrics[metric_name]

        if not torch.isfinite(

            torch.tensor(value)

        ):

            raise RuntimeError(

                f"Metric '{metric_name}' "

                f"is NaN or Inf."

            )

    ###########################################################################
    # Return
    ###########################################################################

    return metrics

###############################################################################
# Reusable Training Pipeline
###############################################################################

def run_training(
    *,
    train_dataset,
    val_dataset,
    epochs: int = EPOCHS,
    checkpoint_root: Path | None = None,
    log_root: Path | None = None,
):
    """
    Reusable training pipeline.

    This contains the complete training logic previously implemented
    inside main().

    Parameters
    ----------
    train_dataset
        Training dataset.

    val_dataset
        Validation dataset.

    epochs
        Number of training epochs.

    checkpoint_root
        Optional checkpoint directory override.

    log_root
        Optional log directory override.
    """

    global CHECKPOINT_ROOT
    global LOG_ROOT
    global LATEST_CHECKPOINT
    global BEST_CHECKPOINT
    global CSV_LOG
    global EPOCHS

    ###########################################################################
    # Override runtime configuration
    ###########################################################################

    if checkpoint_root is not None:

        CHECKPOINT_ROOT = checkpoint_root

        LATEST_CHECKPOINT = (

            CHECKPOINT_ROOT

            / "latest.pth"

        )

        BEST_CHECKPOINT = (

            CHECKPOINT_ROOT

            / "best.pth"

        )

    if log_root is not None:

        LOG_ROOT = log_root

        CSV_LOG = (

            LOG_ROOT

            / "training_log.csv"

        )

    EPOCHS = epochs

    ###########################################################################
    # Directories
    ###########################################################################

    create_directories()

    initialize_csv()

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
        f"Training Batches   : {len(train_loader):,}"
    )

    print(
        f"Validation Batches : {len(val_loader):,}"
    )

    ###########################################################################
    # Model
    ###########################################################################

    model = build_model()

    ###########################################################################
    # Optimizer / Scheduler / Loss
    ###########################################################################

    optimizer, scheduler, criterion = (

        build_training_components(

            model,

            total_steps=len(train_loader),

        )

    )

    ###########################################################################
    # Resume
    ###########################################################################

    start_epoch, best_metric = load_checkpoint(

        model,

        optimizer,

        scheduler,

    )

    ###########################################################################
    # Early Stopping
    ###########################################################################

    epochs_without_improvement = 0

    ###########################################################################
    # Training
    ###########################################################################

    print()

    print("=" * 80)

    print("Starting Training")

    print("=" * 80)

    training_start = time.perf_counter()

    try:

        for epoch in range(

            start_epoch + 1,

            epochs + 1,

        ):

            print()

            print("=" * 80)

            print(
                f"Epoch {epoch}/{epochs}"
            )

            print("=" * 80)

            epoch_start = time.perf_counter()

            train_loss = train_one_epoch(

                epoch=epoch,

                model=model,

                dataloader=train_loader,

                optimizer=optimizer,

                scheduler=scheduler,

                criterion=criterion,

            )

            val_metrics = validate_one_epoch(

                model=model,

                dataloader=val_loader,

            )

            epoch_time = (

                time.perf_counter()

                - epoch_start

            )

            learning_rate = get_learning_rate(

                optimizer,

            )

            append_csv(

                epoch=epoch,

                train_loss=train_loss,

                metrics=val_metrics,

                learning_rate=learning_rate,

                epoch_time=epoch_time,

            )

            current_metric = val_metrics.get(

                "minADE",

                float("inf"),

            )

            is_best = (

                current_metric

                <

                best_metric

            )

            if is_best:

                best_metric = current_metric

                epochs_without_improvement = 0

                print()

                print(

                    f"✓ New Best Model "

                    f"(minADE={best_metric:.6f})"

                )

            else:

                epochs_without_improvement += 1

                print(

                    f"No improvement "

                    f"({epochs_without_improvement}/{PATIENCE})"

                )

            if epoch % SAVE_EVERY == 0:

                save_checkpoint(

                    epoch=epoch,

                    model=model,

                    optimizer=optimizer,

                    scheduler=scheduler,

                    train_loss=train_loss,

                    val_metrics=val_metrics,

                    best=is_best,

                )

            print_epoch_summary(

                epoch=epoch,

                train_loss=train_loss,

                metrics=val_metrics,

                learning_rate=learning_rate,

                epoch_time=epoch_time,

            )

            ###########################################################################
            # Early Stopping
            ###########################################################################

            if (

                EARLY_STOPPING

                and

                epochs_without_improvement >= PATIENCE

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

        print("Training Interrupted")

        print("=" * 80)

        save_checkpoint(

            epoch=epoch,

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            train_loss=train_loss,

            val_metrics=val_metrics,

            best=False,

        )

    total_training_time = (

        time.perf_counter()

        - training_start

    )

    print()

    print("=" * 80)

    print("Training Complete")

    print("=" * 80)

    print(

        f"Best minADE : "

        f"{best_metric:.6f}"

    )

    print(

        f"Total Time  : "

        f"{total_training_time / 3600:.2f} hours"

    )

    print()

    print(

        f"Checkpoints : "

        f"{CHECKPOINT_ROOT}"

    )

    print(

        f"Logs        : "

        f"{CSV_LOG}"

    )

###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet Production Training"
    )

    ###########################################################################
    # Directories
    ###########################################################################

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
        f"Training Scenes   : {len(train_dataset):,}"
    )

    print(
        f"Validation Scenes : {len(val_dataset):,}"
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
        f"Training Batches   : {len(train_loader):,}"
    )

    print(
        f"Validation Batches : {len(val_loader):,}"
    )

    ###########################################################################
    # Model
    ###########################################################################

    model = build_model()

    ###########################################################################
    # Optimizer / Scheduler / Loss
    ###########################################################################

    optimizer, scheduler, criterion = (

        build_training_components(

            model,

            total_steps=len(train_loader),

        )

    )

    ###########################################################################
    # Resume
    ###########################################################################

    start_epoch, best_metric = load_checkpoint(

        model,

        optimizer,

        scheduler,

    )

    ###########################################################################
    # Run Training
    ###########################################################################

    run_training(

        train_dataset=train_dataset,

        val_dataset=val_dataset,

        epochs=EPOCHS,

    )


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()


