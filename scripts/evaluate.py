"""
scripts.evaluate

Production evaluation script for DSTNet.

Responsibilities
----------------
- Load trained checkpoint
- Evaluate validation or test dataset
- Compute trajectory forecasting metrics
- Report inference statistics
- Export evaluation results

Pipeline

Checkpoint
     │
     ▼
DSTNet
     │
     ▼
Validation/Test Dataset
     │
     ▼
DataLoader
     │
     ▼
Evaluator
     │
     ▼
Metrics
     │
     ▼
JSON / CSV Report
"""

from __future__ import annotations

import csv
import json
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
from datasets.transforms import build_eval_transform

###############################################################################
# Model
###############################################################################

from models.dstnet import DSTNet

###############################################################################
# Evaluation
###############################################################################

from engine.evaluator import Evaluator

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

VAL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
    / "val"
)

TEST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
    / "test"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "cache"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "training"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
)

DEFAULT_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "best.pth"
)

###############################################################################
# Runtime
###############################################################################

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

BATCH_SIZE = 2

NUM_WORKERS = 0

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
# Directory
###############################################################################

def create_directories():

    RESULT_ROOT.mkdir(

        parents=True,

        exist_ok=True,

    )

###############################################################################
# Scene Preprocessor
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

    dataset = ArgoverseDataset(

        root=root,

        parser=parser,

        preprocessor=preprocessor,

        transform=build_eval_transform(),

        cache=cache,

    )

    return dataset

###############################################################################
# DataLoader Builder
###############################################################################

def build_dataloader(
    dataset,
):

    return DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        collate_fn=collate_fn,

        pin_memory=False,

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
# Checkpoint Loader
###############################################################################

def load_checkpoint(
    model: DSTNet,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
):
    """
    Load a trained DSTNet checkpoint.
    """

    print_section(
        "Loading Checkpoint"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(

            f"Checkpoint not found:\n"

            f"{checkpoint_path}"

        )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE,

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    print("✓ Checkpoint Loaded")

    print()

    print(
        f"Epoch        : "
        f"{checkpoint.get('epoch','N/A')}"
    )

    print(
        f"Train Loss   : "
        f"{checkpoint.get('train_loss',0.0):.6f}"
    )

    if "val_metrics" in checkpoint:

        metrics = checkpoint["val_metrics"]

        print()

        print("Stored Validation Metrics")

        for key in sorted(metrics):

            print(

                f"{key:<15}"

                f"{metrics[key]:.6f}"

            )

    return checkpoint


###############################################################################
# Evaluator Builder
###############################################################################

def build_evaluator(
    model: DSTNet,
    dataloader,
):

    print_section(
        "Evaluator"
    )

    evaluator = Evaluator(

        model=model,

        dataloader=dataloader,

        device=DEVICE,

    )

    print("✓ Evaluator Ready")

    return evaluator


###############################################################################
# Evaluation Runner
###############################################################################

def execute_evaluation(
    evaluator: Evaluator,
):
    """
    Execute evaluation and measure runtime.
    """

    print_section(
        "Running Evaluation"
    )

    start_time = time.perf_counter()

    metrics = evaluator.evaluate()

    total_time = (

        time.perf_counter()

        - start_time

    )

    return (

        metrics,

        total_time,

    )


###############################################################################
# Inference Statistics
###############################################################################

def compute_statistics(
    *,
    dataset,
    total_time: float,
):

    num_scenes = len(dataset)

    if total_time <= 0:

        fps = 0.0

        latency = 0.0

    else:

        fps = (

            num_scenes

            / max(total_time, 1e-8)

        )

        latency = (

            1000.0

            * total_time

            / max(num_scenes, 1)

        )

    return {

        "num_scenes": num_scenes,

        "total_time": total_time,

        "fps": fps,

        "latency_ms": latency,

    }


###############################################################################
# Metric Validation
###############################################################################

EXPECTED_METRICS = (

    "minADE",

    "minFDE",

    "MissRate",

)


def validate_metrics(
    metrics,
):
    """
    Validate evaluation output.
    """

    print_section(
        "Metric Validation"
    )

    passed = True

    for key in EXPECTED_METRICS:

        if key not in metrics:

            print(

                f"✗ Missing metric: {key}"

            )

            passed = False

            continue

        value = float(metrics[key])

        if not torch.isfinite(

            torch.tensor(value)

        ):

            print(

                f"✗ {key}: NaN/Inf"

            )

            passed = False

            continue

        print(

            f"✓ {key:<12}"

            f"{value:.6f}"

        )

    return passed


###############################################################################
# Result Printer
###############################################################################

def print_results(
    metrics,
    statistics,
):

    print_section(
        "Evaluation Results"
    )

    print()

    print("Forecast Metrics")

    print("-" * 40)

    for key in sorted(metrics):

        print(

            f"{key:<15}"

            f"{metrics[key]:.6f}"

        )

    print()

    print("Inference Statistics")

    print("-" * 40)

    print(

        f"Scenes        : "

        f"{statistics['num_scenes']}"

    )

    print(

        f"Total Time    : "

        f"{statistics['total_time']:.2f} s"

    )

    print(

        f"FPS           : "

        f"{statistics['fps']:.2f}"

    )

    print(

        f"Latency/Scene : "

        f"{statistics['latency_ms']:.2f} ms"

    )

###############################################################################
# Result Files
###############################################################################

JSON_RESULT = (
    RESULT_ROOT
    / "evaluation_results.json"
)

CSV_RESULT = (
    RESULT_ROOT
    / "evaluation_results.csv"
)

###############################################################################
# JSON Export
###############################################################################

def save_json(
    *,
    checkpoint: dict,
    metrics: dict,
    statistics: dict,
) -> None:
    """
    Save evaluation results as JSON.
    """

    output = {

        "checkpoint": {

            "epoch": checkpoint.get(
                "epoch",
            ),

            "train_loss": checkpoint.get(
                "train_loss",
            ),

        },

        "metrics": {

            key: float(value)

            for key, value in metrics.items()

        },

        "statistics": {

            key: float(value)

            if isinstance(value, (int, float))

            else value

            for key, value in statistics.items()

        },

    }

    with open(

        JSON_RESULT,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            output,

            file,

            indent=4,

        )

    print()

    print(

        f"JSON saved : "

        f"{JSON_RESULT}"

    )

###############################################################################
# CSV Export
###############################################################################

def save_csv(
    *,
    checkpoint: dict,
    metrics: dict,
    statistics: dict,
) -> None:
    """
    Save evaluation summary as CSV.
    """

    with open(

        CSV_RESULT,

        "w",

        newline="",

        encoding="utf-8",

    ) as file:

        writer = csv.writer(file)

        writer.writerow(

            [

                "Metric",

                "Value",

            ]

        )

        ###############################################################
        # Checkpoint
        ###############################################################

        writer.writerow(

            [

                "Epoch",

                checkpoint.get(
                    "epoch",
                ),

            ]

        )

        writer.writerow(

            [

                "TrainLoss",

                checkpoint.get(
                    "train_loss",
                ),

            ]

        )

        ###############################################################
        # Metrics
        ###############################################################

        for key in sorted(metrics):

            writer.writerow(

                [

                    key,

                    float(metrics[key]),

                ]

            )

        ###############################################################
        # Statistics
        ###############################################################

        for key in sorted(statistics):

            writer.writerow(

                [

                    key,

                    statistics[key],

                ]

            )

    print(

        f"CSV saved  : "

        f"{CSV_RESULT}"

    )

###############################################################################
# Experiment Summary
###############################################################################

def print_summary(
    *,
    checkpoint: dict,
    metrics: dict,
    statistics: dict,
) -> None:
    """
    Final evaluation summary.
    """

    print()

    print("=" * 80)

    print("Evaluation Summary")

    print("=" * 80)

    print()

    print(

        f"Checkpoint Epoch : "

        f"{checkpoint.get('epoch')}"

    )

    print(

        f"Training Loss    : "

        f"{checkpoint.get('train_loss'):.6f}"

    )

    print()

    print("Forecast Metrics")

    print("-" * 40)

    print(

        f"minADE    : "

        f"{metrics['minADE']:.6f}"

    )

    print(

        f"minFDE    : "

        f"{metrics['minFDE']:.6f}"

    )

    print(

        f"MissRate  : "

        f"{metrics['MissRate']:.6f}"

    )

    print()

    print("Inference")

    print("-" * 40)

    print(

        f"Scenes     : "

        f"{statistics['num_scenes']}"

    )

    print(

        f"FPS        : "

        f"{statistics['fps']:.2f}"

    )

    print(

        f"Latency    : "

        f"{statistics['latency_ms']:.2f} ms"

    )

    print(

        f"Total Time : "

        f"{statistics['total_time']:.2f} s"

    )

###############################################################################
# Result Saver
###############################################################################

def save_results(
    *,
    checkpoint: dict,
    metrics: dict,
    statistics: dict,
) -> None:
    """
    Save all evaluation outputs.
    """

    save_json(

        checkpoint=checkpoint,

        metrics=metrics,

        statistics=statistics,

    )

    save_csv(

        checkpoint=checkpoint,

        metrics=metrics,

        statistics=statistics,

    )

###############################################################################
# Reusable Evaluation Pipeline
###############################################################################

def run_evaluation(
    *,
    dataset,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    result_root: Path | None = None,
):
    """
    Reusable evaluation pipeline.

    This contains the complete logic previously inside main().
    """

    global RESULT_ROOT
    global JSON_RESULT
    global CSV_RESULT

    ###########################################################################
    # Override result directory
    ###########################################################################

    if result_root is not None:

        RESULT_ROOT = result_root

        JSON_RESULT = (

            RESULT_ROOT

            / "evaluation_results.json"

        )

        CSV_RESULT = (

            RESULT_ROOT

            / "evaluation_results.csv"

        )

    create_directories()

    ###########################################################################
    # DataLoader
    ###########################################################################

    print_section(
        "Building DataLoader"
    )

    dataloader = build_dataloader(
        dataset,
    )

    print(
        f"Batches : {len(dataloader):,}"
    )

    ###########################################################################
    # Model
    ###########################################################################

    model = build_model()

    ###########################################################################
    # Checkpoint
    ###########################################################################

    checkpoint_data = load_checkpoint(

        model,

        checkpoint,

    )

    ###########################################################################
    # Evaluator
    ###########################################################################

    evaluator = build_evaluator(

        model,

        dataloader,

    )

    ###########################################################################
    # Evaluation
    ###########################################################################

    metrics, total_time = execute_evaluation(

        evaluator,

    )

    ###########################################################################
    # Statistics
    ###########################################################################

    statistics = compute_statistics(

        dataset=dataset,

        total_time=total_time,

    )

    ###########################################################################
    # Validation
    ###########################################################################

    passed = validate_metrics(

        metrics,

    )

    ###########################################################################
    # Print
    ###########################################################################

    print_results(

        metrics,

        statistics,

    )

    ###########################################################################
    # Save
    ###########################################################################

    save_results(

        checkpoint=checkpoint_data,

        metrics=metrics,

        statistics=statistics,

    )

    ###########################################################################
    # Summary
    ###########################################################################

    print_summary(

        checkpoint=checkpoint_data,

        metrics=metrics,

        statistics=statistics,

    )

    return passed

###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet Production Evaluation"
    )

    ###########################################################################
    # Directories
    ###########################################################################

    create_directories()

    ###########################################################################
    # Dataset
    ###########################################################################

    print_section(
        "Building Dataset"
    )

    dataset = build_dataset(
        VAL_ROOT,
    )

    print(
        f"Validation Scenes : {len(dataset):,}"
    )

    run_evaluation(

    dataset=dataset,

    checkpoint=DEFAULT_CHECKPOINT,

    result_root=RESULT_ROOT,

)


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()


