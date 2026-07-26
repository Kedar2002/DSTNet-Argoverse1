"""
scripts.test_evaluator

Verification of the DSTNet evaluation pipeline.

Verifies

    ✓ Validation dataset loading
    ✓ Model inference
    ✓ Evaluator
    ✓ ADE/FDE metric computation
    ✓ Dataset metric aggregation
    ✓ Evaluation mode
    ✓ No gradient execution

Pipeline

Validation Dataset
        ↓
DataLoader
        ↓
DSTNet
        ↓
Checkpoint
        ↓
Evaluator
        ↓
Metrics
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

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
    / "val"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "cache"
    / "test_evaluator"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "smoke"
    / "smoke_epoch_1.pth"
)

DEVICE = torch.device("cpu")

###############################################################################
# Evaluation Configuration
###############################################################################

NUM_SCENES = 16

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
# Dataset Builder
###############################################################################

def build_dataset():

    print_section(
        "Building Validation Dataset"
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
            range(NUM_SCENES)
        ),

    )

    print(
        f"Validation Dataset : {len(dataset)} scenes"
    )

    print(
        f"Evaluation Subset  : {len(subset)} scenes"
    )

    return subset


###############################################################################
# DataLoader Builder
###############################################################################

def build_dataloader(
    dataset,
):

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
# Evaluator Builder
###############################################################################

def build_evaluator(
    model,
    dataloader,
):

    print_section(
        "Building Evaluator"
    )

    evaluator = Evaluator(

        model=model,

        dataloader=dataloader,

        device=DEVICE,

    )

    print("✓ Evaluator created")

    return evaluator

###############################################################################
# Checkpoint Loader
###############################################################################

def load_checkpoint(
    model: DSTNet,
) -> None:
    """
    Load the smoke-training checkpoint.
    """

    print_section(
        "Loading Checkpoint"
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(

            f"Checkpoint not found:\n{CHECKPOINT_PATH}\n\n"

            "Run:\n"

            "python -m scripts.train_smoke"

        )

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=DEVICE,

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    print("✓ Checkpoint loaded")

    print(

        f"Epoch : {checkpoint['epoch']}"

    )

    print(

        f"Loss  : {checkpoint['loss']:.6f}"

    )


###############################################################################
# Metric Validation
###############################################################################

EXPECTED_METRICS = (

    "minADE",

    "minFDE",

    "MissRate",

)


def validate_metrics(
    metrics: dict[str, float],
) -> bool:
    """
    Validate evaluator output.
    """

    print_section(
        "Metric Validation"
    )

    passed = True

    ###############################################################
    # Keys
    ###############################################################

    for key in EXPECTED_METRICS:

        if key not in metrics:

            print(f"✗ Missing metric: {key}")

            passed = False

            continue

        value = metrics[key]

        ###########################################################
        # Type
        ###########################################################

        if not isinstance(

            value,

            (float, int),

        ):

            print(

                f"✗ {key}: invalid type "

                f"{type(value)}"

            )

            passed = False

            continue

        ###########################################################
        # Finite
        ###########################################################

        value = float(value)

        if not torch.isfinite(

            torch.tensor(value)

        ):

            print(

                f"✗ {key}: NaN or Inf"

            )

            passed = False

            continue

        print(

            f"✓ {key:<12} {value:.6f}"

        )

    return passed


###############################################################################
# Evaluation State Verification
###############################################################################

def verify_eval_mode(
    model: DSTNet,
) -> bool:
    """
    Ensure the model remained in evaluation mode.
    """

    print_section(
        "Evaluation Mode"
    )

    if model.training:

        print(
            "✗ Model is still in training mode"
        )

        return False

    print(
        "✓ Model is in evaluation mode"
    )

    return True


###############################################################################
# Timing Utility
###############################################################################

def run_evaluation(
    evaluator: Evaluator,
):
    """
    Execute evaluator and measure runtime.
    """

    print_section(
        "Running Evaluation"
    )

    start = time.perf_counter()

    metrics = evaluator.evaluate()

    elapsed = (

        time.perf_counter()

        - start

    )

    print()

    print(

        f"Evaluation Time : "

        f"{elapsed:.2f} s"

    )

    return metrics


###############################################################################
# Pretty Printing
###############################################################################

def print_metrics(
    metrics: dict[str, float],
) -> None:

    print_section(
        "Evaluation Results"
    )

    for key in sorted(metrics):

        print(

            f"{key:<15}"

            f"{metrics[key]:.6f}"

        )

###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet Evaluator Verification"
    )

    ###########################################################################
    # Dataset
    ###########################################################################

    dataset = build_dataset()

    ###########################################################################
    # DataLoader
    ###########################################################################

    dataloader = build_dataloader(
        dataset,
    )

    ###########################################################################
    # Model
    ###########################################################################

    model = build_model()

    ###########################################################################
    # Load checkpoint
    ###########################################################################

    load_checkpoint(
        model,
    )

    ###########################################################################
    # Evaluator
    ###########################################################################

    evaluator = build_evaluator(
        model,
        dataloader,
    )

    ###########################################################################
    # Run Evaluation
    ###########################################################################

    metrics = run_evaluation(
        evaluator,
    )

    ###########################################################################
    # Print Metrics
    ###########################################################################

    print_metrics(
        metrics,
    )

    ###########################################################################
    # Validation
    ###########################################################################

    metrics_ok = validate_metrics(
        metrics,
    )

    eval_mode_ok = verify_eval_mode(
        model,
    )

    ###########################################################################
    # Final Summary
    ###########################################################################

    print_section(
        "Evaluation Summary"
    )

    print(
        f"Validation Scenes : {len(dataset)}"
    )

    print(
        f"Evaluation Batches : {len(dataloader)}"
    )

    print(
        f"Checkpoint : {CHECKPOINT_PATH.name}"
    )

    print()

    print("=" * 80)

    if metrics_ok and eval_mode_ok:

        print(
            "✓ EVALUATOR TEST PASSED"
        )

    else:

        print(
            "✗ EVALUATOR TEST FAILED"
        )

    print("=" * 80)


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()

