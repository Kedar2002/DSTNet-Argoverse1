"""
scripts.test_forward

End-to-end forward-pass verification for DSTNet.

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
DataLoader
    ↓
collate_fn
    ↓
DSTNet
    ↓
Prediction
    ↓
RefinedPrediction

Checks
------
✓ Dataset loading
✓ Batch loading
✓ Model initialization
✓ Parameter count
✓ Forward pass
✓ Tensor shapes
✓ NaN / Inf detection
✓ Prediction integrity
✓ Refinement integrity
"""

from __future__ import annotations

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
from datasets.transforms import Identity

###############################################################################
# Model
###############################################################################

from models.dstnet import DSTNet

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
    / "test_forward"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 2

###############################################################################
# Helper Functions
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


def print_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:

    print(
        f"{name:<28}"
        f"{tuple(tensor.shape)}"
        f"    {tensor.dtype}"
    )


def check_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:
    """
    Verify tensor integrity.
    """

    if torch.isnan(
        tensor,
    ).any():

        raise RuntimeError(
            f"{name} contains NaNs."
        )

    if torch.isinf(
        tensor,
    ).any():

        raise RuntimeError(
            f"{name} contains Infs."
        )


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
# Dataset Builder
###############################################################################

def build_dataset() -> ArgoverseDataset:

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

    print(
        "Dataset Size :",
        len(dataset),
    )

    return dataset


###############################################################################
# DataLoader Builder
###############################################################################

def build_dataloader(
    dataset: ArgoverseDataset,
) -> DataLoader:

    print_section(
        "Building DataLoader"
    )

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        collate_fn=collate_fn,

    )

    print(
        "Batch Size :",
        BATCH_SIZE,
    )

    return loader


###############################################################################
# Model Builder
###############################################################################

def build_model() -> DSTNet:

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

    model.eval()

    total, trainable = count_parameters(
        model,
    )

    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Total Parameters    : {total:,}"
    )

    print(
        f"Trainable Parameters: {trainable:,}"
    )

    return model

###############################################################################
# Forward Pass
###############################################################################

def run_forward_pass(
    model: DSTNet,
    loader: DataLoader,
):

    print_header(
        "Loading Batch"
    )

    batch = next(
        iter(loader)
    )

    ###########################################################################
    # Input Summary
    ###########################################################################

    print_section(
        "Input Tensor Shapes"
    )

    print_tensor(
        "agent_trajectories",
        batch["agent_trajectories"],
    )

    print_tensor(
        "future_trajectories",
        batch["future_trajectories"],
    )

    print_tensor(
        "lane_centerlines",
        batch["lane_centerlines"],
    )

    print_tensor(
        "positions",
        batch["positions"],
    )

    print_tensor(
        "headings",
        batch["headings"],
    )

    print_tensor(
        "agent_mask",
        batch["agent_mask"],
    )

    print_tensor(
        "lane_mask",
        batch["lane_mask"],
    )

    ###########################################################################
    # Move to device
    ###########################################################################

    print_section(
        "Moving Batch To Device"
    )

    agent_trajectories = batch[
        "agent_trajectories"
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

    print("✓ Batch transferred")

    ###########################################################################
    # Forward Pass
    ###########################################################################

    print_header(
        "Running DSTNet Forward Pass"
    )

    start = time.perf_counter()

    with torch.no_grad():

        coarse_prediction, refined_prediction = model(

            agent_trajectories=agent_trajectories,

            lane_centerlines=lane_centerlines,

            positions=positions,

            headings=headings,

            graph=graph,

            agent_mask=agent_mask,

            lane_mask=lane_mask,

        )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    print()

    print(
        f"Forward Pass Time : "
        f"{elapsed:.2f} ms"
    )

    ###########################################################################
    # Prediction Diagnostics
    ###########################################################################

    print_header(
        "Prediction Shapes"
    )

    print_tensor(
        "coarse trajectories",
        coarse_prediction.trajectories,
    )

    print_tensor(
        "coarse scores",
        coarse_prediction.scores,
    )

    print()

    print_tensor(
        "refined trajectories",
        refined_prediction.trajectories,
    )

    print_tensor(
        "refined scores",
        refined_prediction.scores,
    )

    if refined_prediction.offsets is not None:

        print_tensor(
            "trajectory offsets",
            refined_prediction.offsets,
        )

    ###########################################################################
    # Tensor Validation
    ###########################################################################

    print_header(
        "Tensor Integrity"
    )

    check_tensor(
        "coarse trajectories",
        coarse_prediction.trajectories,
    )

    print(
        "✓ coarse trajectories"
    )

    check_tensor(
        "coarse scores",
        coarse_prediction.scores,
    )

    print(
        "✓ coarse scores"
    )

    check_tensor(
        "refined trajectories",
        refined_prediction.trajectories,
    )

    print(
        "✓ refined trajectories"
    )

    check_tensor(
        "refined scores",
        refined_prediction.scores,
    )

    print(
        "✓ refined scores"
    )

    if refined_prediction.offsets is not None:

        check_tensor(
            "offsets",
            refined_prediction.offsets,
        )

        print(
            "✓ offsets"
        )

    ###########################################################################
    # Return everything for final verification
    ###########################################################################

    return (

        batch,

        coarse_prediction,

        refined_prediction,

    )

###############################################################################
# Output Validation
###############################################################################

def validate_outputs(
    batch,
    coarse_prediction,
    refined_prediction,
) -> None:

    print_header(
        "Output Validation"
    )

    B = batch["agent_trajectories"].shape[0]
    N = batch["agent_trajectories"].shape[1]

    ###########################################################################
    # Coarse Prediction
    ###########################################################################

    assert (
        coarse_prediction.trajectories.ndim == 5
    ), "Coarse trajectories should have shape (B,N,K,T,2)."

    assert (
        coarse_prediction.scores.ndim == 3
    ), "Coarse scores should have shape (B,N,K)."

    assert (
        coarse_prediction.trajectories.shape[0] == B
    )

    assert (
        coarse_prediction.trajectories.shape[1] == N
    )

    ###########################################################################
    # Refined Prediction
    ###########################################################################

    assert (
        refined_prediction.trajectories.ndim == 5
    ), "Refined trajectories should have shape (B,N,K,T,2)."

    assert (
        refined_prediction.scores.ndim == 3
    ), "Refined scores should have shape (B,N,K)."

    assert (
        refined_prediction.trajectories.shape
        ==
        coarse_prediction.trajectories.shape
    )

    assert (
        refined_prediction.scores.shape
        ==
        coarse_prediction.scores.shape
    )

    if refined_prediction.offsets is not None:

        assert (
            refined_prediction.offsets.shape
            ==
            coarse_prediction.trajectories.shape
        )

    print("✓ Prediction dimensions")

    print("✓ Refinement dimensions")

    print("✓ Batch consistency")

    print("✓ Forward pass successful")


###############################################################################
# Summary
###############################################################################

def print_summary(
    batch,
    coarse_prediction,
) -> None:

    print_header(
        "Forward Pass Summary"
    )

    print(
        f"Batch Size          : {batch['agent_trajectories'].shape[0]}"
    )

    print(
        f"Agents / Scene      : {batch['agent_trajectories'].shape[1]}"
    )

    print(
        f"Lanes / Scene       : {batch['lane_centerlines'].shape[1]}"
    )

    print(
        f"Prediction Modes    : {coarse_prediction.scores.shape[-1]}"
    )

    print(
        f"Prediction Horizon  : {coarse_prediction.trajectories.shape[-2]}"
    )

    print(
        f"Coordinate Dim      : {coarse_prediction.trajectories.shape[-1]}"
    )


###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet Forward Pass Verification"
    )

    dataset = build_dataset()

    loader = build_dataloader(
        dataset,
    )

    model = build_model()

    (
        batch,
        coarse_prediction,
        refined_prediction,
    ) = run_forward_pass(
        model,
        loader,
    )

    validate_outputs(
        batch,
        coarse_prediction,
        refined_prediction,
    )

    print_summary(
        batch,
        coarse_prediction,
    )

    print()

    print("=" * 80)

    print("✓ DSTNET FORWARD TEST PASSED")

    print("=" * 80)


###############################################################################

if __name__ == "__main__":

    main()
