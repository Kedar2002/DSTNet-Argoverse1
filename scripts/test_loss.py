"""
scripts.test_loss

End-to-end training pipeline verification for DSTNet.

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
    ↓
TotalLoss
    ↓
Backward
    ↓
Optimizer Step

Checks
------
✓ Dataset loading
✓ Model construction
✓ Optimizer construction
✓ Loss construction
✓ Forward pass
✓ Loss computation
✓ Backward pass
✓ Gradient integrity
✓ Optimizer step
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
# Training
###############################################################################

from losses.total_loss import TotalLoss

from engine.optimizer import (
    build_optimizer,
    optimizer_summary,
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
    / "test_loss"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 2

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-2

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


def print_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:

    print(
        f"{name:<28}"
        f"{tuple(tensor.shape)}"
        f"    {tensor.dtype}"
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

    model.train()

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
# Optimizer Builder
###############################################################################

def build_training_optimizer(
    model: DSTNet,
):

    print_section(
        "Building Optimizer"
    )

    optimizer = build_optimizer(

        model=model,

        optimizer="adamw",

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )

    print(
        optimizer_summary(
            optimizer,
        )
    )

    return optimizer


###############################################################################
# Loss Builder
###############################################################################

def build_loss() -> TotalLoss:

    print_section(
        "Building Loss Function"
    )

    criterion = TotalLoss()

    print(
        criterion,
    )

    return criterion

###############################################################################
# Forward + Loss
###############################################################################

def run_training_step(
    model: DSTNet,
    criterion: TotalLoss,
    loader: DataLoader,
):

    print_header(
        "Loading Training Batch"
    )

    batch = next(
        iter(loader)
    )

    ###########################################################################
    # Input Shapes
    ###########################################################################

    print_section(
        "Batch Tensors"
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
    # Device Transfer
    ###########################################################################

    print_section(
        "Moving Batch To Device"
    )

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

    print("✓ Batch transferred")

    ###########################################################################
    # Forward Pass
    ###########################################################################

    print_header(
        "Running Forward Pass"
    )

    start = time.perf_counter()

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

    print(
        f"\nForward Time : {elapsed:.2f} ms"
    )

    ###########################################################################
    # Compute Loss
    ###########################################################################

    print_header(
        "Computing Loss"
    )

    loss_dict = criterion(

        prediction=coarse_prediction,

        refined_prediction=refined_prediction,

        ground_truth=future_trajectories,

    )

    expected = (
        loss_dict["proposal_loss"]
        + loss_dict["classification_loss"]
        + 0.5 * loss_dict["score_loss"]
        + loss_dict["refinement_loss"]
    )

    print("\nExpected Total :", expected.item())
    print("Returned Total :", loss_dict["loss"].item())

    ###########################################################################
    # Display Losses
    ###########################################################################

    print_section(
        "Loss Components"
    )

    for name, value in loss_dict.items():

        if torch.is_tensor(value):

            scalar = value.detach().item()

        else:

            scalar = float(value)

        print(
            f"{name:<24}: {scalar:.6f}"
        )

    ###########################################################################
    # Finite Check
    ###########################################################################

    print_section(
        "Loss Validation"
    )

    for name, value in loss_dict.items():

        if not torch.is_tensor(value):
            continue

        if torch.isnan(value).any():

            raise RuntimeError(
                f"{name} contains NaN."
            )

        if torch.isinf(value).any():

            raise RuntimeError(
                f"{name} contains Inf."
            )

        print(
            f"✓ {name}"
        )

    ###########################################################################
    # Return
    ###########################################################################

    return (

        batch,

        loss_dict,

        coarse_prediction,

        refined_prediction,

    )

###############################################################################
# Backward Pass
###############################################################################

def verify_training(
    model: DSTNet,
    optimizer,
    loss_dict,
) -> None:

    print_header(
        "Backward Pass"
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    total_loss = loss_dict["loss"]

    total_loss.backward()

    print("✓ Backward completed")

    ###########################################################################
    # Gradient Validation
    ###########################################################################

    print_section(
        "Gradient Validation"
    )

    parameter_count = 0

    gradient_count = 0

    max_gradient = 0.0

    mean_gradient = 0.0

    for name, parameter in model.named_parameters():

        parameter_count += 1

        if parameter.grad is None:

            continue

        gradient_count += 1

        grad = parameter.grad

        if torch.isnan(grad).any():

            raise RuntimeError(
                f"{name} gradient contains NaN."
            )

        if torch.isinf(grad).any():

            raise RuntimeError(
                f"{name} gradient contains Inf."
            )

        grad_abs = grad.detach().abs()

        mean_gradient += grad_abs.mean().item()

        max_gradient = max(
            max_gradient,
            grad_abs.max().item(),
        )

    if gradient_count == 0:

        raise RuntimeError(
            "No gradients were produced."
        )

    mean_gradient /= gradient_count

    print(
        f"Parameters          : {parameter_count}"
    )

    print(
        f"Gradients           : {gradient_count}"
    )

    print(
        f"Mean Gradient       : {mean_gradient:.6e}"
    )

    print(
        f"Maximum Gradient    : {max_gradient:.6e}"
    )

    ###########################################################################
    # Optimizer Step
    ###########################################################################

    print_header(
        "Optimizer Step"
    )

    ###########################################################################
    # Save one parameter before update
    ###########################################################################

    first_parameter = next(
        model.parameters()
    )

    before = first_parameter.detach().clone()

    optimizer.step()

    after = first_parameter.detach()

    difference = torch.norm(
        after - before,
    ).item()

    print(
        f"Parameter Update L2 : {difference:.6e}"
    )

    if difference == 0.0:

        raise RuntimeError(
            "Optimizer step did not modify parameters."
        )

    print(
        "✓ Parameters updated"
    )


###############################################################################
# Summary
###############################################################################

def print_summary(
    loss_dict,
) -> None:

    print_header(
        "Training Summary"
    )

    print(
        f"Total Loss          : {loss_dict['loss'].item():.6f}"
    )

    print(
        f"Proposal Loss       : {loss_dict['proposal_loss'].item():.6f}"
    )

    print(
        f"Classification Loss : {loss_dict['classification_loss'].item():.6f}"
    )

    print(
        f"Score Loss          : {loss_dict['score_loss'].item():.6f}"
    )

    print(
        f"Refinement Loss     : {loss_dict['refinement_loss'].item():.6f}"
    )


###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet Loss Verification"
    )

    dataset = build_dataset()

    loader = build_dataloader(
        dataset,
    )

    model = build_model()

    optimizer = build_training_optimizer(
        model,
    )

    criterion = build_loss()

    (
        batch,
        loss_dict,
        coarse_prediction,
        refined_prediction,
    ) = run_training_step(

        model,

        criterion,

        loader,

    )

    verify_training(

        model,

        optimizer,

        loss_dict,

    )

    print_summary(
        loss_dict,
    )

    print()

    print("=" * 80)

    print("✓ LOSS TEST PASSED")

    print("=" * 80)


###############################################################################

if __name__ == "__main__":

    main()
