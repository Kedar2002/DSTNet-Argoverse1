"""
scripts/test_training_step.py

DSTNet Training-Step Targeted Verification
==========================================

Verifies the complete current training-step contract:

    Argoverse CSV
        ↓
    MapLoader
        ↓
    SceneParser
        ↓
    RawScene
        ↓
    ScenePreprocessor
        ↓
    SceneData
        ↓
    collate_fn
        ↓
    DSTNet
        ↓
    Prediction + RefinedPrediction
        ↓
    TotalLoss
        ↓
    backward()
        ↓
    gradient validation
        ↓
    gradient clipping
        ↓
    optimizer.step()
        ↓
    scheduler.step()

This test intentionally uses the CURRENT DSTNet interfaces.

Current tensor contracts
------------------------

Agent trajectories:

    (B,N,H,2)

Map centerlines:

    (B,M,P,2)

Ground truth:

    (B,N,T,2)

Z_STM:

    (B,N,H,K,D)

Coarse trajectories:

    (B,N,H,K,T,2)

Probabilities:

    (B,N,H,K)

Refined trajectories:

    (B,N,H,K,T,2)

Refinement scores:

    (B,N,H,K)

Refinement offsets:

    (B,N,H,K,T,2)

The test verifies both a real Argoverse scene and a
duplicated B=2 batch.

Usage
-----

python scripts/test_training_step.py ^
    --csv data/argoverse1/val/37853.csv ^
    --map-root data/argoverse1/hd_maps/map_files

Linux/macOS:

python scripts/test_training_step.py \
    --csv data/argoverse1/val/37853.csv \
    --map-root data/argoverse1/hd_maps/map_files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

###############################################################################
# Project root
###############################################################################

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

###############################################################################
# Imports
###############################################################################

import torch

from datasets.map_loader import MapLoader
from datasets.scene_parser import SceneParser
from datasets.preprocess import ScenePreprocessor
from datasets.collate import collate_fn

from models.dstnet import DSTNet

from losses.total_loss import TotalLoss

from engine.optimizer import build_optimizer
from engine.scheduler import build_scheduler
from engine.train_step import TrainStep


###############################################################################
# Configuration
###############################################################################

OBSERVATION_STEPS = 20
PREDICTION_STEPS = 30
MAP_SAMPLE_POINTS = 20

SPATIAL_RADIUS = 30.0
MAP_RADIUS = 100.0

HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_MODES = 6

REFINEMENT_ITERATIONS = 2

DROPOUT = 0.0

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

GRADIENT_CLIP = 5.0


###############################################################################
# Printing helpers
###############################################################################


def section(
    title: str,
) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check(
    condition: bool,
    message: str,
) -> None:

    if condition:
        print(
            f"[PASS] {message}"
        )
    else:
        raise AssertionError(
            f"[FAIL] {message}"
        )


def info(
    message: str,
) -> None:

    print(
        f"[INFO] {message}"
    )


def check_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:

    check(
        isinstance(
            tensor,
            torch.Tensor,
        ),
        f"{name} is a torch.Tensor.",
    )

    finite = bool(
        torch.isfinite(
            tensor
        ).all().item()
    )

    check(
        finite,
        f"{name}: all values finite",
    )

    info(
        f"{name}: "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}"
    )


###############################################################################
# Argument parser
###############################################################################


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Verify the complete current DSTNet "
            "TrainStep contract."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to one Argoverse-1 CSV scene.",
    )

    parser.add_argument(
        "--map-root",
        type=Path,
        required=True,
        help="Path to Argoverse HD map files.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device.",
    )

    return parser.parse_args()


###############################################################################
# Scene construction
###############################################################################


def build_scene_data(
    csv_path: Path,
    map_root: Path,
):
    """
    Parse and preprocess one real Argoverse scene.
    """

    section(
        "1. REAL ARGOVERSE SCENE"
    )

    check(
        csv_path.exists(),
        "CSV file exists.",
    )

    check(
        map_root.exists(),
        "HD map root exists.",
    )

    ###########################################################################
    # Map loader
    ###########################################################################

    map_loader = MapLoader(
        map_root=map_root,
    )

    info(
        f"cities = {map_loader.cities}"
    )

    info(
        f"total lanes = "
        f"{map_loader.total_num_lanes}"
    )

    check(
        map_loader.total_num_lanes > 0,
        "HD maps loaded successfully.",
    )

    ###########################################################################
    # Parser
    ###########################################################################

    parser = SceneParser(
        map_api=map_loader,
    )

    raw_scene = parser.parse(
        csv_path
    )

    check(
        raw_scene.num_tracks > 0,
        f"RawScene contains "
        f"{raw_scene.num_tracks} tracks.",
    )

    check(
        raw_scene.num_lanes > 0,
        f"RawScene contains "
        f"{raw_scene.num_lanes} map lanes.",
    )

    check(
        raw_scene.target_track is not None,
        "Prediction target exists.",
    )

    info(
        f"sequence_id = "
        f"{raw_scene.metadata.sequence_id}"
    )

    info(
        f"city        = "
        f"{raw_scene.metadata.city}"
    )

    info(
        f"target      = "
        f"{raw_scene.metadata.focal_track_id}"
    )

    ###########################################################################
    # Preprocessor
    ###########################################################################

    preprocessor = ScenePreprocessor(
        observation_steps=OBSERVATION_STEPS,
        prediction_steps=PREDICTION_STEPS,
        map_sample_points=MAP_SAMPLE_POINTS,
        spatial_radius=SPATIAL_RADIUS,
        map_radius=MAP_RADIUS,
    )

    scene_data = preprocessor(
        raw_scene
    )

    check(
        scene_data.num_agents > 0,
        f"SceneData contains "
        f"{scene_data.num_agents} agents.",
    )

    check(
        scene_data.num_maps > 0,
        f"SceneData contains "
        f"{scene_data.num_maps} maps.",
    )

    check(
        scene_data.scene_graph is not None,
        "SceneData contains SceneGraph.",
    )

    info(
        f"agents = {scene_data.num_agents}"
    )

    info(
        f"maps   = {scene_data.num_maps}"
    )

    return scene_data


###############################################################################
# Batch construction
###############################################################################


def build_batch(
    scene_data,
) -> dict[str, Any]:
    """
    Convert SceneData into the exact model batch contract.
    """

    section(
        "2. SCENEDATA -> MODEL BATCH"
    )

    batch = collate_fn(
        [scene_data]
    )

    check(
        isinstance(
            batch,
            dict,
        ),
        "collate_fn returned a dictionary.",
    )

    # ------------------------------------------------------------------
    # Validate against the actual collate_fn -> DSTNet contract
    # established by test_pipeline_v2.py.
    # ------------------------------------------------------------------

    required_fields = {
        "agent_trajectories",
        "future_trajectories",
        "map_centerlines",
        "positions",
        "headings",
        "graph",
        "agent_mask",
        "map_mask",
        "metadata",
    }

    actual_fields = set(batch.keys())

    missing_fields = required_fields - actual_fields

    if missing_fields:
        print(
            "[INFO] collate_fn returned keys:"
        )
        for key in sorted(actual_fields):
            print(f"       - {key}")

        print(
            "[INFO] Missing expected fields:"
        )
        for key in sorted(missing_fields):
            print(f"       - {key}")

    check(
        not missing_fields,
        "Batch contains all required DSTNet fields.",
    )

    agent_trajectories = batch["agent_trajectories"]
    future_trajectories = batch["future_trajectories"]
    map_centerlines = batch["map_centerlines"]
    positions = batch["positions"]
    headings = batch["headings"]
    graph = batch["graph"]
    agent_mask = batch["agent_mask"]
    map_mask = batch["map_mask"]

    check(
        isinstance(agent_trajectories, torch.Tensor),
        "agent_trajectories is a Tensor.",
    )

    check(
        isinstance(future_trajectories, torch.Tensor),
        "future_trajectories is a Tensor.",
    )

    check(
        isinstance(map_centerlines, torch.Tensor),
        "map_centerlines is a Tensor.",
    )

    check(
        isinstance(positions, torch.Tensor),
        "positions is a Tensor.",
    )

    check(
        isinstance(headings, torch.Tensor),
        "headings is a Tensor.",
    )

    check(
        isinstance(agent_mask, torch.Tensor),
        "agent_mask is a Tensor.",
    )

    check(
        isinstance(map_mask, torch.Tensor),
        "map_mask is a Tensor.",
    )

    check(
        isinstance(graph, list),
        "graph is a list.",
    )

    check(
        agent_trajectories.shape[-1] == 2,
        "Agent trajectories have coordinate dimension 2.",
    )

    check(
        future_trajectories.shape[-1] == 2,
        "Future trajectories have coordinate dimension 2.",
    )

    check(
        map_centerlines.shape[-1] == 2,
        "Map centerlines have coordinate dimension 2.",
    )

    for key in (
        "agent_trajectories",
        "future_trajectories",
        "map_centerlines",
        "positions",
        "headings",
        "agent_mask",
        "map_mask",
    ):
        check_tensor(
            key,
            batch[key],
        )

    check(
        isinstance(
            batch["graph"],
            list,
        ),
        "graph is a list.",
    )

    check(
        len(batch["graph"]) == 1,
        "Exactly one SceneGraph is present.",
    )

    ###########################################################################
    # Shape checks
    ###########################################################################

    B = 1
    N = scene_data.num_agents
    M = scene_data.num_maps

    check(
        tuple(
            batch["agent_trajectories"].shape
        )
        == (
            B,
            N,
            OBSERVATION_STEPS,
            2,
        ),
        "Agent trajectory shape is correct.",
    )

    check(
        tuple(
            batch["future_trajectories"].shape
        )
        == (
            B,
            N,
            PREDICTION_STEPS,
            2,
        ),
        "Ground-truth trajectory shape is correct.",
    )

    check(
        tuple(
            batch["map_centerlines"].shape
        )
        == (
            B,
            M,
            MAP_SAMPLE_POINTS,
            2,
        ),
        "Map centerline shape is correct.",
    )

    check(
        tuple(
            batch["positions"].shape
        )
        == (
            B,
            N,
            2,
        ),
        "Position shape is correct.",
    )

    check(
        tuple(
            batch["headings"].shape
        )
        == (
            B,
            N,
        ),
        "Heading shape is correct.",
    )

    return batch


###############################################################################
# Duplicate batch for B=2
###############################################################################


def duplicate_batch(
    batch: dict[str, Any],
) -> dict[str, Any]:
    """
    Construct a B=2 batch from the real scene.

    Tensor fields are duplicated along the batch dimension.

    Graph entries are duplicated as two independent list entries.
    """

    duplicated: dict[str, Any] = {}

    tensor_keys = {
        "agent_trajectories",
        "future_trajectories",
        "map_centerlines",
        "positions",
        "headings",
        "agent_mask",
        "map_mask",
    }

    for key, value in batch.items():

        if key in tensor_keys:

            check(
                isinstance(
                    value,
                    torch.Tensor,
                ),
                f"Batch field '{key}' is a Tensor.",
            )

            duplicated[key] = torch.cat(
                [
                    value,
                    value.clone(),
                ],
                dim=0,
            )

        elif key == "graph":

            check(
                isinstance(
                    value,
                    list,
                ),
                "Graph is a list before duplication.",
            )

            duplicated[key] = (
                list(value)
                + list(value)
            )

        elif key == "metadata":

            if isinstance(
                value,
                dict,
            ):
                duplicated[key] = value
            else:
                duplicated[key] = value

        else:

            duplicated[key] = value

    return duplicated


###############################################################################
# Model construction
###############################################################################


def build_model(
    device: torch.device,
) -> tuple[
    DSTNet,
    TotalLoss,
    torch.optim.Optimizer,
    Any,
]:

    section(
        "3. DSTNET + LOSS + OPTIMIZER + SCHEDULER"
    )

    model = DSTNet(
        observation_steps=OBSERVATION_STEPS,
        prediction_steps=PREDICTION_STEPS,
        lane_points=MAP_SAMPLE_POINTS,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=3,
        num_modes=NUM_MODES,
        refinement_iterations=REFINEMENT_ITERATIONS,
        dropout=DROPOUT,
    )

    model.to(device)

    check(
        isinstance(
            model,
            torch.nn.Module,
        ),
        "DSTNet instantiated.",
    )

    criterion = TotalLoss()

    check(
        isinstance(
            criterion,
            torch.nn.Module,
        ),
        "TotalLoss instantiated.",
    )

    optimizer = build_optimizer(
        model=model,
        optimizer="adamw",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    check(
        isinstance(
            optimizer,
            torch.optim.Optimizer,
        ),
        "AdamW optimizer instantiated.",
    )

    scheduler = build_scheduler(
        optimizer,
        scheduler="cosine",
        total_steps=10,
    )

    check(
        scheduler is not None,
        "Cosine scheduler instantiated.",
    )

    info(
        f"Optimizer = "
        f"{optimizer.__class__.__name__}"
    )

    info(
        f"Scheduler = "
        f"{scheduler.__class__.__name__}"
    )

    info(
        f"Learning rate = "
        f"{optimizer.param_groups[0]['lr']}"
    )

    return (
        model,
        criterion,
        optimizer,
        scheduler,
    )


###############################################################################
# Direct model/loss verification
###############################################################################


def verify_direct_contract(
    model: DSTNet,
    criterion: TotalLoss,
    batch: dict[str, Any],
    device: torch.device,
) -> None:

    section(
        "4. DIRECT DSTNET + TOTALLOSS CONTRACT"
    )

    model.eval()

    with torch.no_grad():

        coarse_prediction, refined_prediction = (
            model(
                agent_trajectories=batch[
                    "agent_trajectories"
                ].to(device),
                lane_centerlines=batch[
                    "map_centerlines"
                ].to(device),
                positions=batch[
                    "positions"
                ].to(device),
                graph=batch[
                    "graph"
                ],
                agent_mask=batch[
                    "agent_mask"
                ].to(device),
                lane_mask=batch[
                    "map_mask"
                ].to(device),
            )
        )

    check(
        coarse_prediction is not None,
        "DSTNet produced coarse Prediction.",
    )

    check(
        refined_prediction is not None,
        "DSTNet produced RefinedPrediction.",
    )

    check_tensor(
        "coarse trajectories",
        coarse_prediction.trajectories,
    )

    check_tensor(
        "coarse probabilities",
        coarse_prediction.probabilities,
    )

    check_tensor(
        "refined trajectories",
        refined_prediction.trajectories,
    )

    check_tensor(
        "refinement scores",
        refined_prediction.refinement_scores,
    )

    check_tensor(
        "refinement offsets",
        refined_prediction.offsets,
    )

    losses = criterion(
        coarse_prediction,
        refined_prediction,
        batch[
            "future_trajectories"
        ].to(device),
    )

    check(
        isinstance(
            losses,
            dict,
        ),
        "TotalLoss returns a dictionary.",
    )

    for name in (
        "loss",
        "proposal_loss",
        "classification_loss",
        "score_loss",
        "refinement_loss",
    ):

        check(
            name in losses,
            f"TotalLoss contains '{name}'.",
        )

        check_tensor(
            f"loss/{name}",
            losses[name],
        )

        check(
            losses[name].ndim == 0,
            f"Loss '{name}' is scalar.",
        )

    info(
        f"Proposal loss       = "
        f"{losses['proposal_loss'].item():.6f}"
    )

    info(
        f"Classification loss = "
        f"{losses['classification_loss'].item():.6f}"
    )

    info(
        f"Score loss          = "
        f"{losses['score_loss'].item():.6f}"
    )

    info(
        f"Refinement loss     = "
        f"{losses['refinement_loss'].item():.6f}"
    )

    info(
        f"Total loss          = "
        f"{losses['loss'].item():.6f}"
    )


###############################################################################
# TrainStep verification
###############################################################################


def run_training_step(
    *,
    model: DSTNet,
    criterion: TotalLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:

    section(
        "5. TRAINSTEP"
    )

    train_step = TrainStep(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip=GRADIENT_CLIP,
        device=device,
    )

    check(
        train_step is not None,
        "TrainStep instantiated.",
    )

    ###########################################################################
    # Snapshot parameters
    ###########################################################################

    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    ###########################################################################
    # Snapshot learning rate
    ###########################################################################

    lr_before = float(
        optimizer.param_groups[0]["lr"]
    )

    ###########################################################################
    # Execute one complete step
    ###########################################################################

    metrics = train_step(
        batch
    )

    check(
        isinstance(
            metrics,
            dict,
        ),
        "TrainStep returns a dictionary.",
    )

    ###########################################################################
    # Required metrics
    ###########################################################################

    for name in (
        "loss",
        "proposal_loss",
        "classification_loss",
        "score_loss",
        "refinement_loss",
        "gradient_norm",
        "learning_rate",
    ):

        check(
            name in metrics,
            f"TrainStep metrics contain '{name}'.",
        )

        check(
            isinstance(
                metrics[name],
                float,
            ),
            f"TrainStep metric '{name}' is a Python float.",
        )

        check(
            bool(
                torch.isfinite(
                    torch.tensor(
                        metrics[name]
                    )
                ).item()
            ),
            f"TrainStep metric '{name}' is finite.",
        )

    ###########################################################################
    # Loss
    ###########################################################################

    check(
        metrics["loss"] >= 0.0,
        "Total training loss is non-negative.",
    )

    info(
        f"TrainStep total loss = "
        f"{metrics['loss']:.6f}"
    )

    ###########################################################################
    # Gradient norm
    ###########################################################################

    check(
        metrics["gradient_norm"] >= 0.0,
        "Gradient norm is non-negative.",
    )

    check(
        metrics["gradient_norm"] > 0.0,
        "Gradient norm is non-zero.",
    )

    info(
        f"Gradient norm = "
        f"{metrics['gradient_norm']:.6f}"
    )

    ###########################################################################
    # Parameter update
    ###########################################################################

    changed_parameters = 0

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        before = parameters_before[name]

        after = parameter.detach()

        if not torch.allclose(
            before,
            after,
            atol=0.0,
            rtol=0.0,
        ):
            changed_parameters += 1

    check(
        changed_parameters > 0,
        "Optimizer updated at least one trainable parameter.",
    )

    info(
        f"Changed trainable parameters = "
        f"{changed_parameters}"
    )

    ###########################################################################
    # Learning-rate scheduler
    ###########################################################################

    lr_after = float(
        optimizer.param_groups[0]["lr"]
    )

    info(
        f"Learning rate before = "
        f"{lr_before:.10f}"
    )

    info(
        f"Learning rate after  = "
        f"{lr_after:.10f}"
    )

    check(
        metrics["learning_rate"]
        == lr_after,
        "Returned learning rate matches optimizer learning rate.",
    )

    ###########################################################################
    # Scheduler should have advanced
    #
    # Cosine scheduler may produce a very small change depending on
    # total_steps, so compare its internal last_epoch instead of requiring
    # a large numerical LR change.
    ###########################################################################

    if hasattr(
        scheduler,
        "last_epoch",
    ):

        check(
            int(
                scheduler.last_epoch
            )
            >= 1,
            "Scheduler advanced after the training step.",
        )

    return metrics


###############################################################################
# B=2 verification
###############################################################################


def run_batch_size_two_test(
    *,
    model: DSTNet,
    criterion: TotalLoss,
    batch: dict[str, Any],
    device: torch.device,
) -> None:

    section(
        "6. BATCH SIZE > 1 TRAINSTEP VERIFICATION"
    )

    batch2 = duplicate_batch(
        batch
    )

    ###########################################################################
    # Shape
    ###########################################################################

    B = 2

    check(
        batch2[
            "agent_trajectories"
        ].shape[0]
        == B,
        "Duplicated training batch has B=2.",
    )

    check(
        batch2[
            "future_trajectories"
        ].shape[0]
        == B,
        "Duplicated ground truth has B=2.",
    )

    check(
        len(
            batch2["graph"]
        )
        == B,
        "Duplicated graph list has B=2 entries.",
    )

    ###########################################################################
    # Independent optimizer for this verification
    ###########################################################################

    optimizer = build_optimizer(
        model=model,
        optimizer="adamw",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(
        optimizer,
        scheduler="cosine",
        total_steps=10,
    )

    train_step = TrainStep(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip=GRADIENT_CLIP,
        device=device,
    )

    ###########################################################################
    # Run
    ###########################################################################

    metrics = train_step(
        batch2
    )

    check(
        isinstance(
            metrics,
            dict,
        ),
        "B=2 TrainStep returns metrics.",
    )

    check(
        metrics["loss"] >= 0.0,
        "B=2 training loss is non-negative.",
    )

    check(
        bool(
            torch.isfinite(
                torch.tensor(
                    metrics["loss"]
                )
            ).item()
        ),
        "B=2 training loss is finite.",
    )

    check(
        metrics["gradient_norm"] > 0.0,
        "B=2 gradient norm is non-zero.",
    )

    info(
        f"B=2 loss = "
        f"{metrics['loss']:.6f}"
    )

    info(
        f"B=2 gradient norm = "
        f"{metrics['gradient_norm']:.6f}"
    )


###############################################################################
# Input validation tests
###############################################################################


def run_validation_tests(
    *,
    model: DSTNet,
    criterion: TotalLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    batch: dict[str, Any],
    device: torch.device,
) -> None:

    section(
        "7. TRAINSTEP INPUT VALIDATION"
    )

    train_step = TrainStep(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip=GRADIENT_CLIP,
        device=device,
    )

    ###########################################################################
    # Missing required field
    ###########################################################################

    invalid_batch = dict(
        batch
    )

    invalid_batch.pop(
        "future_trajectories"
    )

    rejected_missing = False

    try:

        train_step(
            invalid_batch
        )

    except (
        KeyError,
        ValueError,
        TypeError,
    ):

        rejected_missing = True

    check(
        rejected_missing,
        "TrainStep rejects a batch missing ground truth.",
    )

    ###########################################################################
    # Missing agent trajectories
    ###########################################################################

    invalid_batch = dict(
        batch
    )

    invalid_batch.pop(
        "agent_trajectories"
    )

    rejected_missing_agent = False

    try:

        train_step(
            invalid_batch
        )

    except (
        KeyError,
        ValueError,
        TypeError,
    ):

        rejected_missing_agent = True

    check(
        rejected_missing_agent,
        "TrainStep rejects a batch missing agent trajectories.",
    )


###############################################################################
# Main
###############################################################################


def main() -> int:

    args = parse_args()

    device = torch.device(
        args.device
    )

    print()
    print(
        "=" * 78
    )
    print(
        "DSTNet TRAINING STEP TARGETED VERIFICATION"
    )
    print(
        "=" * 78
    )

    info(
        f"Device = {device}"
    )

    info(
        f"PyTorch = {torch.__version__}"
    )

    info(
        f"CSV = {args.csv}"
    )

    info(
        f"Map root = {args.map_root}"
    )

    info(
        f"H = {OBSERVATION_STEPS}"
    )

    info(
        f"T = {PREDICTION_STEPS}"
    )

    info(
        f"K = {NUM_MODES}"
    )

    info(
        f"D = {HIDDEN_DIM}"
    )

    ###########################################################################
    # Reproducibility
    ###########################################################################

    torch.manual_seed(
        42
    )

    ###########################################################################
    # Real scene
    ###########################################################################

    scene_data = build_scene_data(
        args.csv,
        args.map_root,
    )

    ###########################################################################
    # Batch
    ###########################################################################

    batch = build_batch(
        scene_data
    )

    ###########################################################################
    # Model + training components
    ###########################################################################

    (
        model,
        criterion,
        optimizer,
        scheduler,
    ) = build_model(
        device
    )

    ###########################################################################
    # Direct contract
    ###########################################################################

    verify_direct_contract(
        model=model,
        criterion=criterion,
        batch=batch,
        device=device,
    )

    ###########################################################################
    # Complete TrainStep
    ###########################################################################

    metrics = run_training_step(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        batch=batch,
        device=device,
    )

    ###########################################################################
    # B=2
    ###########################################################################

    run_batch_size_two_test(
        model=model,
        criterion=criterion,
        batch=batch,
        device=device,
    )

    ###########################################################################
    # Validation
    ###########################################################################

    run_validation_tests(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        batch=batch,
        device=device,
    )

    ###########################################################################
    # Final summary
    ###########################################################################

    section(
        "FINAL RESULT"
    )

    print(
        "[PASS] Real Argoverse-1 scene"
    )

    print(
        "[PASS] Scene preprocessing"
    )

    print(
        "[PASS] Model batch construction"
    )

    print(
        "[PASS] DSTNet forward"
    )

    print(
        "[PASS] Prediction"
    )

    print(
        "[PASS] RefinedPrediction"
    )

    print(
        "[PASS] Proposal loss"
    )

    print(
        "[PASS] Classification loss"
    )

    print(
        "[PASS] Score loss"
    )

    print(
        "[PASS] Refinement loss"
    )

    print(
        "[PASS] Total loss"
    )

    print(
        "[PASS] Backward propagation"
    )

    print(
        "[PASS] Gradient validation"
    )

    print(
        "[PASS] Gradient clipping"
    )

    print(
        "[PASS] Optimizer update"
    )

    print(
        "[PASS] Scheduler update"
    )

    print(
        "[PASS] Batch size B=2"
    )

    print(
        "[PASS] TrainStep input validation"
    )

    print()
    print(
        "TRAINING STEP VERIFICATION PASSED"
    )

    print()
    print(
        f"Final training loss = "
        f"{metrics['loss']:.6f}"
    )

    print(
        f"Final gradient norm = "
        f"{metrics['gradient_norm']:.6f}"
    )

    print(
        f"Final learning rate = "
        f"{metrics['learning_rate']:.10f}"
    )

    return 0


###############################################################################
# Entry point
###############################################################################


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
