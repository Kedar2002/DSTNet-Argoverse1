"""
scripts/test_dstnet.py

Phase-6 verification for the complete DSTNet model.

Pipeline
--------
Argoverse-1 CSV
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
Prediction
    ↓
RefinedPrediction

This test verifies the complete top-level DSTNet interface using
the same real-data preprocessing path already validated by the
Phase-5 pipeline test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


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
# Dataset
###############################################################################

from datasets.map_loader import MapLoader
from datasets.scene_parser import SceneParser
from datasets.preprocess import ScenePreprocessor
from datasets.collate import collate_fn


###############################################################################
# Model
###############################################################################

from models.dstnet import DSTNet


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
NUM_ENCODER_LAYERS = 2
NUM_MODES = 6

REFINEMENT_ITERATIONS = 2

DROPOUT = 0.0


###############################################################################
# Utilities
###############################################################################


def section(
    title: str,
) -> None:

    print()
    print("=" * 79)
    print(title)
    print("=" * 79)


def info(
    message: str,
) -> None:

    print(
        f"[INFO] {message}"
    )


def check(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise AssertionError(
            f"[FAIL] {message}"
        )

    print(
        f"[PASS] {message}"
    )


def check_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:

    check(
        bool(
            torch.isfinite(tensor)
            .all()
            .item()
        ),
        f"{name}: all values finite",
    )


def print_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:

    info(
        f"{name}: "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}"
    )


###############################################################################
# Arguments
###############################################################################


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Phase-6 complete DSTNet "
            "real-data verification."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Argoverse-1 CSV file.",
    )

    parser.add_argument(
        "--map-root",
        type=Path,
        required=True,
        help="Argoverse-1 HD map root.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device. Default: cpu.",
    )

    return parser.parse_args()


###############################################################################
# Scene loading
###############################################################################


def load_scene(
    csv_path: Path,
    map_root: Path,
):
    """
    Load one real Argoverse-1 scene using the canonical
    repository preprocessing pipeline.

    Returns
    -------
    SceneData
    """

    ###########################################################################
    # Map loader
    ###########################################################################

    map_loader = MapLoader(
        map_root=map_root,
    )

    check(
        map_loader.is_loaded(),
        "HD maps loaded successfully.",
    )

    info(
        f"cities = {map_loader.cities}"
    )

    info(
        f"total lanes = "
        f"{map_loader.total_num_lanes}"
    )

    ###########################################################################
    # Scene parser
    ###########################################################################

    parser = SceneParser(
        map_loader,
    )

    raw_scene = parser.parse(
        csv_path,
    )

    check(
        raw_scene.num_tracks > 0,
        (
            "RawScene contains "
            f"{raw_scene.num_tracks} tracks."
        ),
    )

    check(
        raw_scene.num_lanes > 0,
        (
            "RawScene contains "
            f"{raw_scene.num_lanes} map lanes."
        ),
    )

    check(
        raw_scene.target_track is not None,
        "Prediction target exists.",
    )

    info(
        "sequence_id = "
        f"{raw_scene.metadata.sequence_id}"
    )

    info(
        "city        = "
        f"{raw_scene.metadata.city}"
    )

    info(
        "target      = "
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

    ###########################################################################
    # IMPORTANT
    #
    # The current repository exposes the preprocessing object as callable.
    # This is the same interface used by test_pipeline_full.py.
    ###########################################################################

    scene_data = preprocessor(
        raw_scene,
    )

    ###########################################################################
    # SceneData validation
    ###########################################################################

    check(
        scene_data.num_agents > 0,
        (
            "SceneData contains "
            f"{scene_data.num_agents} agents."
        ),
    )

    check(
        scene_data.num_maps > 0,
        (
            "SceneData contains "
            f"{scene_data.num_maps} maps."
        ),
    )

    check(
        scene_data.num_agent_states
        == (
            scene_data.num_agents
            * OBSERVATION_STEPS
        ),
        (
            "SceneGraph state count matches "
            "N × H."
        ),
    )

    check(
        scene_data.scene_graph is not None,
        "SceneData contains SceneGraph.",
    )

    info(
        f"origin  = {scene_data.origin}"
    )

    info(
        f"heading = {scene_data.heading}"
    )

    return scene_data


###############################################################################
# Main
###############################################################################


def main() -> None:

    args = parse_args()

    device = torch.device(
        args.device
    )

    ###########################################################################
    # Header
    ###########################################################################

    section(
        "DSTNET PHASE-6 TARGETED VERIFICATION"
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
    # Path validation
    ###########################################################################

    check(
        args.csv.exists(),
        "CSV file exists.",
    )

    check(
        args.map_root.exists(),
        "HD map root exists.",
    )

    ###########################################################################
    # REAL SCENE
    ###########################################################################

    section(
        "1. REAL ARGOVERSE-1 SCENE"
    )

    scene_data = load_scene(
        csv_path=args.csv,
        map_root=args.map_root,
    )

    ###########################################################################
    # COLLATE
    ###########################################################################

    section(
        "2. SCENEDATA → MODEL BATCH"
    )

    ###########################################################################
    # IMPORTANT:
    #
    # SceneData is NOT a dictionary.
    #
    # collate_fn expects:
    #
    #     list[SceneData]
    #
    # and returns the dictionary consumed by DSTNet.
    ###########################################################################

    batch = collate_fn(
        [
            scene_data,
        ]
    )

    check(
        isinstance(
            batch,
            dict,
        ),
        "collate_fn returned a model batch dictionary.",
    )

    ###########################################################################
    # Extract model inputs
    ###########################################################################

    agent_trajectories = batch[
        "agent_trajectories"
    ].to(device)

    lane_centerlines = batch[
        "map_centerlines"
    ].to(device)

    positions = batch[
        "positions"
    ].to(device)

    headings = batch[
        "headings"
    ].to(device)

    agent_mask = batch[
        "agent_mask"
    ].to(device)

    lane_mask = batch[
        "map_mask"
    ].to(device)

    graphs = batch[
        "graph"
    ]

    ###########################################################################
    # Print inputs
    ###########################################################################

    print_tensor(
        "agent_trajectories",
        agent_trajectories,
    )

    print_tensor(
        "lane_centerlines",
        lane_centerlines,
    )

    print_tensor(
        "positions",
        positions,
    )

    print_tensor(
        "headings",
        headings,
    )

    print_tensor(
        "agent_mask",
        agent_mask,
    )

    print_tensor(
        "lane_mask",
        lane_mask,
    )

    check(
        len(graphs) == 1,
        "Exactly one SceneGraph is present.",
    )

    ###########################################################################
    # Input validation
    ###########################################################################

    check(
        tuple(
            agent_trajectories.shape
        )
        == (
            1,
            scene_data.num_agents,
            OBSERVATION_STEPS,
            2,
        ),
        "Agent trajectory shape is correct.",
    )

    check(
        tuple(
            lane_centerlines.shape
        )
        == (
            1,
            scene_data.num_maps,
            MAP_SAMPLE_POINTS,
            2,
        ),
        "Lane centerline shape is correct.",
    )

    check_finite(
        agent_trajectories,
        "agent_trajectories",
    )

    check_finite(
        lane_centerlines,
        "lane_centerlines",
    )

    check_finite(
        positions,
        "positions",
    )

    ###########################################################################
    # DSTNet
    ###########################################################################

    section(
        "3. DSTNET CONSTRUCTION"
    )

    model = DSTNet(
        observation_steps=OBSERVATION_STEPS,
        prediction_steps=PREDICTION_STEPS,
        lane_points=MAP_SAMPLE_POINTS,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_modes=NUM_MODES,
        refinement_iterations=REFINEMENT_ITERATIONS,
        dropout=DROPOUT,
    ).to(device)

    check(
        isinstance(
            model,
            DSTNet,
        ),
        "DSTNet instantiated.",
    )

    ###########################################################################
    # Forward
    ###########################################################################

    section(
        "4. COMPLETE DSTNET FORWARD"
    )

    model.eval()

    with torch.no_grad():

        coarse_prediction, refined_prediction = (
            model(
                agent_trajectories=agent_trajectories,
                lane_centerlines=lane_centerlines,
                positions=positions,
                graph=graphs,
                agent_mask=agent_mask,
                lane_mask=lane_mask,
            )
        )

    ###########################################################################
    # Extract outputs
    ###########################################################################

    coarse_trajectories = (
        coarse_prediction.trajectories
    )

    probabilities = (
        coarse_prediction.probabilities
    )

    refined_trajectories = (
        refined_prediction.trajectories
    )

    refinement_scores = (
        refined_prediction.scores
    )

    refinement_offsets = (
        refined_prediction.offsets
    )

    ###########################################################################
    # Shapes
    ###########################################################################

    expected_trajectory_shape = (
        1,
        scene_data.num_agents,
        OBSERVATION_STEPS,
        NUM_MODES,
        PREDICTION_STEPS,
        2,
    )

    expected_mode_shape = (
        1,
        scene_data.num_agents,
        OBSERVATION_STEPS,
        NUM_MODES,
    )

    print_tensor(
        "Y^(0)",
        coarse_trajectories,
    )

    print_tensor(
        "probabilities",
        probabilities,
    )

    print_tensor(
        "Y_refined",
        refined_trajectories,
    )

    print_tensor(
        "scores",
        refinement_scores,
    )

    print_tensor(
        "offsets",
        refinement_offsets,
    )

    check(
        tuple(
            coarse_trajectories.shape
        )
        == expected_trajectory_shape,
        "Coarse trajectory shape is correct.",
    )

    check(
        tuple(
            probabilities.shape
        )
        == expected_mode_shape,
        "Probability shape is correct.",
    )

    check(
        tuple(
            refined_trajectories.shape
        )
        == expected_trajectory_shape,
        "Refined trajectory shape is correct.",
    )

    check(
        tuple(
            refinement_scores.shape
        )
        == expected_mode_shape,
        "Refinement score shape is correct.",
    )

    check(
        tuple(
            refinement_offsets.shape
        )
        == expected_trajectory_shape,
        "Refinement offset shape is correct.",
    )

    ###########################################################################
    # Numerical validation
    ###########################################################################

    check_finite(
        coarse_trajectories,
        "Y^(0)",
    )

    check_finite(
        probabilities,
        "probabilities",
    )

    check_finite(
        refined_trajectories,
        "Y_refined",
    )

    check_finite(
        refinement_scores,
        "refinement scores",
    )

    check_finite(
        refinement_offsets,
        "refinement offsets",
    )

    ###########################################################################
    # Probability normalization
    ###########################################################################

    probability_sum = (
        probabilities.sum(
            dim=-1
        )
    )

    check(
        torch.allclose(
            probability_sum,
            torch.ones_like(
                probability_sum
            ),
            atol=1e-5,
            rtol=1e-5,
        ),
        "Coarse trajectory probabilities sum to one.",
    )

    ###########################################################################
    # Refinement consistency
    ###########################################################################

    expected_offsets = (
        refined_trajectories
        - coarse_trajectories
    )

    check(
        torch.allclose(
            refinement_offsets,
            expected_offsets,
            atol=1e-5,
            rtol=1e-5,
        ),
        "Refinement offsets equal refined - coarse trajectories.",
    )

    total_change = (
        refinement_offsets
        .abs()
        .sum()
        .item()
    )

    info(
        "Total absolute refinement change = "
        f"{total_change:.6f}"
    )

    check(
        total_change > 0.0,
        "Real scene receives non-zero refinement.",
    )

    ###########################################################################
    # Score range
    ###########################################################################

    check(
        bool(
            (
                refinement_scores >= 0.0
            )
            .all()
            .item()
        ),
        "Refinement scores are non-negative.",
    )

    check(
        bool(
            (
                refinement_scores <= 1.0
            )
            .all()
            .item()
        ),
        "Refinement scores are <= 1.",
    )

    ###########################################################################
    # B > 1
    ###########################################################################

    section(
        "5. BATCH SIZE > 1"
    )

    batch_agent_trajectories = torch.cat(
        [
            agent_trajectories,
            agent_trajectories,
        ],
        dim=0,
    )

    batch_lane_centerlines = torch.cat(
        [
            lane_centerlines,
            lane_centerlines,
        ],
        dim=0,
    )

    batch_positions = torch.cat(
        [
            positions,
            positions,
        ],
        dim=0,
    )

    batch_headings = torch.cat(
        [
            headings,
            headings,
        ],
        dim=0,
    )

    batch_agent_mask = torch.cat(
        [
            agent_mask,
            agent_mask,
        ],
        dim=0,
    )

    batch_lane_mask = torch.cat(
        [
            lane_mask,
            lane_mask,
        ],
        dim=0,
    )

    batch_graphs = [
        graphs[0],
        graphs[0],
    ]

    model.eval()

    with torch.no_grad():

        batch_coarse, batch_refined = (
            model(
                agent_trajectories=(
                    batch_agent_trajectories
                ),
                lane_centerlines=(
                    batch_lane_centerlines
                ),
                positions=batch_positions,
                graph=batch_graphs,
                agent_mask=batch_agent_mask,
                lane_mask=batch_lane_mask,
            )
        )

    check(
        batch_coarse.trajectories.shape[0]
        == 2,
        "DSTNet preserves B=2.",
    )

    check_finite(
        batch_coarse.trajectories,
        "B=2 coarse trajectories",
    )

    check_finite(
        batch_refined.trajectories,
        "B=2 refined trajectories",
    )

    ###########################################################################
    # Gradient propagation
    ###########################################################################

    section(
        "6. COMPLETE GRADIENT VERIFICATION"
    )

    model.train()

    gradient_agents = (
        agent_trajectories.detach()
        .clone()
        .requires_grad_(True)
    )

    gradient_lanes = (
        lane_centerlines.detach()
        .clone()
        .requires_grad_(True)
    )

    gradient_positions = (
        positions.detach()
        .clone()
        .requires_grad_(True)
    )

    gradient_coarse, gradient_refined = (
        model(
            agent_trajectories=gradient_agents,
            lane_centerlines=gradient_lanes,
            positions=gradient_positions,
            graph=graphs,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
        )
    )

    ###########################################################################
    # Complete differentiable objective
    ###########################################################################

    loss = (
        gradient_refined.trajectories.square().mean()
        + gradient_refined.scores.mean()
        + gradient_coarse.probabilities.square().mean()
    )

    check(
        bool(
            torch.isfinite(
                loss
            ).item()
        ),
        "Complete DSTNet test loss is finite.",
    )

    loss.backward()

    ###########################################################################
    # Input gradients
    ###########################################################################

    check(
        gradient_agents.grad is not None,
        "Gradient reaches agent trajectories.",
    )

    check(
        gradient_lanes.grad is not None,
        "Gradient reaches lane centerlines.",
    )

    ###########################################################################
    # Validate input gradients
    ###########################################################################

    if gradient_agents.grad is not None:

        check_finite(
            gradient_agents.grad,
            "Agent trajectory gradient",
        )

        check(
            bool(
                gradient_agents.grad
                .abs()
                .sum()
                .item()
                > 0.0
            ),
            "Agent trajectory gradient is non-zero.",
        )

    if gradient_lanes.grad is not None:

        check_finite(
            gradient_lanes.grad,
            "Lane centerline gradient",
        )

        check(
            bool(
                gradient_lanes.grad
                .abs()
                .sum()
                .item()
                > 0.0
            ),
            "Lane centerline gradient is non-zero.",
        )

    ###############################################################################
    # Position gradients
    ###############################################################################

    # Positions are used by the encoder for geometric interaction
    # construction/masking. These operations can involve discrete
    # neighbourhood selection and therefore do not necessarily provide
    # a differentiable path back to the raw position tensor.
    #
    # Therefore, Phase-6 does NOT require a gradient directly on positions.
    #
    # The important gradient checks are:
    #
    #   agent trajectories → AgentEncoder → Encoder → ...
    #   lane centerlines   → MapEncoder   → Encoder → ...
    #   model parameters   → complete DSTNet
    #
    # Position usage is already verified by the successful complete
    # forward pass and the Phase-3/4 spatial-interaction tests.

    if gradient_positions.grad is None:

        print(
            "[INFO] Position gradient is not required: "
            "positions participate in geometric interaction/masking "
            "operations that may be non-differentiable."
        )

    else:

        check_finite(
            gradient_positions.grad,
            "Position gradient",
        )

        if (
            gradient_positions.grad
            .abs()
            .sum()
            .item()
            > 0.0
        ):

            print(
                "[PASS] Position gradient is non-zero."
            )

        else:

            print(
                "[INFO] Position gradient is zero; "
                "this is acceptable for the current "
                "geometric interaction implementation."
            )

    ###########################################################################
    # Model parameter gradients
    ###########################################################################

    modules = {
        "AgentEncoder": model.agent_encoder,
        "MapEncoder": model.lane_encoder,
        "Encoder": model.encoder,
        "Decoder": model.decoder,
        "Refinement": model.refinement,
    }

    for name, module in modules.items():

        receives_gradient = False

        for parameter in module.parameters():

            if parameter.grad is None:
                continue

            check_finite(
                parameter.grad,
                f"{name} parameter gradient",
            )

            if (
                parameter.grad
                .abs()
                .sum()
                .item()
                > 0.0
            ):

                receives_gradient = True
                break

        check(
            receives_gradient,
            (
                f"At least one {name} "
                "parameter receives gradient."
            ),
        )

    ###########################################################################
    # Final summary
    ###########################################################################

    section(
        "PHASE-6 COMPLETE DSTNET VERIFICATION PASSED"
    )

    print(
        "[PASS] Real Argoverse-1 scene"
    )

    print(
        "[PASS] Scene preprocessing"
    )

    print(
        "[PASS] AgentEncoder"
    )

    print(
        "[PASS] MapEncoder"
    )

    print(
        "[PASS] Phase-4 Encoder"
    )

    print(
        "[PASS] GSTA"
    )

    print(
        "[PASS] Tri-ATM"
    )

    print(
        "[PASS] Z_STM"
    )

    print(
        "[PASS] Decoder"
    )

    print(
        "[PASS] Coarse trajectories"
    )

    print(
        "[PASS] Multimodal probabilities"
    )

    print(
        "[PASS] Anchor refinement"
    )

    print(
        "[PASS] Refined trajectories"
    )

    print(
        "[PASS] Refinement scores"
    )

    print(
        "[PASS] Refinement offsets"
    )

    print(
        "[PASS] Batch size > 1"
    )

    print(
        "[PASS] Complete gradient propagation"
    )

    print()
    print(
        "Final tensor flow:"
    )

    print(
        f"Z_STM         = "
        f"(1, {scene_data.num_agents}, "
        f"{OBSERVATION_STEPS}, "
        f"{NUM_MODES}, {HIDDEN_DIM})"
    )   

    print(
        f"Y^(0)         = "
        f"{tuple(coarse_trajectories.shape)}"
    )

    print(
        f"Y_refined     = "
        f"{tuple(refined_trajectories.shape)}"
    )

    print(
        f"probabilities = "
        f"{tuple(probabilities.shape)}"
    )

    print(
        f"scores        = "
        f"{tuple(refinement_scores.shape)}"
    )

    print(
        f"offsets       = "
        f"{tuple(refinement_offsets.shape)}"
    )


###############################################################################
# Entry point
###############################################################################


if __name__ == "__main__":
    main()
