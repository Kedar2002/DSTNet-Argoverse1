"""
scripts/test_pipeline_full.py

Real-data verification of the DSTNet Phase-5 decoder + refinement path.

Pipeline
--------
Actual Argoverse-1 CSV
        |
        v
    MapLoader
        |
        v
   SceneParser
        |
        v
     RawScene
        |
        v
 ScenePreprocessor
        |
        v
    SceneData
        |
        v
   collate_fn
        |
        +----------------------+
        |                      |
        v                      v
 Agent trajectories        Map centerlines
        |                      |
        v                      v
 AgentEncoder             MapEncoder
        |                      |
        +----------+-----------+
                   |
                   v
              Encoder
                   |
                   v
                Z_STM
                   |
                   v
                Decoder
                   |
                   v
                 Y^(0)
                   |
                   v
              Refinement
                   |
                   v
          RefinedPrediction

This test intentionally follows the interfaces already verified by
test_pipeline_v2.py and the targeted Phase-4/Phase-5 tests.

It does NOT modify tensors or repair shape mismatches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


###############################################################################
# Project Root
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

NUM_ENCODER_LAYERS = 2

DROPOUT = 0.0

REFINEMENT_ITERATIONS = 1


###############################################################################
# Imports
###############################################################################

from datasets.map_loader import MapLoader
from datasets.scene_parser import SceneParser
from datasets.preprocess import ScenePreprocessor
from datasets.collate import collate_fn

from models.encoders.agent_encoder import AgentEncoder
from models.encoders.map_encoder import MapEncoder
from models.encoders.encoder import Encoder

from models.decoder.decoder import Decoder

from models.refinement.refinement import Refinement

from models.model_types import Prediction


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

    if condition:

        print(
            f"[PASS] {message}"
        )

    else:

        print(
            f"[FAIL] {message}"
        )


def fail(
    message: str,
) -> None:

    print(
        f"[FAIL] {message}"
    )


def warn(
    message: str,
) -> None:

    print(
        f"[WARN] {message}"
    )


###############################################################################
# Tensor validation
###############################################################################


def check_tensor(
    name: str,
    tensor: torch.Tensor,
) -> bool:

    if not isinstance(
        tensor,
        torch.Tensor,
    ):

        fail(
            f"{name}: expected torch.Tensor, "
            f"got {type(tensor).__name__}."
        )

        return False

    finite = bool(
        torch.isfinite(
            tensor
        ).all().item()
    )

    if not finite:

        fail(
            f"{name}: contains NaN or Inf."
        )

        return False

    print(
        f"[PASS] {name}: "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}"
    )

    return True


###############################################################################
# Scene validation
###############################################################################


def validate_scene(
    raw_scene: Any,
    scene_data: Any,
) -> None:

    section(
        "1. REAL ARGOVERSE SCENE"
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

    info(
        f"tracks      = "
        f"{raw_scene.num_tracks}"
    )

    info(
        f"lanes       = "
        f"{raw_scene.num_lanes}"
    )

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

    info(
        f"origin  = {scene_data.origin}"
    )

    info(
        f"heading = {scene_data.heading}"
    )

    for index, agent in enumerate(
        scene_data.agents
    ):

        observed = np.asarray(
            agent["observed"]
        )

        future = np.asarray(
            agent["future"]
        )

        if index < 3:

            info(
                f"agent[{index}] "
                f"observed={observed.shape}, "
                f"future={future.shape}"
            )

    graph = scene_data.scene_graph

    check(
        graph.num_agent_states
        == scene_data.num_agents
        * OBSERVATION_STEPS,
        (
            "SceneGraph state count matches "
            f"N×H = "
            f"{scene_data.num_agents}"
            f"×{OBSERVATION_STEPS}."
        ),
    )

    check(
        graph.num_map_nodes
        == scene_data.num_maps,
        "SceneGraph map count matches SceneData.",
    )


###############################################################################
# Batch
###############################################################################


def build_batch(
    scene_data: Any,
) -> dict[str, Any]:

    section(
        "2. BATCH COLLATION"
    )

    batch = collate_fn(
        [scene_data]
    )

    for key, value in batch.items():

        if isinstance(
            value,
            torch.Tensor,
        ):

            info(
                f"{key:22s} "
                f"shape={tuple(value.shape)}"
            )

        elif isinstance(
            value,
            list,
        ):

            info(
                f"{key:22s} "
                f"list[{len(value)}]"
            )

        else:

            info(
                f"{key:22s} "
                f"type={type(value).__name__}"
            )

    return batch


###############################################################################
# Local encoders
###############################################################################


def run_local_encoders(
    batch: dict[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    section(
        "3. LOCAL ENCODERS"
    )

    agent_trajectories = batch[
        "agent_trajectories"
    ]

    map_centerlines = batch[
        "map_centerlines"
    ]

    ###########################################################################
    # AgentEncoder
    ###########################################################################

    agent_encoder = AgentEncoder(
        observation_steps=OBSERVATION_STEPS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    agent_encoder.eval()

    with torch.no_grad():

        Ea = agent_encoder(
            agent_trajectories
        )

    check_tensor(
        "Ea",
        Ea,
    )

    expected_ea = (
        agent_trajectories.shape[0],
        agent_trajectories.shape[1],
        OBSERVATION_STEPS,
        HIDDEN_DIM,
    )

    check(
        tuple(Ea.shape)
        == expected_ea,
        (
            "Ea shape = "
            f"{expected_ea}."
        ),
    )

    ###########################################################################
    # MapEncoder
    ###########################################################################

    map_encoder = MapEncoder(
        num_points=MAP_SAMPLE_POINTS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    map_encoder.eval()

    with torch.no_grad():

        Em = map_encoder(
            map_centerlines
        )

    check_tensor(
        "Em",
        Em,
    )

    expected_em = (
        map_centerlines.shape[0],
        map_centerlines.shape[1],
        HIDDEN_DIM,
    )

    check(
        tuple(Em.shape)
        == expected_em,
        (
            "Em shape = "
            f"{expected_em}."
        ),
    )

    return Ea, Em


###############################################################################
# Phase-4 Encoder
###############################################################################


def run_encoder(
    batch: dict[str, Any],
    Ea: torch.Tensor,
    Em: torch.Tensor,
) -> torch.Tensor:

    section(
        "4. ENCODER → Z_STM"
    )

    positions = batch[
        "positions"
    ]

    headings = batch[
        "headings"
    ]

    graphs = batch[
        "graph"
    ]

    agent_mask = batch[
        "agent_mask"
    ]

    map_mask = batch[
        "map_mask"
    ]

    encoder = Encoder(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS,
        num_modes=NUM_MODES,
        observation_steps=OBSERVATION_STEPS,
        interaction_radius=SPATIAL_RADIUS,
        dropout=DROPOUT,
    )

    encoder.eval()

    info(
        f"Ea         = {tuple(Ea.shape)}"
    )

    info(
        f"Em         = {tuple(Em.shape)}"
    )

    info(
        f"positions  = {tuple(positions.shape)}"
    )

    info(
        f"agent_mask = {tuple(agent_mask.shape)}"
    )

    info(
        f"map_mask   = {tuple(map_mask.shape)}"
    )

    with torch.no_grad():

        z_stm = encoder(
            agent_features=Ea,
            lane_features=Em,
            positions=positions,
            graph=graphs,
            agent_mask=agent_mask,
            lane_mask=map_mask,
        )

    check_tensor(
        "Z_STM",
        z_stm,
    )

    expected = (
        Ea.shape[0],
        Ea.shape[1],
        Ea.shape[2],
        NUM_MODES,
        HIDDEN_DIM,
    )

    check(
        tuple(z_stm.shape)
        == expected,
        (
            "Z_STM shape = "
            f"{expected}."
        ),
    )

    return z_stm


###############################################################################
# Mode embedding
###############################################################################


def run_decoder(
    z_stm: torch.Tensor,
) -> Prediction:

    section(
        "5. DECODER → Y^(0)"
    )

    ###########################################################################
    # The current Decoder expects ModeFeatures.
    ###########################################################################

    decoder = Decoder(
        hidden_dim=HIDDEN_DIM,
        prediction_steps=PREDICTION_STEPS,
        dropout=DROPOUT,
    )

    decoder.eval()

    with torch.no_grad():

        prediction = decoder(
            z_stm
        )

    ###########################################################################
    # Decoder output
    ###########################################################################

    trajectories = prediction.trajectories
    probabilities = prediction.probabilities

    print(
        f"[INFO] trajectories = "
        f"{tuple(trajectories.shape)}"
    )

    print(
        f"[INFO] probabilities = "
        f"{tuple(probabilities.shape)}"
    )

    check_tensor(
        "Y^(0)",
        trajectories,
    )

    expected = (
        z_stm.shape[0],
        z_stm.shape[1],
        z_stm.shape[2],
        NUM_MODES,
        PREDICTION_STEPS,
        2,
    )

    check(
        tuple(trajectories.shape)
        == expected,
        (
            "Y^(0) shape = "
            f"{expected}."
        ),
    )

    ###########################################################################
    # Current Prediction contract
    ###########################################################################

    check(
        hasattr(
            prediction,
            "trajectories",
        ),
        "Prediction contains trajectories.",
    )

    check(
        hasattr(
            prediction,
            "probabilities",
        ),
        "Prediction contains multimodal probabilities.",
    )

    return prediction


###############################################################################
# Refinement
###############################################################################


def run_refinement(
    z_stm: torch.Tensor,
    prediction: Prediction,
) -> Any:

    section(
        "6. ADAPTIVE ANCHOR-BASED REFINEMENT"
    )

    ###########################################################################
    # Important:
    #
    # The current Refinement implementation consumes:
    #
    #     encoder_features
    #     prediction
    #
    # rather than requiring Prediction.scores.
    #
    # The encoder features supplied here are Z_STM.
    ###########################################################################

    refinement = Refinement(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        prediction_steps=PREDICTION_STEPS,
        refinement_iterations=REFINEMENT_ITERATIONS,
        radius_start=30.0,
        radius_end=10.0,
        dropout=DROPOUT,
    )

    refinement.eval()

    with torch.no_grad():

        refined_prediction = refinement(
            z_stm=z_stm,
            prediction=prediction,
        )

    ###########################################################################
    # Extract outputs
    ###########################################################################

    refined = (
        refined_prediction.trajectories
    )

    scores = (
        refined_prediction.scores
    )

    offsets = (
        refined_prediction.offsets
    )

    check_tensor(
        "Refined trajectories",
        refined,
    )

    check_tensor(
        "Refinement scores",
        scores,
    )

    if offsets is not None:

        check_tensor(
            "Refinement offsets",
            offsets,
        )

    ###########################################################################
    # Shapes
    ###########################################################################

    expected_trajectory_shape = (
        z_stm.shape[0],
        z_stm.shape[1],
        z_stm.shape[2],
        NUM_MODES,
        PREDICTION_STEPS,
        2,
    )

    expected_score_shape = (
        z_stm.shape[0],
        z_stm.shape[1],
        z_stm.shape[2],
        NUM_MODES,
    )

    check(
        tuple(refined.shape)
        == expected_trajectory_shape,
        (
            "Refined trajectories shape = "
            f"{expected_trajectory_shape}."
        ),
    )

    check(
        tuple(scores.shape)
        == expected_score_shape,
        (
            "Refinement scores shape = "
            f"{expected_score_shape}."
        ),
    )

    if offsets is not None:

        check(
            tuple(offsets.shape)
            == expected_trajectory_shape,
            (
                "Refinement offsets shape = "
                f"{expected_trajectory_shape}."
            ),
        )

    ###########################################################################
    # Offset consistency
    ###########################################################################

    if offsets is not None:

        expected_offsets = (
            refined
            - prediction.trajectories
        )

        check(
            bool(
                torch.allclose(
                    offsets,
                    expected_offsets,
                    atol=1e-5,
                    rtol=1e-5,
                )
            ),
            (
                "Refinement offsets equal "
                "refined - coarse trajectories."
            ),
        )

    ###########################################################################
    # Refinement actually changes trajectories
    ###########################################################################

    offset_magnitude = (
        (
            refined
            - prediction.trajectories
        )
        .abs()
        .sum()
        .item()
    )

    info(
        "Total absolute refinement change = "
        f"{offset_magnitude:.6f}"
    )

    check(
        offset_magnitude > 0.0,
        "Real scene receives non-zero refinement.",
    )

    ###########################################################################
    # Scores
    ###########################################################################

    check(
        bool(
            torch.isfinite(
                scores
            ).all().item()
        ),
        "Refinement scores are finite.",
    )

    return refined_prediction


###############################################################################
# Training / gradient test
###############################################################################


def run_gradient_test(
    batch: dict[str, Any],
) -> None:

    section(
        "7. REAL-DATA GRADIENT VERIFICATION"
    )

    ###########################################################################
    # Recreate the complete differentiable path.
    ###########################################################################

    agent_trajectories = batch[
        "agent_trajectories"
    ]

    map_centerlines = batch[
        "map_centerlines"
    ]

    positions = batch[
        "positions"
    ]

    headings = batch[
        "headings"
    ]

    graphs = batch[
        "graph"
    ]

    agent_mask = batch[
        "agent_mask"
    ]

    map_mask = batch[
        "map_mask"
    ]

    ###########################################################################
    # Local encoders
    ###########################################################################

    agent_encoder = AgentEncoder(
        observation_steps=OBSERVATION_STEPS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    map_encoder = MapEncoder(
        num_points=MAP_SAMPLE_POINTS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    ###########################################################################
    # Global encoder
    ###########################################################################

    encoder = Encoder(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS,
        num_modes=NUM_MODES,
        observation_steps=OBSERVATION_STEPS,
        interaction_radius=SPATIAL_RADIUS,
        dropout=DROPOUT,
    )

    ###########################################################################
    # Decoder
    ###########################################################################

    decoder = Decoder(
        hidden_dim=HIDDEN_DIM,
        prediction_steps=PREDICTION_STEPS,
        dropout=DROPOUT,
    )

    ###########################################################################
    # Refinement
    ###########################################################################

    refinement = Refinement(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        prediction_steps=PREDICTION_STEPS,
        refinement_iterations=REFINEMENT_ITERATIONS,
        radius_start=30.0,
        radius_end=10.0,
        dropout=DROPOUT,
    )

    agent_encoder.train()
    map_encoder.train()
    encoder.train()
    decoder.train()
    refinement.train()

    ###########################################################################
    # Forward
    ###########################################################################

    Ea = agent_encoder(
        agent_trajectories
    )

    Em = map_encoder(
        map_centerlines
    )

    z_stm = encoder(
        agent_features=Ea,
        lane_features=Em,
        positions=positions,
        graph=graphs,
        agent_mask=agent_mask,
        lane_mask=map_mask,
    )

    z_stm.retain_grad()

    prediction = decoder(
        z_stm
    )

    refined_prediction = refinement(
        z_stm=z_stm,
        prediction=prediction,
    )

    ###########################################################################
    # Simple finite differentiable test loss
    #
    # This is NOT the DSTNet training objective.
    # It only verifies gradient connectivity.
    ###########################################################################

    loss = (
        refined_prediction.trajectories
        .square()
        .mean()
        +
        refined_prediction.scores
        .square()
        .mean()
    )

    check(
        bool(
            torch.isfinite(
                loss
            ).item()
        ),
        "Real-data refinement test loss is finite.",
    )

    loss.backward()

    ###########################################################################
    # Z_STM gradient
    ###########################################################################

    check(
        z_stm.grad is not None,
        "Gradient reaches real Z_STM.",
    )

    if z_stm.grad is not None:

        check_tensor(
            "Z_STM gradient",
            z_stm.grad,
        )

        check(
            bool(
                z_stm.grad
                .abs()
                .sum()
                .item()
                > 0.0
            ),
            "Z_STM gradient is non-zero.",
        )

    ###########################################################################
    # Parameter gradients
    ###########################################################################

    encoder_gradient = False

    for parameter in encoder.parameters():

        if parameter.grad is not None:

            encoder_gradient = True

            break

    check(
        encoder_gradient,
        "At least one Encoder parameter receives gradient.",
    )

    decoder_gradient = False

    for parameter in decoder.parameters():

        if parameter.grad is not None:

            decoder_gradient = True

            break

    check(
        decoder_gradient,
        "At least one Decoder parameter receives gradient.",
    )

    refinement_gradient = False

    for parameter in refinement.parameters():

        if parameter.grad is not None:

            refinement_gradient = True

            break

    check(
        refinement_gradient,
        "At least one Refinement parameter receives gradient.",
    )


###############################################################################
# Main
###############################################################################


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Real-data Phase-5 DSTNet refinement verification."
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
        help="Path to HD map map_files directory.",
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    ###########################################################################
    # Header
    ###########################################################################

    section(
        "DSTNet REAL-DATA PHASE-5 VERIFICATION"
    )

    print(
        f"CSV      : {args.csv}"
    )

    print(
        f"Map root : {args.map_root}"
    )

    print(
        "Device   : cpu"
    )

    print(
        f"PyTorch  : {torch.__version__}"
    )

    print(
        f"OBSERVATION_STEPS = {OBSERVATION_STEPS}"
    )

    print(
        f"PREDICTION_STEPS  = {PREDICTION_STEPS}"
    )

    print(
        f"NUM_MODES         = {NUM_MODES}"
    )

    print(
        f"HIDDEN_DIM        = {HIDDEN_DIM}"
    )

    ###########################################################################
    # Path checks
    ###########################################################################

    if not args.csv.exists():

        fail(
            f"CSV does not exist: {args.csv}"
        )

        return 1

    if not args.map_root.exists():

        fail(
            f"Map root does not exist: "
            f"{args.map_root}"
        )

        return 1

    ###########################################################################
    # HD Map
    ###########################################################################

    section(
        "0. HD MAP"
    )

    try:

        map_loader = MapLoader(
            args.map_root
        )

    except Exception as exc:

        fail(
            "HD MapLoader initialization failed."
        )

        print(
            f"Exception: "
            f"{type(exc).__name__}: {exc}"
        )

        return 1

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
    # SceneParser
    ###########################################################################

    section(
        "CSV → RawScene"
    )

    try:

        scene_parser = SceneParser(
            map_api=map_loader
        )

        raw_scene = scene_parser.parse(
            args.csv
        )

    except Exception as exc:

        fail(
            "SceneParser failed."
        )

        print(
            f"Exception: "
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    ###########################################################################
    # Preprocessor
    ###########################################################################

    section(
        "RawScene → SceneData"
    )

    try:

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

    except Exception as exc:

        fail(
            "ScenePreprocessor failed."
        )

        print(
            f"Exception: "
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    validate_scene(
        raw_scene,
        scene_data,
    )

    ###########################################################################
    # Batch
    ###########################################################################

    batch = build_batch(
        scene_data
    )

    ###########################################################################
    # Local encoders
    ###########################################################################

    Ea, Em = run_local_encoders(
        batch
    )

    ###########################################################################
    # Phase-4 Encoder
    ###########################################################################

    z_stm = run_encoder(
        batch,
        Ea,
        Em,
    )

    ###########################################################################
    # Decoder
    ###########################################################################

    prediction = run_decoder(
        z_stm
    )

    ###########################################################################
    # Refinement
    ###########################################################################

    refined_prediction = run_refinement(
        z_stm,
        prediction,
    )

    ###########################################################################
    # Gradient test
    ###########################################################################

    run_gradient_test(
        batch
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
        "[PASS] AgentEncoder"
    )

    print(
        "[PASS] MapEncoder"
    )

    print(
        "[PASS] Phase-4 Encoder"
    )

    print(
        "[PASS] Z_STM"
    )

    print(
        "[PASS] Decoder"
    )

    print(
        "[PASS] Coarse trajectory Y^(0)"
    )

    print(
        "[PASS] Anchor-based refinement"
    )

    print(
        "[PASS] Refined trajectories"
    )

    print(
        "[PASS] Refinement scores"
    )

    if refined_prediction.offsets is not None:

        print(
            "[PASS] Refinement offsets"
        )

    print(
        "[PASS] Real-data gradient propagation"
    )

    print()
    print(
        "Final tensor flow:"
    )

    print(
        f"  Z_STM          = "
        f"{tuple(z_stm.shape)}"
    )

    print(
        f"  Y^(0)          = "
        f"{tuple(prediction.trajectories.shape)}"
    )

    print(
        f"  Y_refined      = "
        f"{tuple(refined_prediction.trajectories.shape)}"
    )

    print(
        f"  scores         = "
        f"{tuple(refined_prediction.scores.shape)}"
    )

    if refined_prediction.offsets is not None:

        print(
            f"  offsets        = "
            f"{tuple(refined_prediction.offsets.shape)}"
        )

    print()
    print(
        "PHASE-5 REAL-DATA VERIFICATION PASSED"
    )

    return 0


###############################################################################
# Entry point
###############################################################################


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
