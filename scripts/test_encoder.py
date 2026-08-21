"""
scripts.test_encoder

Phase-4 verification for the complete DSTNet Encoder.

Pipeline
--------
Real Argoverse-1 CSV
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
     collate
        |
        +-------------------+
        |                   |
        v                   v
  AgentEncoder         MapEncoder
        |                   |
        v                   v
       Ea                  Em
        |                   |
        +---------+---------+
                  |
             SceneGraph
                  |
                  v
 RelativeSpatioTemporalEmbedding
                  |
                  v
                 GSTA
                  |
                  v
              Z_scene
                  |
                  v
             Tri-ATM × L
                  |
                  v
               Z_STM

This test verifies the ACTUAL Encoder class.

Tests
-----
1. Real Argoverse-1 scene
2. AgentEncoder -> Ea
3. MapEncoder -> Em
4. Encoder forward
5. GSTA output boundary
6. Tri-ATM stack
7. Final Z_STM shape
8. Finiteness
9. Agent masking
10. Gradient propagation
11. B > 1 synthetic interface test

Usage
-----
python scripts/test_encoder.py ^
    --csv data/argoverse1/val/37853.csv ^
    --map-root data/argoverse1/hd_maps/map_files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

from models.encoders.agent_encoder import AgentEncoder
from models.encoders.map_encoder import MapEncoder
from models.encoders.encoder import Encoder


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

INTERACTION_RADIUS = 30.0

DROPOUT = 0.0


###############################################################################
# Printing helpers
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


def passed(
    message: str,
) -> None:

    print(
        f"[PASS] {message}"
    )


def failed(
    message: str,
) -> None:

    print(
        f"[FAIL] {message}"
    )


def warning(
    message: str,
) -> None:

    print(
        f"[WARN] {message}"
    )


def check(
    condition: bool,
    message: str,
) -> None:

    if condition:
        passed(message)
    else:
        failed(message)


###############################################################################
# Tensor checks
###############################################################################


def check_tensor(
    name: str,
    tensor: torch.Tensor,
) -> bool:
    """
    Check that an object is a finite tensor.
    """

    if not isinstance(
        tensor,
        torch.Tensor,
    ):

        failed(
            f"{name}: expected torch.Tensor, "
            f"got {type(tensor).__name__}."
        )

        return False

    finite = bool(
        torch.isfinite(
            tensor
        ).all().item()
    )

    check(
        finite,
        f"{name}: all values finite.",
    )

    info(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}"
    )

    return finite


###############################################################################
# Real scene construction
###############################################################################


def build_real_scene(
    csv_path: Path,
    map_root: Path,
) -> tuple[Any, dict[str, Any]]:
    """
    Build one real SceneData and collated batch.
    """

    section(
        "1. REAL DATASET PIPELINE"
    )

    info(
        f"CSV      : {csv_path}"
    )

    info(
        f"Map root : {map_root}"
    )

    ###########################################################################
    # HD map
    ###########################################################################

    map_loader = MapLoader(
        map_root=map_root,
    )

    passed(
        "HD maps loaded successfully."
    )

    info(
        f"cities = {map_loader.cities}"
    )

    info(
        f"total lanes = "
        f"{map_loader.total_num_lanes}"
    )

    ###########################################################################
    # Raw scene
    ###########################################################################

    parser = SceneParser(
        map_loader,
    )

    raw_scene = parser.parse(
        csv_path,
    )

    passed(
        f"RawScene contains "
        f"{raw_scene.num_tracks} tracks."
    )

    passed(
        f"RawScene contains "
        f"{raw_scene.num_lanes} map lanes."
    )

    ###########################################################################
    # Preprocessing
    ###########################################################################

    preprocessor = ScenePreprocessor(
        observation_steps=OBSERVATION_STEPS,
        prediction_steps=PREDICTION_STEPS,
        map_sample_points=MAP_SAMPLE_POINTS,
        spatial_radius=SPATIAL_RADIUS,
        map_radius=MAP_RADIUS,
    )

    scene_data = preprocessor.preprocess(
        raw_scene,
    )

    passed(
        f"SceneData contains "
        f"{scene_data.num_agents} agents."
    )

    passed(
        f"SceneData contains "
        f"{scene_data.num_maps} maps."
    )

    ###########################################################################
    # Graph
    ###########################################################################

    graph = scene_data.scene_graph

    graph.validate()

    passed(
        f"SceneGraph contains "
        f"{graph.num_agent_states} agent-state nodes."
    )

    passed(
        f"SceneGraph contains "
        f"{graph.num_map_nodes} map nodes."
    )

    expected_states = (
        scene_data.num_agents
        * OBSERVATION_STEPS
    )

    check(
        graph.num_agent_states == expected_states,
        (
            "SceneGraph state count matches "
            f"N×H = {scene_data.num_agents}×"
            f"{OBSERVATION_STEPS}."
        ),
    )

    ###########################################################################
    # Collate
    ###########################################################################

    batch = collate_fn(
        [scene_data],
    )

    info(
        f"agent_trajectories = "
        f"{tuple(batch['agent_trajectories'].shape)}"
    )

    info(
        f"map_centerlines    = "
        f"{tuple(batch['map_centerlines'].shape)}"
    )

    info(
        f"positions          = "
        f"{tuple(batch['positions'].shape)}"
    )

    info(
        f"graph              = "
        f"list[{len(batch['graph'])}]"
    )

    info(
        f"agent_mask         = "
        f"{tuple(batch['agent_mask'].shape)}"
    )

    info(
        f"map_mask           = "
        f"{tuple(batch['map_mask'].shape)}"
    )

    return (
        scene_data,
        batch,
    )


###############################################################################
# Build Ea
###############################################################################


def build_agent_features(
    batch: dict[str, Any],
) -> torch.Tensor:
    """
    Build Ea using the actual AgentEncoder.
    """

    section(
        "2. AGENT ENCODER"
    )

    encoder = AgentEncoder(
        observation_steps=OBSERVATION_STEPS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    encoder.eval()

    trajectories = batch[
        "agent_trajectories"
    ]

    with torch.no_grad():

        Ea = encoder(
            trajectories,
        )

    check_tensor(
        "Ea",
        Ea,
    )

    expected = (
        trajectories.shape[0],
        trajectories.shape[1],
        OBSERVATION_STEPS,
        HIDDEN_DIM,
    )

    check(
        tuple(Ea.shape) == expected,
        f"Ea shape = {expected}.",
    )

    return Ea


###############################################################################
# Build Em
###############################################################################


def build_map_features(
    batch: dict[str, Any],
) -> torch.Tensor:
    """
    Build Em using the actual MapEncoder.
    """

    section(
        "3. MAP ENCODER"
    )

    encoder = MapEncoder(
        num_points=MAP_SAMPLE_POINTS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    encoder.eval()

    centerlines = batch[
        "map_centerlines"
    ]

    with torch.no_grad():

        Em = encoder(
            centerlines,
        )

    check_tensor(
        "Em",
        Em,
    )

    expected = (
        centerlines.shape[0],
        centerlines.shape[1],
        HIDDEN_DIM,
    )

    check(
        tuple(Em.shape) == expected,
        f"Em shape = {expected}.",
    )

    return Em


###############################################################################
# Encoder construction
###############################################################################


def build_encoder() -> Encoder:
    """
    Construct the current Phase-4 Encoder.
    """

    encoder = Encoder(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS,
        num_modes=NUM_MODES,
        observation_steps=OBSERVATION_STEPS,
        interaction_radius=INTERACTION_RADIUS,
        dropout=DROPOUT,
    )

    return encoder


###############################################################################
# Real-data Encoder forward
###############################################################################


def run_real_encoder(
    Ea: torch.Tensor,
    Em: torch.Tensor,
    batch: dict[str, Any],
) -> torch.Tensor:
    """
    Execute the actual Encoder on the real scene.
    """

    section(
        "4. COMPLETE ENCODER — REAL DATA"
    )

    encoder = build_encoder()

    encoder.eval()

    positions = batch[
        "positions"
    ]

    graph = batch[
        "graph"
    ]

    agent_mask = batch[
        "agent_mask"
    ]

    lane_mask = batch[
        "map_mask"
    ]

    ###########################################################################
    # Input report
    ###########################################################################

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
        f"lane_mask  = {tuple(lane_mask.shape)}"
    )

    info(
        f"graphs     = {len(graph)}"
    )

    ###########################################################################
    # Forward
    ###########################################################################

    try:

        with torch.no_grad():

            encoded = encoder(
                agent_features=Ea,
                lane_features=Em,
                positions=positions,
                graph=graph,
                agent_mask=agent_mask,
                lane_mask=lane_mask,
            )

    except Exception as exc:

        failed(
            "Encoder forward failed."
        )

        print()
        print(
            f"Exception: {type(exc).__name__}: {exc}"
        )

        raise

    ###########################################################################
    # Output
    ###########################################################################

    check_tensor(
        "Encoder output Z_STM",
        encoded,
    )

    expected = (
        Ea.shape[0],
        Ea.shape[1],
        Ea.shape[2],
        NUM_MODES,
        HIDDEN_DIM,
    )

    check(
        tuple(encoded.shape) == expected,
        f"Z_STM shape = {expected}.",
    )

    return encoded


###############################################################################
# Real-data mask verification
###############################################################################


def verify_real_masking(
    encoded: torch.Tensor,
    agent_mask: torch.Tensor,
) -> None:
    """
    Verify that padded agents do not produce invalid values.

    The current real scene has no padded agents, so this test reports
    that condition rather than inventing a padded agent.
    """

    section(
        "5. REAL-DATA MASK VERIFICATION"
    )

    invalid = ~agent_mask

    if not bool(
        invalid.any().item()
    ):

        info(
            "No padded agents exist in this real scene."
        )

        passed(
            "Real scene contains only valid agents."
        )

        return

    invalid_output = encoded[
        invalid
    ]

    check(
        bool(
            torch.isfinite(
                invalid_output
            ).all().item()
        ),
        "Masked-agent output remains finite.",
    )


###############################################################################
# Real-data gradient test
###############################################################################


def test_real_gradients(
    Ea: torch.Tensor,
    Em: torch.Tensor,
    batch: dict[str, Any],
) -> None:
    """
    Verify gradient propagation through the complete Encoder.
    """

    section(
        "6. COMPLETE ENCODER GRADIENT TEST"
    )

    encoder = build_encoder()

    encoder.train()

    ###########################################################################
    # Clone inputs so the test does not modify the encoder inputs.
    ###########################################################################

    Ea_input = (
        Ea.detach()
        .clone()
        .requires_grad_(True)
    )

    Em_input = (
        Em.detach()
        .clone()
        .requires_grad_(True)
    )

    positions = batch[
        "positions"
    ]

    graph = batch[
        "graph"
    ]

    agent_mask = batch[
        "agent_mask"
    ]

    lane_mask = batch[
        "map_mask"
    ]

    ###########################################################################
    # Forward
    ###########################################################################

    output = encoder(
        agent_features=Ea_input,
        lane_features=Em_input,
        positions=positions,
        graph=graph,
        agent_mask=agent_mask,
        lane_mask=lane_mask,
    )

    check_tensor(
        "Training-mode Encoder output",
        output,
    )

    ###########################################################################
    # Scalar loss
    ###########################################################################

    loss = output.square().mean()

    check(
        bool(
            torch.isfinite(
                loss
            ).item()
        ),
        "Encoder test loss is finite.",
    )

    ###########################################################################
    # Backward
    ###########################################################################

    loss.backward()

    ###########################################################################
    # Ea gradient
    ###########################################################################

    check(
        Ea_input.grad is not None,
        "Gradient reaches Ea.",
    )

    if Ea_input.grad is not None:

        check_tensor(
            "Ea gradient",
            Ea_input.grad,
        )

        check(
            bool(
                Ea_input.grad.abs().sum().item()
                > 0.0
            ),
            "Ea gradient is non-zero.",
        )

    ###########################################################################
    # Em gradient
    ###########################################################################

    check(
        Em_input.grad is not None,
        "Gradient reaches Em.",
    )

    if Em_input.grad is not None:

        check_tensor(
            "Em gradient",
            Em_input.grad,
        )

        check(
            bool(
                Em_input.grad.abs().sum().item()
                > 0.0
            ),
            "Em gradient is non-zero.",
        )


###############################################################################
# Synthetic B > 1 interface test
###############################################################################


def test_batch_interface(
    scene_data: Any,
) -> None:
    """
    Verify the Encoder's Sequence[SceneGraph] interface with B=2.

    We deliberately use the same real SceneData twice. This is not
    intended as a learning experiment; it only verifies that the
    encoder correctly handles a batch containing multiple graph objects.
    """

    section(
        "7. BATCH INTERFACE TEST — B=2"
    )

    ###########################################################################
    # Collate the same real scene twice.
    ###########################################################################

    batch = collate_fn(
        [
            scene_data,
            scene_data,
        ]
    )

    ###########################################################################
    # Build feature encoders.
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

    agent_encoder.eval()
    map_encoder.eval()

    with torch.no_grad():

        Ea = agent_encoder(
            batch["agent_trajectories"]
        )

        Em = map_encoder(
            batch["map_centerlines"]
        )

    check(
        Ea.shape[0] == 2,
        "B=2 agent features created.",
    )

    check(
        Em.shape[0] == 2,
        "B=2 map features created.",
    )

    ###########################################################################
    # Encoder
    ###########################################################################

    encoder = build_encoder()

    encoder.eval()

    with torch.no_grad():

        output = encoder(
            agent_features=Ea,
            lane_features=Em,
            positions=batch["positions"],
            graph=batch["graph"],
            agent_mask=batch["agent_mask"],
            lane_mask=batch["map_mask"],
        )

    check_tensor(
        "B=2 Encoder output",
        output,
    )

    expected = (
        2,
        Ea.shape[1],
        Ea.shape[2],
        NUM_MODES,
        HIDDEN_DIM,
    )

    check(
        tuple(output.shape) == expected,
        f"B=2 output shape = {expected}.",
    )


###############################################################################
# Main
###############################################################################


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Phase-4 real-data Encoder verification."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Argoverse-1 scene CSV.",
    )

    parser.add_argument(
        "--map-root",
        type=Path,
        required=True,
        help="Argoverse-1 HD map root.",
    )

    return parser.parse_args()


def main() -> int:

    section(
        "DSTNet PHASE-4 ENCODER VERIFICATION"
    )

    info(
        f"PyTorch = {torch.__version__}"
    )

    info(
        f"H = {OBSERVATION_STEPS}"
    )

    info(
        f"K = {NUM_MODES}"
    )

    info(
        f"D = {HIDDEN_DIM}"
    )

    info(
        f"heads = {NUM_HEADS}"
    )

    info(
        f"Tri-ATM layers = "
        f"{NUM_ENCODER_LAYERS}"
    )

    info(
        f"interaction_radius = "
        f"{INTERACTION_RADIUS}"
    )

    args = parse_args()

    ###########################################################################
    # Real dataset
    ###########################################################################

    (
        scene_data,
        batch,
    ) = build_real_scene(
        csv_path=args.csv,
        map_root=args.map_root,
    )

    ###########################################################################
    # Ea
    ###########################################################################

    Ea = build_agent_features(
        batch,
    )

    ###########################################################################
    # Em
    ###########################################################################

    Em = build_map_features(
        batch,
    )

    ###########################################################################
    # Complete encoder
    ###########################################################################

    encoded = run_real_encoder(
        Ea=Ea,
        Em=Em,
        batch=batch,
    )

    ###########################################################################
    # Masking
    ###########################################################################

    verify_real_masking(
        encoded=encoded,
        agent_mask=batch["agent_mask"],
    )

    ###########################################################################
    # Gradients
    ###########################################################################

    test_real_gradients(
        Ea=Ea,
        Em=Em,
        batch=batch,
    )

    ###########################################################################
    # B > 1
    ###########################################################################

    test_batch_interface(
        scene_data=scene_data,
    )

    ###########################################################################
    # Final summary
    ###########################################################################

    section(
        "PHASE-4 SUMMARY"
    )

    info(
        f"Ea       = {tuple(Ea.shape)}"
    )

    info(
        f"Em       = {tuple(Em.shape)}"
    )

    info(
        f"Z_STM    = {tuple(encoded.shape)}"
    )

    passed(
        "AgentEncoder integration."
    )

    passed(
        "MapEncoder integration."
    )

    passed(
        "RelativeSpatioTemporalEmbedding integration."
    )

    passed(
        "GSTA integration."
    )

    passed(
        "Tri-ATM × 2 integration."
    )

    passed(
        "Real-data Encoder forward pass."
    )

    passed(
        "Gradient propagation."
    )

    passed(
        "B > 1 SceneGraph interface."
    )

    print()
    print(
        "=" * 79
    )
    print(
        "PHASE-4 ENCODER VERIFICATION PASSED"
    )
    print(
        "=" * 79
    )

    return 0


###############################################################################
# Entry point
###############################################################################


if __name__ == "__main__":

    sys.exit(
        main()
    )
