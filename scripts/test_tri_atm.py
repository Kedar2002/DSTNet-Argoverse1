"""
scripts.test_tri_atm_real

Real-data Phase-3 verification for DSTNet.

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
   SceneGraph
        |
        +------------------+
        |                  |
        v                  v
   AgentEncoder        MapEncoder
        |                  |
        +--------+---------+
                 |
                 v
                Ea
                Em
                Er
                 |
                 v
                GSTA
                 |
                 v
        Z_scene (B,N,H,K,D)
                 |
                 v
              Tri-ATM
          +------+------+
          |             |
         MSPA          MHCA
          |             |
          +------+------+
                 |
                MMIA
                 |
                 v
              Z_STM

This test uses a REAL Argoverse-1 scene and does not modify,
truncate, or repair the graph.

Usage
-----
python scripts/test_tri_atm_real.py ^
    --csv data/argoverse1/val/37853.csv ^
    --map-root data/argoverse1/hd_maps/map_files
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
# Existing Pipeline Components
###############################################################################

from datasets.map_loader import MapLoader
from datasets.scene_parser import SceneParser
from datasets.preprocess import ScenePreprocessor
from datasets.collate import collate_fn

from models.encoders.agent_encoder import AgentEncoder
from models.encoders.map_encoder import MapEncoder
from models.encoders.gsta import GSTA
from models.encoders.relative_spatiotemporal_embeddings import (
    RelativeSpatioTemporalEmbeddingModule,
)

from models.attention.mspa import MSPA
from models.attention.mhca import MHCA
from models.attention.mmia import MMIA
from models.attention.tri_atm import TriATM


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

INTERACTION_RADIUS = 30.0

WINDOW_SIZES = (
    2,
    4,
    8,
)

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


###############################################################################
# Validation helpers
###############################################################################


def check(
    condition: bool,
    message: str,
) -> None:

    if condition:
        passed(message)
    else:
        failed(message)


def check_tensor(
    name: str,
    tensor: torch.Tensor,
) -> bool:

    if not isinstance(
        tensor,
        torch.Tensor,
    ):

        failed(
            f"{name}: expected Tensor, "
            f"got {type(tensor).__name__}"
        )

        return False

    finite = bool(
        torch.isfinite(
            tensor
        ).all().item()
    )

    if not finite:

        failed(
            f"{name}: contains NaN or Inf."
        )

        return False

    passed(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}"
    )

    return True


###############################################################################
# Real scene construction
###############################################################################


def build_real_scene(
    csv_path: Path,
    map_root: Path,
) -> tuple[Any, Any, dict[str, Any]]:

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
    # Map
    ###########################################################################

    map_loader = MapLoader(
        map_root
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
    # RawScene
    ###########################################################################

    parser = SceneParser(
        map_api=map_loader
    )

    raw_scene = parser.parse(
        csv_path
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

    scene_data = preprocessor(
        raw_scene
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
    # SceneGraph
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
        graph.num_agent_states
        == expected_states,
        (
            "SceneGraph state count matches "
            f"N×H = {scene_data.num_agents}×"
            f"{OBSERVATION_STEPS}={expected_states}."
        ),
    )

    ###########################################################################
    # Collation
    ###########################################################################

    batch = collate_fn(
        [scene_data]
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

        value = batch[key]

        if isinstance(
            value,
            torch.Tensor,
        ):

            info(
                f"{key:22s} "
                f"{tuple(value.shape)}"
            )

    return (
        scene_data,
        graph,
        batch,
    )


###############################################################################
# Agent Encoder
###############################################################################


def run_agent_encoder(
    batch: dict[str, Any],
) -> torch.Tensor:

    section(
        "2. AgentEncoder"
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
            trajectories
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
# Map Encoder
###############################################################################


def run_map_encoder(
    batch: dict[str, Any],
) -> torch.Tensor:

    section(
        "3. MapEncoder"
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
            centerlines
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
# Relative Spatio-Temporal Embedding
###############################################################################


def run_relative_embedding(
    graph: Any,
) -> Any:

    section(
        "4. Relative Spatio-Temporal Embedding"
    )

    module = (
        RelativeSpatioTemporalEmbeddingModule(
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        )
    )

    module.eval()

    with torch.no_grad():

        Er = module(
            graph
        )

    check_tensor(
        "Er.edge_index",
        Er.edge_index,
    )

    check_tensor(
        "Er.embeddings",
        Er.embeddings,
    )

    if Er.edge_type is not None:

        check_tensor(
            "Er.edge_type",
            Er.edge_type,
        )

    check(
        tuple(
            Er.edge_index.shape
        )
        == (
            2,
            graph.num_edges,
        ),
        "Er.edge_index matches graph edge count.",
    )

    check(
        tuple(
            Er.embeddings.shape
        )
        == (
            graph.num_edges,
            HIDDEN_DIM,
        ),
        "Er.embeddings matches graph edge count.",
    )

    return Er


###############################################################################
# GSTA
###############################################################################


def run_gsta(
    Ea: torch.Tensor,
    Em: torch.Tensor,
    Er: Any,
    graph: Any,
    batch: dict[str, Any],
) -> torch.Tensor:

    section(
        "5. GSTA"
    )

    gsta = GSTA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_modes=NUM_MODES,
        observation_steps=OBSERVATION_STEPS,
        dropout=DROPOUT,
    )

    gsta.eval()

    agent_mask = batch[
        "agent_mask"
    ]

    map_mask = batch[
        "map_mask"
    ]

    info(
        f"Ea         = {tuple(Ea.shape)}"
    )

    info(
        f"Em         = {tuple(Em.shape)}"
    )

    info(
        f"Er         = {tuple(Er.embeddings.shape)}"
    )

    with torch.no_grad():

        Z_scene = gsta(
            Ea=Ea,
            Em=Em,
            Er=Er,
            scene_graph=graph,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )

    check_tensor(
        "Z_scene",
        Z_scene,
    )

    expected = (
        Ea.shape[0],
        Ea.shape[1],
        Ea.shape[2],
        NUM_MODES,
        HIDDEN_DIM,
    )

    check(
        tuple(Z_scene.shape) == expected,
        f"Z_scene shape = {expected}.",
    )

    return Z_scene


###############################################################################
# Phase-3 component verification
###############################################################################


def test_mspa_real(
    Z_scene: torch.Tensor,
    positions: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:

    section(
        "6. MSPA — REAL DATA"
    )

    mspa = MSPA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        dropout=DROPOUT,
    )

    mspa.eval()

    with torch.no_grad():

        output = mspa(
            Z_scene,
            positions,
            agent_mask=agent_mask,
        )

    check_tensor(
        "MSPA output",
        output,
    )

    check(
        tuple(output.shape)
        == tuple(Z_scene.shape),
        "MSPA preserves Z_scene shape.",
    )

    ###########################################################################
    # Real neighbourhood statistics
    ###########################################################################

    with torch.no_grad():

        delta = (
            positions[:, :, None, :]
            - positions[:, None, :, :]
        )

        distance = torch.linalg.norm(
            delta,
            dim=-1,
        )

        neighbourhood = (
            distance
            <= INTERACTION_RADIUS
        )

        valid = (
            agent_mask[:, :, None]
            & agent_mask[:, None, :]
        )

        neighbourhood &= valid

        counts = neighbourhood.sum(
            dim=-1
        )

        info(
            "Real-agent neighbourhood counts "
            f"within {INTERACTION_RADIUS}m:"
        )

        info(
            f"min = {int(counts.min().item())}"
        )

        info(
            f"max = {int(counts.max().item())}"
        )

        info(
            f"mean = {float(counts.float().mean().item()):.2f}"
        )

    return output


###############################################################################
# MHCA
###############################################################################


def test_mhca_real(
    Z_spatial: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:

    section(
        "7. MHCA — REAL DATA"
    )

    mhca = MHCA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        window_sizes=WINDOW_SIZES,
        dropout=DROPOUT,
    )

    mhca.eval()

    with torch.no_grad():

        output = mhca(
            Z_spatial,
            agent_mask=agent_mask,
        )

    check_tensor(
        "MHCA output",
        output,
    )

    check(
        tuple(output.shape)
        == tuple(Z_spatial.shape),
        "MHCA preserves MSPA output shape.",
    )

    ###########################################################################
    # Verify causal mask independently.
    ###########################################################################

    causal = mhca._causal_mask(
        OBSERVATION_STEPS,
        device=Z_spatial.device,
        dtype=Z_spatial.dtype,
    )

    lower = torch.tril(
        causal
    )

    upper = torch.triu(
        causal,
        diagonal=1,
    )

    check(
        bool(
            torch.all(
                lower == 0
            ).item()
        ),
        "MHCA allows current and historical states.",
    )

    future_values = upper[
        upper != 0
    ]

    check(
        bool(
            torch.all(
                future_values < 0
            ).item()
        ),
        "MHCA blocks future states.",
    )

    return output


###############################################################################
# MMIA
###############################################################################


def test_mmia_real(
    Z_temporal: torch.Tensor,
) -> torch.Tensor:

    section(
        "8. MMIA — REAL DATA"
    )

    mmia = MMIA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )

    mmia.eval()

    with torch.no_grad():

        output = mmia(
            Z_temporal
        )

    check_tensor(
        "MMIA output",
        output,
    )

    check(
        tuple(output.shape)
        == tuple(Z_temporal.shape),
        "MMIA preserves MHCA output shape.",
    )

    check(
        output.shape[3] == NUM_MODES,
        "MMIA preserves K prediction modes.",
    )

    return output


###############################################################################
# Full Tri-ATM
###############################################################################


def test_tri_atm_real(
    Z_scene: torch.Tensor,
    positions: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:

    section(
        "9. FULL TRI-ATM — REAL DATA"
    )

    tri_atm = TriATM(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        window_sizes=WINDOW_SIZES,
        dropout=DROPOUT,
    )

    tri_atm.eval()

    with torch.no_grad():

        Z_stm = tri_atm(
            Z_scene,
            positions,
            agent_mask=agent_mask,
        )

    check_tensor(
        "Z_STM",
        Z_stm,
    )

    expected = tuple(
        Z_scene.shape
    )

    check(
        tuple(Z_stm.shape) == expected,
        f"Z_STM shape = {expected}.",
    )

    ###########################################################################
    # Invalid agents, if any, should remain masked.
    ###########################################################################

    invalid = ~agent_mask

    if bool(
        invalid.any().item()
    ):

        invalid_features = Z_stm[
            invalid
        ]

        check(
            bool(
                torch.allclose(
                    invalid_features,
                    torch.zeros_like(
                        invalid_features
                    ),
                    atol=1e-6,
                )
            ),
            "Tri-ATM preserves agent masking.",
        )

    else:

        info(
            "No padded agents in this real scene."
        )

    return Z_stm


###############################################################################
# Real-data gradient verification
###############################################################################


def test_gradients_real(
    Z_scene: torch.Tensor,
    positions: torch.Tensor,
    agent_mask: torch.Tensor,
) -> None:

    section(
        "10. TRI-ATM GRADIENTS — REAL DATA"
    )

    tri_atm = TriATM(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        window_sizes=WINDOW_SIZES,
        dropout=DROPOUT,
    )

    tri_atm.train()

    Z_input = (
        Z_scene.detach()
        .clone()
        .requires_grad_(True)
    )

    output = tri_atm(
        Z_input,
        positions,
        agent_mask=agent_mask,
    )

    check_tensor(
        "Training-mode Tri-ATM output",
        output,
    )

    loss = output.square().mean()

    check(
        bool(
            torch.isfinite(
                loss
            ).item()
        ),
        "Tri-ATM loss is finite.",
    )

    loss.backward()

    check(
        Z_input.grad is not None,
        "Gradient reaches real Z_scene.",
    )

    if Z_input.grad is not None:

        check_tensor(
            "Real Z_scene gradient",
            Z_input.grad,
        )

        check(
            bool(
                Z_input.grad.abs().sum().item()
                > 0.0
            ),
            "Real Z_scene gradient is non-zero.",
        )


###############################################################################
# Main
###############################################################################


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Real-data Phase-3 Tri-ATM verification."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--map-root",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:

    section(
        "DSTNet PHASE-3 REAL-DATA VERIFICATION"
    )

    print(
        f"[INFO] PyTorch = {torch.__version__}"
    )

    info(
        f"B expected = 1"
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
        f"interaction_radius = "
        f"{INTERACTION_RADIUS}"
    )

    info(
        f"window_sizes = "
        f"{WINDOW_SIZES}"
    )

    args = parse_args()

    ###########################################################################
    # Real data
    ###########################################################################

    (
        scene_data,
        graph,
        batch,
    ) = build_real_scene(
        csv_path=args.csv,
        map_root=args.map_root,
    )

    ###########################################################################
    # Encoders
    ###########################################################################

    Ea = run_agent_encoder(
        batch
    )

    Em = run_map_encoder(
        batch
    )

    ###########################################################################
    # Relative embedding
    ###########################################################################

    Er = run_relative_embedding(
        graph
    )

    ###########################################################################
    # GSTA
    ###########################################################################

    Z_scene = run_gsta(
        Ea=Ea,
        Em=Em,
        Er=Er,
        graph=graph,
        batch=batch,
    )

    ###########################################################################
    # Real-data inputs
    ###########################################################################

    positions = batch[
        "positions"
    ]

    agent_mask = batch[
        "agent_mask"
    ]

    check_tensor(
        "Real positions",
        positions,
    )

    ###########################################################################
    # Phase-3 components
    ###########################################################################

    Z_spatial = test_mspa_real(
        Z_scene,
        positions,
        agent_mask,
    )

    Z_temporal = test_mhca_real(
        Z_spatial,
        agent_mask,
    )

    Z_mmia = test_mmia_real(
        Z_temporal
    )

    ###########################################################################
    # Full Tri-ATM
    ###########################################################################

    Z_stm = test_tri_atm_real(
        Z_scene,
        positions,
        agent_mask,
    )

    ###########################################################################
    # Staged vs full shape consistency
    ###########################################################################

    section(
        "11. PHASE-3 SHAPE CONSISTENCY"
    )

    check(
        tuple(Z_mmia.shape)
        == tuple(Z_stm.shape),
        "Staged and full Tri-ATM outputs have identical shapes.",
    )

    ###########################################################################
    # Gradient
    ###########################################################################

    test_gradients_real(
        Z_scene,
        positions,
        agent_mask,
    )

    ###########################################################################
    # Final
    ###########################################################################

    section(
        "FINAL RESULT"
    )

    passed(
        "Real Argoverse-1 scene reached GSTA."
    )

    passed(
        "Real Z_scene reached MSPA."
    )

    passed(
        "Real Z_scene reached MHCA."
    )

    passed(
        "Real Z_scene reached MMIA."
    )

    passed(
        "Real Z_scene reached complete Tri-ATM."
    )

    print()
    info(
        f"Z_scene = {tuple(Z_scene.shape)}"
    )

    info(
        f"Z^S     = {tuple(Z_spatial.shape)}"
    )

    info(
        f"Z^ST    = {tuple(Z_temporal.shape)}"
    )

    info(
        f"Z^MMIA  = {tuple(Z_mmia.shape)}"
    )

    info(
        f"Z^STM   = {tuple(Z_stm.shape)}"
    )

    print()
    print(
        "============================================================================="
    )
    print(
        "PHASE-3 REAL-DATA VERIFICATION PASSED"
    )
    print(
        "============================================================================="
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
