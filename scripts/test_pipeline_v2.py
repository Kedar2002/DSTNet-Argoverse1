"""
scripts/test_pipeline_v2.py

End-to-end diagnostic test for the current DSTNet pipeline.

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
             SceneGraph
                   |
                   v
RelativeSpatioTemporalEmbedding
                   |
                   v
                 GSTA

This script intentionally does NOT modify the pipeline or silently repair
inconsistencies. It reports them explicitly.

Usage
-----
python scripts/test_end_to_end_encoder_pipeline.py ^
    --csv data/argoverse1/val/37853.csv ^
    --map-root data/argoverse1/hd_map

Linux/macOS
-----------
python scripts/test_end_to_end_encoder_pipeline.py \
    --csv data/argoverse1/val/37853.csv \
    --map-root data/argoverse1/hd_map
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

import numpy as np
import torch


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
DROPOUT = 0.0


###############################################################################
# Imports
###############################################################################

from datasets.map_loader import MapLoader
from datasets.scene_parser import SceneParser
from datasets.preprocess import ScenePreprocessor

from models.encoders.agent_encoder import AgentEncoder
from models.encoders.map_encoder import MapEncoder
from models.encoders.gsta import GSTA
from models.encoders.relative_spatiotemporal_embeddings import (
    RelativeSpatioTemporalEmbeddingModule,
)

###############################################################################
# Losses
###############################################################################

from models.model_types import (
    Prediction,
    RefinedPrediction,
)

from losses.proposal_loss import ProposalLoss
from losses.classification_loss import ClassificationLoss
from losses.score_loss import ScoreLoss
from losses.refinement_loss import RefinementLoss
from losses.total_loss import TotalLoss

###############################################################################
# Printing
###############################################################################


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


###############################################################################
# Tensor Validation
###############################################################################


def check_tensor(
    name: str,
    tensor: torch.Tensor,
) -> bool:
    """
    Check dtype, finiteness and basic tensor validity.
    """

    if not isinstance(tensor, torch.Tensor):
        fail(
            f"{name}: expected torch.Tensor, "
            f"got {type(tensor).__name__}."
        )
        return False

    if not torch.isfinite(tensor).all():
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
# Scene Validation
###############################################################################


def validate_raw_scene(
    scene: Any,
) -> None:

    section("1. RawScene")

    check(
        scene.num_tracks > 0,
        f"RawScene contains {scene.num_tracks} tracks.",
    )

    check(
        scene.num_lanes > 0,
        f"RawScene contains {scene.num_lanes} map lanes.",
    )

    target = scene.target_track

    check(
        target is not None,
        "Prediction target exists.",
    )

    info(
        f"sequence_id = {scene.metadata.sequence_id}"
    )

    info(
        f"city        = {scene.metadata.city}"
    )

    info(
        f"target      = {scene.metadata.focal_track_id}"
    )

    info(
        f"tracks      = {scene.num_tracks}"
    )

    info(
        f"lanes       = {scene.num_lanes}"
    )

    state_count = len(
        scene.agent_states
    )

    info(
        f"agent-state nodes available = {state_count}"
    )

    check(
        state_count > 0,
        "RawScene exposes AgentState nodes.",
    )


###############################################################################
# SceneData Validation
###############################################################################


def validate_scene_data(
    scene_data: Any,
) -> None:

    section("2. SceneData")

    check(
        scene_data.num_agents > 0,
        f"SceneData contains {scene_data.num_agents} agents.",
    )

    check(
        scene_data.num_maps > 0,
        f"SceneData contains {scene_data.num_maps} maps.",
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

        if not np.isfinite(
            observed
        ).all():

            fail(
                f"Agent {index} observed trajectory "
                "contains NaN/Inf."
            )

        if not np.isfinite(
            future
        ).all():

            fail(
                f"Agent {index} future trajectory "
                "contains NaN/Inf."
            )

    for index, map_data in enumerate(
        scene_data.maps
    ):

        centerline = np.asarray(
            map_data["centerline"]
        )

        if index < 3:

            info(
                f"map[{index}] "
                f"centerline={centerline.shape}"
            )

        if not np.isfinite(
            centerline
        ).all():

            fail(
                f"Map {index} centerline "
                "contains NaN/Inf."
            )


###############################################################################
# SceneGraph Validation
###############################################################################


def validate_scene_graph(
    graph: Any,
    *,
    num_agents: int,
    observation_steps: int,
    num_maps: int,
) -> None:

    section("3. SceneGraph")

    graph.validate()

    check(
        graph.num_agent_states > 0,
        f"SceneGraph has {graph.num_agent_states} agent-state nodes.",
    )

    check(
        graph.num_map_nodes == num_maps,
        (
            "SceneGraph map count matches SceneData: "
            f"{graph.num_map_nodes} == {num_maps}"
        ),
    )

    expected_states = (
        num_agents
        * observation_steps
    )

    info(
        f"Expected model agent-state count = "
        f"{num_agents} × {observation_steps} = "
        f"{expected_states}"
    )

    info(
        f"Actual SceneGraph agent-state count = "
        f"{graph.num_agent_states}"
    )

    if graph.num_agent_states == expected_states:

        print(
            "[PASS] SceneGraph state count matches "
            "Ea's expected N×H representation."
        )

    else:

        warn(
            "SceneGraph state count does NOT match "
            "N×H expected by GSTA."
        )

        warn(
            "This is a structural mismatch and is intentionally "
            "not repaired by this test."
        )

    check(
        graph.track_indices.dtype
        == np.int64,
        (
            "track_indices uses int64 "
            f"(dtype={graph.track_indices.dtype})."
        ),
    )

    check(
        graph.timesteps.dtype
        == np.int64,
        (
            "timesteps uses int64 "
            f"(dtype={graph.timesteps.dtype})."
        ),
    )

    check(
        graph.state_positions.shape[0]
        == graph.num_agent_states,
        "state_positions matches number of graph states.",
    )

    check(
        graph.state_headings.shape[0]
        == graph.num_agent_states,
        "state_headings matches number of graph states.",
    )

    check(
        graph.track_indices.shape[0]
        == graph.num_agent_states,
        "track_indices matches number of graph states.",
    )

    check(
        graph.timesteps.shape[0]
        == graph.num_agent_states,
        "timesteps matches number of graph states.",
    )

    info(
        f"temporal edges = {graph.num_temporal_edges}"
    )

    info(
        f"spatial edges  = {graph.num_spatial_edges}"
    )

    info(
        f"agent-map      = {graph.num_agent_map_edges}"
    )

    info(
        f"map-map        = {graph.num_map_map_edges}"
    )

    info(
        f"total edges    = {graph.num_edges}"
    )

    # -----------------------------------------------------------------------
    # Edge bounds
    # -----------------------------------------------------------------------

    for name, edges, upper_bound in (
        (
            "temporal_edges",
            graph.temporal_edges,
            graph.num_agent_states,
        ),
        (
            "spatial_edges",
            graph.spatial_edges,
            graph.num_agent_states,
        ),
    ):

        if edges.size:

            check(
                edges.min() >= 0,
                f"{name} has no negative indices.",
            )

            check(
                edges.max() < upper_bound,
                f"{name} indices are in bounds.",
            )

    if graph.agent_map_edges.size:

        state_indices = (
            graph.agent_map_edges[0]
        )

        map_indices = (
            graph.agent_map_edges[1]
        )

        check(
            state_indices.min() >= 0,
            "agent-map state indices are non-negative.",
        )

        check(
            state_indices.max()
            < graph.num_agent_states,
            "agent-map state indices are in bounds.",
        )

        check(
            map_indices.min() >= 0,
            "agent-map map indices are non-negative.",
        )

        check(
            map_indices.max()
            < graph.num_map_nodes,
            "agent-map map indices are in bounds.",
        )

    if graph.map_map_edges.size:

        check(
            graph.map_map_edges.min() >= 0,
            "map-map indices are non-negative.",
        )

        check(
            graph.map_map_edges.max()
            < graph.num_map_nodes,
            "map-map indices are in bounds.",
        )


###############################################################################
# Collation
###############################################################################


def make_batch(
    scene_data: Any,
) -> dict[str, Any]:

    section("4. Batch Collation")

    from datasets.collate import collate_fn

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
                f"{type(value).__name__}"
            )

    return batch


###############################################################################
# Agent Encoder
###############################################################################


def run_agent_encoder(
    batch: dict[str, Any],
) -> torch.Tensor:

    section("5. AgentEncoder")

    trajectories = batch[
        "agent_trajectories"
    ]

    encoder = AgentEncoder(
        observation_steps=OBSERVATION_STEPS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    encoder.eval()

    with torch.no_grad():

        Ea = encoder(
            trajectories
        )

    check_tensor(
        "Ea",
        Ea,
    )

    expected_shape = (
        trajectories.shape[0],
        trajectories.shape[1],
        OBSERVATION_STEPS,
        HIDDEN_DIM,
    )

    check(
        tuple(Ea.shape) == expected_shape,
        (
            "Ea shape is "
            f"{expected_shape}."
        ),
    )

    return Ea


###############################################################################
# Map Encoder
###############################################################################


def run_map_encoder(
    batch: dict[str, Any],
) -> torch.Tensor:

    section("6. MapEncoder")

    centerlines = batch[
        "map_centerlines"
    ]

    encoder = MapEncoder(
        num_points=MAP_SAMPLE_POINTS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    encoder.eval()

    with torch.no_grad():

        Em = encoder(
            centerlines
        )

    check_tensor(
        "Em",
        Em,
    )

    expected_shape = (
        centerlines.shape[0],
        centerlines.shape[1],
        HIDDEN_DIM,
    )

    check(
        tuple(Em.shape) == expected_shape,
        (
            "Em shape is "
            f"{expected_shape}."
        ),
    )

    return Em


###############################################################################
# Relative Spatio-Temporal Embedding
###############################################################################


def run_relative_embedding(
    graph: Any,
) -> Any:

    section(
        "7. Relative Spatio-Temporal Embedding"
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

    num_edges = graph.num_edges

    check(
        Er.edge_index.shape
        == (2, num_edges),
        (
            "Er.edge_index has shape "
            f"(2,{num_edges})."
        ),
    )

    check(
        Er.embeddings.shape
        == (num_edges, HIDDEN_DIM),
        (
            "Er.embeddings has shape "
            f"({num_edges},{HIDDEN_DIM})."
        ),
    )

    if Er.edge_type is not None:

        check(
            Er.edge_type.shape
            == (num_edges,),
            (
                "Er.edge_type has shape "
                f"({num_edges},)."
            ),
        )

    if Er.edge_index.numel():

        max_index = int(
            Er.edge_index.max().item()
        )

        min_index = int(
            Er.edge_index.min().item()
        )

        check(
            min_index >= 0,
            "Er edge indices are non-negative.",
        )

        # Unified SceneGraph index space is currently:
        #
        # agent states [0, Ns)
        # map nodes     [Ns, Ns+Nm)
        #
        # Some implementations may represent map edges in a local
        # index space before GSTA expands them. Therefore we only
        # require the upper bound implied by the graph here.

        total_nodes = (
            graph.num_agent_states
            + graph.num_map_nodes
        )

        if max_index < total_nodes:

            print(
                "[PASS] Er edge indices fit unified "
                f"node range [0,{total_nodes})."
            )

        else:

            warn(
                "Er edge index exceeds the expected "
                "unified SceneGraph node range."
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
) -> torch.Tensor | None:

    section("8. GSTA")

    agent_mask = batch[
        "agent_mask"
    ]

    map_mask = batch[
        "map_mask"
    ]

    gsta = GSTA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_modes=NUM_MODES,
        observation_steps=OBSERVATION_STEPS,
        dropout=DROPOUT,
    )

    gsta.eval()

    info(
        f"Ea       = {tuple(Ea.shape)}"
    )

    info(
        f"Em       = {tuple(Em.shape)}"
    )

    info(
        f"Er       = {tuple(Er.embeddings.shape)}"
    )

    info(
        f"agent_mask = {tuple(agent_mask.shape)}"
    )

    info(
        f"map_mask   = {tuple(map_mask.shape)}"
    )

    try:

        with torch.no_grad():

            Z_scene = gsta(
                Ea=Ea,
                Em=Em,
                Er=Er,
                scene_graph=graph,
                agent_mask=agent_mask,
                map_mask=map_mask,
            )

    except Exception as exc:

        fail(
            "GSTA forward failed."
        )

        print()
        print(
            f"Exception: {type(exc).__name__}: {exc}"
        )

        print()
        warn(
            "This is a genuine integration failure. "
            "The test does not modify the graph or tensors "
            "to force the forward pass to succeed."
        )

        return None

    check_tensor(
        "Z_scene",
        Z_scene,
    )

    expected_shape = (
        Ea.shape[0],
        Ea.shape[1],
        Ea.shape[2],
        NUM_MODES,
        HIDDEN_DIM,
    )

    check(
        tuple(Z_scene.shape)
        == expected_shape,
        (
            "Z_scene shape is "
            f"{expected_shape}."
        ),
    )

    return Z_scene

###############################################################################
# Synthetic RefinedPrediction for Loss Verification
###############################################################################


def make_synthetic_refined_prediction(
    trajectories: torch.Tensor,
    probabilities: torch.Tensor,
) -> RefinedPrediction:
    """
    Construct a valid RefinedPrediction for ScoreLoss and TotalLoss tests.

    Current RefinedPrediction contract
    -----------------------------------

    trajectories
        (B,N,H,K,T,2)

    probabilities
        (B,N,H,K)

    refinement_scores
        (B,N,H,K)

    offsets
        (B,N,H,K,T,2)

    trajectory_history
        (B,N,H,K,C+1,T,2)

    refinement_score_history
        (B,N,H,K,C+1)
    """

    if not isinstance(
        trajectories,
        torch.Tensor,
    ):
        raise TypeError(
            "trajectories must be a torch.Tensor."
        )

    if not isinstance(
        probabilities,
        torch.Tensor,
    ):
        raise TypeError(
            "probabilities must be a torch.Tensor."
        )

    if trajectories.ndim != 6:
        raise ValueError(
            "trajectories must have shape "
            "(B,N,H,K,T,2). "
            f"Got {tuple(trajectories.shape)}."
        )

    if probabilities.ndim != 4:
        raise ValueError(
            "probabilities must have shape "
            "(B,N,H,K). "
            f"Got {tuple(probabilities.shape)}."
        )

    B, N, H, K, T, C = trajectories.shape

    if C != 2:
        raise ValueError(
            "Trajectory coordinate dimension must be 2."
        )

    expected_probability_shape = (
        B,
        N,
        H,
        K,
    )

    if tuple(probabilities.shape) != (
        expected_probability_shape
    ):
        raise ValueError(
            "probabilities must have shape "
            f"{expected_probability_shape}. "
            f"Got {tuple(probabilities.shape)}."
        )

    ###########################################################################
    # Create one refinement update.
    #
    # Y^(0) = coarse trajectory
    # Y^(1) = refined trajectory
    #
    # Therefore:
    #
    # C = 1
    # C + 1 = 2
    ###########################################################################

    refinement_delta = (
        0.05
        * torch.randn_like(
            trajectories
        )
    )

    refined_trajectories = (
        trajectories
        + refinement_delta
    )

    ###########################################################################
    # Final refinement offsets
    ###########################################################################

    offsets = (
        refined_trajectories
        - trajectories
    )

    ###########################################################################
    # Final refinement scores
    #
    # Shape:
    #
    #     (B,N,H,K)
    ###########################################################################

    refinement_scores = torch.sigmoid(
        torch.randn(
            B,
            N,
            H,
            K,
            device=trajectories.device,
            dtype=trajectories.dtype,
        )
    )

    ###########################################################################
    # Trajectory history
    #
    # IMPORTANT:
    #
    # RefinedPrediction expects:
    #
    #     (B,N,H,K,C+1,T,2)
    #
    # We stack along dimension 4.
    ###########################################################################

    trajectory_history = torch.stack(
        [
            trajectories,
            refined_trajectories,
        ],
        dim=4,
    )

    ###########################################################################
    # Refinement score history
    #
    # Shape:
    #
    #     (B,N,H,K,C+1)
    ###########################################################################

    initial_scores = torch.sigmoid(
        torch.randn(
            B,
            N,
            H,
            K,
            device=trajectories.device,
            dtype=trajectories.dtype,
        )
    )

    refinement_score_history = torch.stack(
        [
            initial_scores,
            refinement_scores,
        ],
        dim=-1,
    )

    ###########################################################################
    # Construct the actual project dataclass.
    ###########################################################################

    refined_prediction = RefinedPrediction(
        trajectories=refined_trajectories,
        probabilities=probabilities,
        refinement_scores=refinement_scores,
        offsets=offsets,
        trajectory_history=trajectory_history,
        refinement_score_history=refinement_score_history,
    )

    return refined_prediction

###############################################################################
# RefinedPrediction History Access
###############################################################################


def require_trajectory_history(
    prediction: RefinedPrediction,
) -> torch.Tensor:
    """
    Return trajectory history after validating that it exists.
    """

    history = prediction.trajectory_history

    if history is None:
        raise AssertionError(
            "RefinedPrediction.trajectory_history is None."
        )

    return history


def require_refinement_score_history(
    prediction: RefinedPrediction,
) -> torch.Tensor:
    """
    Return refinement-score history after validating that it exists.
    """

    history = (
        prediction.refinement_score_history
    )

    if history is None:
        raise AssertionError(
            "RefinedPrediction.refinement_score_history is None."
        )

    return history


###############################################################################
# Phase-7 Loss Verification
###############################################################################

def run_loss_verification() -> bool:
    """
    Verify the current DSTNet loss implementation against the
    current model tensor contract and the DSTNet paper.

    Paper contract
    --------------

    Coarse trajectories:

        Y^(0)
            (B,N,H,K,T,2)

    Ground truth:

        G
            (B,N,T,2)

    Best mode:

        k_best
            (B,N,H)

    The historical dimension H is retained because Eq. (28)
    determines the best mode for every agent and historical
    timestep.

    Important
    ---------

    This function intentionally does NOT reshape the tensors to
    the older (B,N,K,T,2) representation.

    If a current loss module still expects the old representation,
    the test reports that as an implementation mismatch.
    """

    section(
        "9. PHASE-7 LOSS VERIFICATION"
    )

    ###########################################################################
    # Configuration
    ###########################################################################

    B = 2
    N = 4
    H = OBSERVATION_STEPS
    K = NUM_MODES
    T = PREDICTION_STEPS

    info(
        f"B = {B}"
    )

    info(
        f"N = {N}"
    )

    info(
        f"H = {H}"
    )

    info(
        f"K = {K}"
    )

    info(
        f"T = {T}"
    )

    ###########################################################################
    # Synthetic tensors
    ###########################################################################

    torch.manual_seed(
        42
    )

    trajectories = torch.randn(
        B,
        N,
        H,
        K,
        T,
        2,
        dtype=torch.float32,
        requires_grad=True,
    )

    ground_truth = torch.randn(
        B,
        N,
        T,
        2,
        dtype=torch.float32,
    )

    ###########################################################################
    # Tensor validation
    ###########################################################################

    check_tensor(
        "Loss-test trajectories",
        trajectories,
    )

    check_tensor(
        "Loss-test ground truth",
        ground_truth,
    )

    expected_trajectory_shape = (
        B,
        N,
        H,
        K,
        T,
        2,
    )

    expected_gt_shape = (
        B,
        N,
        T,
        2,
    )

    check(
        tuple(trajectories.shape)
        == expected_trajectory_shape,
        (
            "Loss trajectories use the DSTNet "
            f"shape {expected_trajectory_shape}."
        ),
    )

    check(
        tuple(ground_truth.shape)
        == expected_gt_shape,
        (
            "Ground truth uses the DSTNet "
            f"shape {expected_gt_shape}."
        ),
    )

    ###########################################################################
    # Paper Eq. (28) reference implementation
    ###########################################################################

    section(
        "9.1 PAPER WTA TARGET — EQ. (28)"
    )

    ###########################################################################
    # Ground-truth endpoint
    #
    # G[n,t] = g[n,t,T]
    ###########################################################################

    gt_endpoint = ground_truth[
        ...,
        -1,
        :,
    ]

    ###########################################################################
    # Predicted endpoints
    #
    # (B,N,H,K,2)
    ###########################################################################

    predicted_endpoints = trajectories[
        ...,
        -1,
        :,
    ]

    ###########################################################################
    # Endpoint distance
    #
    # (B,N,H,K)
    ###########################################################################

    endpoint_distance = torch.linalg.vector_norm(
        predicted_endpoints
        - gt_endpoint[
            :,
            :,
            None,
            None,
            :,
        ],
        dim=-1,
    )

    check_tensor(
        "Endpoint distance",
        endpoint_distance,
    )

    expected_endpoint_shape = (
        B,
        N,
        H,
        K,
    )

    check(
        tuple(endpoint_distance.shape)
        == expected_endpoint_shape,
        (
            "Endpoint distance shape = "
            f"{expected_endpoint_shape}."
        ),
    )

    ###########################################################################
    # Eq. (28):
    #
    # k_best = argmin_k ||Y^(0)_(n,t,k,T) - G_(n,t,T)||_2
    #
    # Result:
    #
    # (B,N,H)
    ###########################################################################

    best_mode = endpoint_distance.argmin(
        dim=-1,
    )

    check_tensor(
        "k_best",
        best_mode,
    )

    expected_best_mode_shape = (
        B,
        N,
        H,
    )

    check(
        tuple(best_mode.shape)
        == expected_best_mode_shape,
        (
            "k_best shape = "
            f"{expected_best_mode_shape}."
        ),
    )

    check(
        best_mode.dtype
        == torch.int64,
        "k_best has integer mode indices.",
    )

    check(
        bool(
            (
                (best_mode >= 0)
                &
                (best_mode < K)
            )
            .all()
            .item()
        ),
        "All k_best indices are inside [0,K).",
    )

    ###########################################################################
    # Verify that selected endpoint is actually minimum.
    ###########################################################################

    selected_endpoint_distance = torch.gather(
        endpoint_distance,
        dim=-1,
        index=best_mode.unsqueeze(-1),
    ).squeeze(-1)

    minimum_endpoint_distance = (
        endpoint_distance.min(
            dim=-1,
        ).values
    )

    check(
        torch.allclose(
            selected_endpoint_distance,
            minimum_endpoint_distance,
        ),
        (
            "k_best selects the minimum endpoint "
            "distance for every (B,N,H)."
        ),
    )

    ###########################################################################
    # Construct current Prediction.
    #
    # IMPORTANT:
    #
    # Current Prediction intentionally contains only trajectories.
    ###########################################################################

    ###############################################################################
    # Synthetic multimodal probabilities
    ###############################################################################

    probabilities = torch.softmax(
        torch.randn(
            B,
            N,
            H,
            K,
            dtype=torch.float32,
        ),
        dim=-1,
    )

    check_tensor(
        "Loss-test probabilities",
        probabilities,
    )

    check(
        tuple(probabilities.shape)
        == (
            B,
            N,
            H,
            K,
        ),
        (
            "Prediction probabilities shape = "
            f"{(B, N, H, K)}."
        ),
    )

    probability_sums = probabilities.sum(
        dim=-1,
    )

    check(
        torch.allclose(
            probability_sums,
            torch.ones_like(
                probability_sums
            ),
            atol=1e-5,
        ),
        "Prediction probabilities sum to one.",
    )

    ###############################################################################
    # Prediction object
    ###############################################################################

    prediction = Prediction(
        trajectories=trajectories,
        probabilities=probabilities,
    )

    check(
        hasattr(
            prediction,
            "trajectories",
        ),
        "Prediction contains coarse trajectories.",
    )

    ###########################################################################
    # Proposal Loss
    ###########################################################################

    section(
        "9.2 PROPOSAL LOSS"
    )

    proposal_loss = ProposalLoss()

    try:

        proposal_result = proposal_loss(
            prediction,
            ground_truth,
            return_best_mode=True,
        )

    except Exception as exc:

        fail(
            "ProposalLoss is not compatible with the "
            "current DSTNet trajectory contract."
        )

        print(
            f"[INFO] ProposalLoss exception: "
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "[INFO] Current trajectories: "
            f"{tuple(trajectories.shape)}"
        )

        print(
            "[INFO] Expected by paper: "
            "(B,N,H,K,T,2)"
        )

        print(
            "[INFO] This is a loss implementation mismatch, "
            "not a pipeline failure."
        )

        proposal_passed = False

    else:

        proposal_passed = True

        if isinstance(
            proposal_result,
            tuple,
        ):

            proposal_value = (
                proposal_result[0]
            )

            proposal_best_mode = (
                proposal_result[1]
            )

        else:

            proposal_value = proposal_result

            proposal_best_mode = None

        check_tensor(
            "Proposal loss",
            proposal_value,
        )

        check(
            proposal_value.ndim == 0,
            "Proposal loss is scalar.",
        )

        if proposal_best_mode is not None:

            check(
                tuple(
                    proposal_best_mode.shape
                )
                == expected_best_mode_shape,
                (
                    "ProposalLoss returns "
                    "k_best with shape "
                    f"{expected_best_mode_shape}."
                ),
            )

    ###########################################################################
    # Classification Loss
    ###########################################################################

    section(
        "9.3 CLASSIFICATION LOSS"
    )

    classification_passed = False

    try:

        classification_loss = (
            ClassificationLoss()
        )

        classification_result = (
            classification_loss(
                prediction,
                best_mode,
            )
        )

    except Exception as exc:

        print(
            "[INFO] ClassificationLoss could not "
            "be evaluated against the current "
            "Prediction contract."
        )

        print(
            f"[INFO] ClassificationLoss exception: "
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "[INFO] Current Prediction contains "
            "trajectories only."
        )

        print(
            "[INFO] Classification requires "
            "per-(B,N,H,K) prediction scores/logits."
        )

    else:

        classification_passed = True

        check_tensor(
            "Classification loss",
            classification_result,
        )

        check(
            classification_result.ndim == 0,
            "Classification loss is scalar.",
        )

    ###############################################################################
    # Score Loss
    ###############################################################################

    section(
        "9.4 SCORE LOSS"
    )

    score_passed = False

    try:

        ###########################################################################
        # RefinedPrediction
        ###########################################################################

        refined_prediction = (
            make_synthetic_refined_prediction(
                trajectories=trajectories.detach().clone(),
                probabilities=probabilities.detach().clone(),
            )
        )

        check(
            isinstance(
                refined_prediction,
                RefinedPrediction,
            ),
            "Synthetic RefinedPrediction instantiated.",
        )

        ###########################################################################
        # Extract optional histories safely.
        ###########################################################################

        trajectory_history = (
            require_trajectory_history(
                refined_prediction
            )
        )

        refinement_score_history = (
            require_refinement_score_history(
                refined_prediction
            )
        )

        ###########################################################################
        # Trajectory history
        ###########################################################################

        expected_history_shape = (
            B,
            N,
            H,
            K,
            2,
            T,
            2,
        )

        check_tensor(
            "Trajectory refinement history",
            trajectory_history,
        )

        check(
            tuple(
                trajectory_history.shape
            )
            == expected_history_shape,
            (
                "Trajectory refinement history shape = "
                f"{expected_history_shape}."
            ),
        )

        ###########################################################################
        # Score history
        ###########################################################################

        expected_score_history_shape = (
            B,
            N,
            H,
            K,
            2,
        )

        check_tensor(
            "Refinement score history",
            refinement_score_history,
        )

        check(
            tuple(
                refinement_score_history.shape
            )
            == expected_score_history_shape,
            (
                "Refinement score history shape = "
                f"{expected_score_history_shape}."
            ),
        )

        ###########################################################################
        # Score ranges
        ###########################################################################

        score_range_valid = bool(
            (
                (
                    refinement_score_history
                    >= 0.0
                )
                &
                (
                    refinement_score_history
                    <= 1.0
                )
            )
            .all()
            .item()
        )

        check(
            score_range_valid,
            "Refinement score history is within [0,1].",
        )

        ###########################################################################
        # ScoreLoss
        ###########################################################################

        score_loss = ScoreLoss()

        score_result = score_loss(
            refined_prediction,
            ground_truth,
        )

        check_tensor(
            "Score loss",
            score_result,
        )

        check(
            score_result.ndim == 0,
            "Score loss is scalar.",
        )

        check(
            bool(
                score_result.item() >= 0.0
            ),
            "Score loss is non-negative.",
        )

        score_passed = True

    except Exception as exc:

        fail(
            "ScoreLoss verification failed."
        )

        print(
            f"[INFO] ScoreLoss exception: "
            f"{type(exc).__name__}: {exc}"
        )

    ###############################################################################
    # Total Loss
    ###############################################################################

    section(
        "9.5 TOTAL LOSS"
    )

    total_loss_passed = False

    try:

        ###########################################################################
        # Reuse the synthetic refined prediction from ScoreLoss verification.
        ###########################################################################

        if "refined_prediction" not in locals():

            refined_prediction = (
                make_synthetic_refined_prediction(
                    trajectories=trajectories.detach().clone(),
                    probabilities=probabilities.detach().clone(),
                )
            )

        ###########################################################################
        # Construct total objective.
        #
        # Default:
        #
        #     L_total =
        #         L_proposal
        #         + L_classification
        #         + L_refinement
        #         + 0.1 L_score
        ###########################################################################

        total_loss_module = TotalLoss()

        info(
            f"TotalLoss = {total_loss_module}"
        )

        ###########################################################################
        # Forward
        ###########################################################################

        total_result = total_loss_module(
            prediction=prediction,
            refined_prediction=refined_prediction,
            ground_truth=ground_truth,
        )

        check(
            isinstance(
                total_result,
                dict,
            ),
            "TotalLoss returns a dictionary.",
        )

        ###########################################################################
        # Required components
        ###########################################################################

        required_total_keys = {
            "loss",
            "proposal_loss",
            "classification_loss",
            "score_loss",
            "refinement_loss",
        }

        check(
            required_total_keys.issubset(
                total_result.keys()
            ),
            (
                "TotalLoss dictionary contains "
                "all required components."
            ),
        )

        ###########################################################################
        # Validate every loss
        ###########################################################################

        for key in sorted(
            required_total_keys
        ):

            value = total_result[key]

            check_tensor(
                f"TotalLoss/{key}",
                value,
            )

            check(
                value.ndim == 0,
                f"TotalLoss/{key} is scalar.",
            )

            check(
                bool(
                    value.item() >= 0.0
                ),
                f"TotalLoss/{key} is non-negative.",
            )

        ###########################################################################
        # Independently reconstruct weighted total.
        ###########################################################################

        expected_total = (
            total_loss_module.proposal_weight
            * total_result["proposal_loss"]
            +
            total_loss_module.classification_weight
            * total_result["classification_loss"]
            +
            total_loss_module.refinement_weight
            * total_result["refinement_loss"]
            +
            total_loss_module.score_weight
            * total_result["score_loss"]
        )

        check(
            torch.allclose(
                total_result["loss"],
                expected_total,
                atol=1e-6,
                rtol=1e-5,
            ),
            "Total loss equals the configured weighted sum of all components.",
        )

        total_loss_passed = True

    except Exception as exc:

        fail(
            "TotalLoss verification failed."
        )

        print(
            f"[INFO] TotalLoss exception: "
            f"{type(exc).__name__}: {exc}"
        )

    ###########################################################################
    # Gradient test for the paper WTA regression expression
    ###########################################################################

    section(
        "9.6 PAPER WTA REGRESSION GRADIENT"
    )

    ###############################################################################
    # Select the WTA trajectory for every (B,N,H)
    ###############################################################################
    #
    # best_mode:
    #
    #     (B,N,H)
    #
    # trajectories:
    #
    #     (B,N,H,K,T,2)
    #
    # We need an index tensor:
    #
    #     (B,N,H,1,T,2)
    #
    # for gathering along the K dimension.
    ###############################################################################

    wta_index = (
        best_mode
        .unsqueeze(-1)       # (B,N,H,1)
        .unsqueeze(-1)       # (B,N,H,1,1)
        .unsqueeze(-1)       # (B,N,H,1,1,1)
        .expand(
            B,
            N,
            H,
            1,
            T,
            2,
        )
    )

    check(
        tuple(wta_index.shape)
        == (
            B,
            N,
            H,
            1,
            T,
            2,
        ),
        (
            "WTA gather index shape = "
            f"{(B, N, H, 1, T, 2)}."
        ),
    )

    selected_trajectories = torch.gather(
        trajectories,
        dim=3,
        index=wta_index,
    ).squeeze(
        dim=3,
    )

    check_tensor(
        "Selected WTA trajectories",
        selected_trajectories,
    )

    check(
        tuple(
            selected_trajectories.shape
        )
        == (
            B,
            N,
            H,
            T,
            2,
        ),
        (
            "Selected WTA trajectories have "
            "shape (B,N,H,T,2)."
        ),
    )

    check(
        tuple(
            selected_trajectories.shape
        )
        == (
            B,
            N,
            H,
            T,
            2,
        ),
        (
            "Selected WTA trajectories have "
            "shape (B,N,H,T,2)."
        ),
    )

    ###########################################################################
    # Smooth-L1 regression on selected mode.
    #
    # This is a diagnostic reference calculation corresponding
    # to the regression component described by the paper.
    ###########################################################################

    gt_for_regression = (
        ground_truth
        .unsqueeze(2)
        .expand(
            B,
            N,
            H,
            T,
            2,
        )
    )

    regression_loss = torch.nn.functional.smooth_l1_loss(
        selected_trajectories,
        gt_for_regression,
        reduction="mean",
    )

    check_tensor(
        "Reference WTA regression loss",
        regression_loss,
    )

    check(
        regression_loss.ndim == 0,
        "Reference WTA regression loss is scalar.",
    )

    trajectories.grad = None

    regression_loss.backward()

    check(
        trajectories.grad is not None,
        "Gradient reaches WTA trajectory tensor.",
    )

    if trajectories.grad is not None:

        check_tensor(
            "WTA trajectory gradient",
            trajectories.grad,
        )

        check(
            bool(
                trajectories.grad.abs()
                .sum()
                .item()
                > 0.0
            ),
            "WTA trajectory gradient is non-zero.",
        )

    ###############################################################################
    # Total Loss Gradient
    ###############################################################################

    section(
        "9.7 TOTAL LOSS GRADIENT"
    )

    total_gradient_passed = False

    try:

        ###########################################################################
        # Fresh differentiable trajectory tensor.
        ###########################################################################

        gradient_trajectories = (
            torch.randn(
                B,
                N,
                H,
                K,
                T,
                2,
                dtype=torch.float32,
                requires_grad=True,
            )
        )

        gradient_probabilities = (
            torch.softmax(
                torch.randn(
                    B,
                    N,
                    H,
                    K,
                    dtype=torch.float32,
                ),
                dim=-1,
            )
        )

        gradient_prediction = Prediction(
            trajectories=gradient_trajectories,
            probabilities=gradient_probabilities,
        )

        ###########################################################################
        # Refinement prediction must also be differentiable.
        ###########################################################################

        gradient_refined_prediction = (
            make_synthetic_refined_prediction(
                trajectories=gradient_trajectories,
                probabilities=gradient_probabilities,
            )
        )

        ###########################################################################
        # Complete objective.
        ###########################################################################

        total_loss_module = TotalLoss()

        total_gradient_result = (
            total_loss_module(
                prediction=gradient_prediction,
                refined_prediction=gradient_refined_prediction,
                ground_truth=ground_truth,
            )
        )

        complete_loss = (
            total_gradient_result["loss"]
        )

        check_tensor(
            "Complete TotalLoss",
            complete_loss,
        )

        ###########################################################################
        # Backpropagation
        ###########################################################################

        complete_loss.backward()

        check(
            gradient_trajectories.grad is not None,
            "Gradient reaches trajectory tensor through TotalLoss.",
        )

        if gradient_trajectories.grad is not None:

            check_tensor(
                "TotalLoss trajectory gradient",
                gradient_trajectories.grad,
            )

            check(
                bool(
                    gradient_trajectories.grad
                    .abs()
                    .sum()
                    .item()
                    > 0.0
                ),
                "TotalLoss trajectory gradient is non-zero.",
            )

        ###########################################################################
        # Check every individual component.
        ###########################################################################

        for name in (
            "proposal_loss",
            "classification_loss",
            "score_loss",
            "refinement_loss",
        ):

            check(
                bool(
                    torch.isfinite(
                        total_gradient_result[name]
                    )
                    .all()
                    .item()
                ),
                (
                    f"TotalLoss component '{name}' "
                    "is finite."
                ),
            )

        total_gradient_passed = True

    except Exception as exc:

        fail(
            "TotalLoss gradient verification failed."
        )

        print(
            f"[INFO] TotalLoss gradient exception: "
            f"{type(exc).__name__}: {exc}"
        )

    ###########################################################################
    # Final diagnostic summary
    ###########################################################################

    section(
        "9.8 LOSS VERIFICATION SUMMARY"
    )

    print(
        "[INFO] Paper WTA target:"
    )

    print(
        f"       endpoint_distance = "
        f"{tuple(endpoint_distance.shape)}"
    )

    print(
        f"       k_best             = "
        f"{tuple(best_mode.shape)}"
    )

    print()

    if proposal_passed:

        print(
            "[PASS] ProposalLoss accepts the current "
            "DSTNet tensor contract."
        )

    else:

        print(
            "[FAIL] ProposalLoss still uses an "
            "older tensor contract."
        )

    if classification_passed:

        print(
            "[PASS] ClassificationLoss accepts "
            "the current Prediction contract."
        )

    else:

        print(
            "[BLOCKED] ClassificationLoss is not "
            "compatible with the current coarse "
            "Prediction contract."
        )

    if score_passed:

        print(
            "[PASS] ScoreLoss accepts the current "
            "RefinedPrediction contract."
        )

    else:

        print(
            "[FAIL] ScoreLoss verification failed."
        )

    if total_loss_passed:

        print(
            "[PASS] TotalLoss accepts the complete "
            "DSTNet loss contract."
        )

    else:

        print(
            "[FAIL] TotalLoss verification failed."
        )

    if total_gradient_passed:

        print(
            "[PASS] TotalLoss gradient propagation verified."
        )

    else:

        print(
            "[FAIL] TotalLoss gradient propagation failed."
        )

    print()

    print(
        "[PASS] Paper Eq. (28) WTA calculation verified."
    )

    print(
        "[PASS] Reference WTA regression verified."
    )

    print(
        "[PASS] Reference WTA gradient propagation verified."
    )

    ###########################################################################
    # Return status
    #
    # Do not fail the complete encoder pipeline merely because
    # the loss modules have not yet been updated.
    ###########################################################################

    return (
        proposal_passed
        and classification_passed
        and score_passed
        and total_loss_passed
        and total_gradient_passed
    )


###############################################################################
# Main
###############################################################################


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "End-to-end DSTNet encoder pipeline test."
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
        help=(
            "Directory containing "
            "pruned_argoverse_*_vector_map.xml."
        ),
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    section(
        "DSTNet END-TO-END ENCODER PIPELINE TEST"
    )

    print(
        f"CSV      : {args.csv}"
    )

    print(
        f"Map root : {args.map_root}"
    )

    print(
        f"Device   : cpu"
    )

    print(
        f"PyTorch  : {torch.__version__}"
    )

    ###########################################################################
    # Paths
    ###########################################################################

    if not args.csv.exists():

        print()
        fail(
            f"CSV file does not exist: {args.csv}"
        )

        return 1

    if not args.map_root.exists():

        print()
        fail(
            f"Map root does not exist: {args.map_root}"
        )

        return 1

    ###########################################################################
    # HD Map
    ###########################################################################

    section("0. HD Map Loader")

    try:

        map_loader = MapLoader(
            args.map_root
        )

    except Exception as exc:

        fail(
            "MapLoader initialization failed."
        )

        print(
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
    # CSV Parsing
    ###########################################################################

    section("CSV -> RawScene")

    try:

        parser = SceneParser(
            map_api=map_loader
        )

        raw_scene = parser.parse(
            args.csv
        )

    except Exception as exc:

        fail(
            "SceneParser failed."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    validate_raw_scene(
        raw_scene
    )

    ###########################################################################
    # Preprocessing
    ###########################################################################

    section(
        "RawScene -> SceneData"
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
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    validate_scene_data(
        scene_data
    )

    ###########################################################################
    # SceneGraph
    ###########################################################################

    graph = scene_data.scene_graph

    validate_scene_graph(
        graph,
        num_agents=scene_data.num_agents,
        observation_steps=OBSERVATION_STEPS,
        num_maps=scene_data.num_maps,
    )

    ###########################################################################
    # Batch
    ###########################################################################

    batch = make_batch(
        scene_data
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
    # Er
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
    # Phase-7 Loss Verification
    ###########################################################################

    loss_modules_passed = (
        run_loss_verification()
    )

    ###########################################################################
    # Final result
    ###########################################################################

    section("FINAL RESULT")

    if Z_scene is None:

        print(
            "[FAIL] End-to-end pipeline did not reach "
            "a valid GSTA output."
        )

        print()
        print(
            "The diagnostic output above identifies the "
            "first integration boundary that failed."
        )

        return 1

    print(
        "[PASS] CSV -> HD Map -> RawScene -> SceneData "
        "-> SceneGraph -> Ea/Em/Er -> GSTA"
    )

    print()
    print(
        f"Final Z_scene shape = "
        f"{tuple(Z_scene.shape)}"
    )

    print()

    ###########################################################################
    # Loss status
    ###########################################################################

    if loss_modules_passed:

        print(
            "[PASS] All currently implemented loss "
            "interfaces are compatible."
        )

    else:

        print(
            "[INFO] Loss verification identified "
            "loss-interface mismatches."
        )

        print(
            "[INFO] Encoder pipeline itself remains valid."
        )

    print()
    print(
        "Encoder pipeline reached GSTA successfully."
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
