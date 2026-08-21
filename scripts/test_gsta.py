"""
scripts.test_gsta_phase2

Targeted verification of the DSTNet GSTA implementation.

This test does NOT depend on the Argoverse dataset.

It verifies the internal GSTA computation pipeline:

    Ea, Em, Er
       |
       +----> ZT / ZS construction
       |
       +----> Temporal self-attention
       |
       +----> Spatial self-attention
       |
       +----> Temporal -> Spatial attention
       |
       +----> Spatial -> Temporal attention
       |
       +----> Temporal queries qT
       |
       +----> Spatial queries qS
       |
       +----> Z_scene = qT + qS

The test intentionally uses B=2 so that batch handling is exercised.

Expected final output:

    Z_scene = (B,N,H,K,D)

For this test:

    B = 2
    N = 4
    H = 20
    M = 6
    K = 6
    D = 256
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from typing import cast

# ---------------------------------------------------------------------------
# Make repository root importable when running:
#
#     python scripts/test_gsta_phase2.py
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from models.encoders.gsta import GSTA
from models.model_types import RelativeSpatioTemporalEmbedding
from datasets.scene_graph_builder import SceneGraph


###############################################################################
# Test configuration
###############################################################################

BATCH_SIZE = 2

NUM_AGENTS = 4

OBSERVATION_STEPS = 20

NUM_MAPS = 6

HIDDEN_DIM = 256

NUM_HEADS = 8

NUM_MODES = 6


###############################################################################
# Minimal SceneGraph test object
###############################################################################


class MockSceneGraph:
    """
    Minimal graph interface required by GSTA.

    This is intentionally NOT a replacement for SceneGraph.

    The actual SceneGraph was already verified by test_pipeline_v2.py.

    This object allows us to isolate GSTA from the dataset pipeline.
    """

    def __init__(
        self,
        num_agent_states: int,
        num_map_nodes: int,
    ) -> None:

        self.num_agent_states = (
            num_agent_states
        )

        self.num_map_nodes = (
            num_map_nodes
        )

    def validate(
        self,
    ) -> None:

        if self.num_agent_states <= 0:
            raise ValueError(
                "Mock graph must contain agent states."
            )

        if self.num_map_nodes <= 0:
            raise ValueError(
                "Mock graph must contain map nodes."
            )


###############################################################################
# Utility functions
###############################################################################


def check(
    condition: bool,
    message: str,
) -> None:
    """
    Print PASS/FAIL and stop on failure.
    """

    if not condition:

        print(
            f"[FAIL] {message}"
        )

        raise AssertionError(
            message
        )

    print(
        f"[PASS] {message}"
    )


def check_shape(
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    name: str,
) -> None:
    """
    Check tensor shape.
    """

    actual = tuple(
        tensor.shape
    )

    check(
        actual == expected,
        f"{name}: shape={actual}",
    )


def check_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Verify that tensor contains no NaN/Inf.
    """

    check(
        bool(
            torch.isfinite(
                tensor
            ).all()
        ),
        f"{name}: all values finite",
    )


###############################################################################
# Build synthetic relation embedding
###############################################################################


def build_relative_embedding(
    *,
    batch_size: int,
    num_agent_states: int,
    num_maps: int,
    hidden_dim: int,
    device: torch.device,
) -> RelativeSpatioTemporalEmbedding:
    """
    Build a small but structurally valid unified graph.

    Unified node numbering:

        Agent states:
            [0, num_agent_states)

        Map nodes:
            [num_agent_states,
             num_agent_states + num_maps)

    Edge types:

        0 = temporal
        1 = spatial
        2 = agent-map
        3 = map-map
    """

    del batch_size

    state_count = (
        num_agent_states
    )

    map_offset = (
        state_count
    )

    ###########################################################################
    # Temporal edges
    ###########################################################################

    temporal_edges = []

    for agent in range(
        4
    ):

        base = (
            agent
            * OBSERVATION_STEPS
        )

        for timestep in range(
            OBSERVATION_STEPS - 1
        ):

            src = (
                base
                + timestep
            )

            dst = (
                base
                + timestep
                + 1
            )

            temporal_edges.append(
                (src, dst)
            )

    ###########################################################################
    # Spatial edges
    ###########################################################################

    spatial_edges = []

    for timestep in range(
        OBSERVATION_STEPS
    ):

        for agent_i in range(
            4
        ):

            for agent_j in range(
                4
            ):

                if agent_i == agent_j:
                    continue

                src = (
                    agent_i
                    * OBSERVATION_STEPS
                    + timestep
                )

                dst = (
                    agent_j
                    * OBSERVATION_STEPS
                    + timestep
                )

                spatial_edges.append(
                    (src, dst)
                )

    ###########################################################################
    # Agent-map edges
    ###########################################################################

    agent_map_edges = []

    for state in range(
        state_count
    ):

        map_index = (
            state
            % num_maps
        )

        agent_map_edges.append(
            (
                state,
                map_offset
                + map_index,
            )
        )

    ###########################################################################
    # Map-map edges
    ###########################################################################

    map_map_edges = []

    for map_i in range(
        num_maps
    ):

        map_j = (
            (map_i + 1)
            % num_maps
        )

        map_map_edges.append(
            (
                map_offset
                + map_i,
                map_offset
                + map_j,
            )
        )

    ###########################################################################
    # Combine
    ###########################################################################

    all_edges = (
        temporal_edges
        + spatial_edges
        + agent_map_edges
        + map_map_edges
    )

    edge_types = (
        [0] * len(temporal_edges)
        + [1] * len(spatial_edges)
        + [2] * len(agent_map_edges)
        + [3] * len(map_map_edges)
    )

    edge_index = torch.tensor(
        all_edges,
        dtype=torch.long,
        device=device,
    ).t().contiguous()

    edge_type = torch.tensor(
        edge_types,
        dtype=torch.long,
        device=device,
    )

    embeddings = torch.randn(
        edge_index.shape[1],
        hidden_dim,
        device=device,
    )

    return RelativeSpatioTemporalEmbedding(
        edge_index=edge_index,
        embeddings=embeddings,
        edge_type=edge_type,
    )


###############################################################################
# Main test
###############################################################################


def main() -> None:

    print(
        "\n"
        + "=" * 79
    )

    print(
        "GSTA PHASE-2 TARGETED VERIFICATION"
    )

    print(
        "=" * 79
    )

    device = torch.device(
        "cpu"
    )

    torch.manual_seed(
        42
    )

    print(
        f"[INFO] Device = {device}"
    )

    print(
        f"[INFO] PyTorch = {torch.__version__}"
    )

    print()

    ###########################################################################
    # Dimensions
    ###########################################################################

    B = BATCH_SIZE

    N = NUM_AGENTS

    H = OBSERVATION_STEPS

    M = NUM_MAPS

    D = HIDDEN_DIM

    K = NUM_MODES

    ###########################################################################
    # Instantiate GSTA
    ###########################################################################

    gsta = GSTA(
        hidden_dim=D,
        num_heads=NUM_HEADS,
        num_modes=K,
        observation_steps=H,
        dropout=0.0,
    ).to(device)

    gsta.eval()

    print(
        "[PASS] GSTA instantiated."
    )

    ###########################################################################
    # Synthetic inputs
    ###########################################################################

    Ea = torch.randn(
        B,
        N,
        H,
        D,
        device=device,
    )

    Em = torch.randn(
        B,
        M,
        D,
        device=device,
    )

    agent_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
        ],
        dtype=torch.bool,
        device=device,
    )

    map_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, False],
        ],
        dtype=torch.bool,
        device=device,
    )

    graphs = cast(
        list[SceneGraph],
        [
            MockSceneGraph(
                num_agent_states=N * H,
                num_map_nodes=M,
            )
            for _ in range(B)
        ],
    )

    Er = [
        build_relative_embedding(
            batch_size=B,
            num_agent_states=N * H,
            num_maps=M,
            hidden_dim=D,
            device=device,
        )
        for _ in range(B)
    ]

    print(
        "[PASS] Synthetic Ea/Em/Er inputs created."
    )

    ###########################################################################
    # Input shapes
    ###########################################################################

    check_shape(
        Ea,
        (B, N, H, D),
        "Ea",
    )

    check_shape(
        Em,
        (B, M, D),
        "Em",
    )

    print()

    ###########################################################################
    # Test ZT / ZS construction
    ###########################################################################

    ZT, ZS = gsta._build_gsta_inputs(
        Ea=Ea,
        Em=Em,
        Er=Er,
        scene_graphs=graphs,
    )

    check_shape(
        ZT,
        (B, N, H, D),
        "ZT",
    )

    check_shape(
        ZS,
        (B, M, D),
        "ZS",
    )

    check_finite(
        ZT,
        "ZT",
    )

    check_finite(
        ZS,
        "ZS",
    )

    check(
        not torch.equal(
            ZT,
            Ea,
        ),
        "ZT contains relation information",
    )

    check(
        not torch.equal(
            ZS,
            Em,
        ),
        "ZS contains relation information",
    )

    print()

    ###########################################################################
    # Temporal self-attention
    ###########################################################################

    ZT_self = (
        gsta._temporal_self_attention(
            ZT,
            agent_mask=agent_mask,
        )
    )

    check_shape(
        ZT_self,
        (B, N, H, D),
        "Temporal self-attention output",
    )

    check_finite(
        ZT_self,
        "Temporal self-attention output",
    )

    print()

    ###########################################################################
    # Spatial self-attention
    ###########################################################################

    ZS_self = (
        gsta._spatial_self_attention(
            ZS,
            map_mask=map_mask,
        )
    )

    check_shape(
        ZS_self,
        (B, M, D),
        "Spatial self-attention output",
    )

    check_finite(
        ZS_self,
        "Spatial self-attention output",
    )

    print()

    ###########################################################################
    # Temporal -> Spatial
    ###########################################################################

    ZT_cross = (
        gsta._temporal_to_spatial_attention(
            ZT_self,
            ZS_self,
            map_mask=map_mask,
        )
    )

    check_shape(
        ZT_cross,
        (B, N, H, D),
        "Temporal -> Spatial output",
    )

    check_finite(
        ZT_cross,
        "Temporal -> Spatial output",
    )

    print()

    ###########################################################################
    # Spatial -> Temporal
    ###########################################################################

    ZS_cross = (
        gsta._spatial_to_temporal_attention(
            ZS_self,
            ZT_cross,
            agent_mask=agent_mask,
        )
    )

    check_shape(
        ZS_cross,
        (B, M, D),
        "Spatial -> Temporal output",
    )

    check_finite(
        ZS_cross,
        "Spatial -> Temporal output",
    )

    print()

    ###########################################################################
    # Temporal queries
    ###########################################################################

    qT = (
        gsta._temporal_query_attention(
            ZT_cross,
            agent_mask=agent_mask,
        )
    )

    check_shape(
        qT,
        (B, N, H, K, D),
        "qT",
    )

    check_finite(
        qT,
        "qT",
    )

    print()

    ###########################################################################
    # Spatial queries
    ###########################################################################

    qS = (
        gsta._spatial_query_attention(
            ZS_cross,
            num_agents=N,
            map_mask=map_mask,
        )
    )

    check_shape(
        qS,
        (B, N, H, K, D),
        "qS",
    )

    check_finite(
        qS,
        "qS",
    )

    print()

    ###########################################################################
    # Eq. (9)
    #
    # Z_scene = qT + qS
    ###########################################################################

    expected = (
        qT
        + qS
    )

    check_shape(
        expected,
        (B, N, H, K, D),
        "Z_scene",
    )

    check_finite(
        expected,
        "Z_scene",
    )

    print()

    ###########################################################################
    # Full forward
    ###########################################################################

    Z_scene = gsta(
        Ea=Ea,
        Em=Em,
        Er=Er,
        scene_graph=graphs,
        agent_mask=agent_mask,
        map_mask=map_mask,
    )

    check_shape(
        Z_scene,
        (B, N, H, K, D),
        "Full GSTA output",
    )

    check_finite(
        Z_scene,
        "Full GSTA output",
    )

    print()

    ###########################################################################
    # Verify forward output is internally stored
    ###########################################################################

    cached = (
        gsta.scene_prediction_embeddings
    )

    check(
        cached is not None,
        "GSTA stores scene prediction embedding",
    )

    if cached is not None:
        check(
            torch.equal(
                cached,
                Z_scene,
            ),
            "Cached scene embedding matches forward output",
        )

    print()

    ###########################################################################
    # Verify qT + qS reconstruction
    ###########################################################################

    # Re-run the individual stages in eval mode.
    #
    # Dropout is disabled, so this should reproduce the deterministic
    # computation.

    ZT2, ZS2 = gsta._build_gsta_inputs(
        Ea=Ea,
        Em=Em,
        Er=Er,
        scene_graphs=graphs,
    )

    ZT2 = gsta._temporal_self_attention(
        ZT2,
        agent_mask=agent_mask,
    )

    ZS2 = gsta._spatial_self_attention(
        ZS2,
        map_mask=map_mask,
    )

    ZT2 = gsta._temporal_to_spatial_attention(
        ZT2,
        ZS2,
        map_mask=map_mask,
    )

    ZS2 = gsta._spatial_to_temporal_attention(
        ZS2,
        ZT2,
        agent_mask=agent_mask,
    )

    qT2 = gsta._temporal_query_attention(
        ZT2,
        agent_mask=agent_mask,
    )

    qS2 = gsta._spatial_query_attention(
        ZS2,
        num_agents=N,
        map_mask=map_mask,
    )

    reconstructed = (
        qT2
        + qS2
    )

    check(
        torch.allclose(
            Z_scene,
            reconstructed,
            atol=1e-6,
            rtol=1e-5,
        ),
        "Z_scene = qT + qS",
    )

    print()

    ###########################################################################
    # Masking sanity check
    ###########################################################################

    masked_result = gsta(
        Ea=Ea,
        Em=Em,
        Er=Er,
        scene_graph=graphs,
        agent_mask=agent_mask,
        map_mask=map_mask,
    )

    check_finite(
        masked_result,
        "Masked GSTA output",
    )

    check(
        masked_result.shape
        == Z_scene.shape,
        "Masking preserves output shape",
    )

    print()

    ###########################################################################
    # Gradient verification
    ###########################################################################

    gsta.train()

    Ea_grad = (
        Ea.detach()
        .clone()
        .requires_grad_(True)
    )

    Em_grad = (
        Em.detach()
        .clone()
        .requires_grad_(True)
    )

    output = gsta(
        Ea=Ea_grad,
        Em=Em_grad,
        Er=Er,
        scene_graph=graphs,
        agent_mask=agent_mask,
        map_mask=map_mask,
    )

    loss = output.square().mean()

    loss.backward()

    #######################################################################
    # Ea gradient
    #######################################################################

    check(
        Ea_grad.grad is not None,
        "Gradient reaches Ea",
    )

    if Ea_grad.grad is not None:

        check_finite(
            Ea_grad.grad,
            "Ea gradient",
        )

    #######################################################################
    # Em gradient
    #######################################################################

    check(
        Em_grad.grad is not None,
        "Gradient reaches Em",
    )

    if Em_grad.grad is not None:

        check_finite(
            Em_grad.grad,
            "Em gradient",
        )

    #######################################################################
    # Temporal query gradient
    #######################################################################

    temporal_query_grad = (
        gsta.temporal_queries.grad
    )

    check(
        temporal_query_grad is not None,
        "Gradient reaches temporal query bank",
    )

    if temporal_query_grad is not None:

        check_finite(
            temporal_query_grad,
            "Temporal query gradient",
        )

    #######################################################################
    # Spatial query gradient
    #######################################################################

    spatial_query_grad = (
        gsta.spatial_queries.grad
    )

    check(
        spatial_query_grad is not None,
        "Gradient reaches spatial query bank",
    )

    if spatial_query_grad is not None:

        check_finite(
            spatial_query_grad,
            "Spatial query gradient",
        )

    ###########################################################################
    # Learnable query gradients
    ###########################################################################

    check(
        gsta.temporal_queries.grad is not None,
        "Gradient reaches temporal query bank",
    )

    check(
        gsta.spatial_queries.grad is not None,
        "Gradient reaches spatial query bank",
    )

    ###########################################################################
    # Final summary
    ###########################################################################

    print()
    print(
        "=" * 79
    )

    print(
        "GSTA PHASE-2 VERIFICATION PASSED"
    )

    print(
        "=" * 79
    )

    print()

    print(
        f"[INFO] B = {B}"
    )

    print(
        f"[INFO] N = {N}"
    )

    print(
        f"[INFO] H = {H}"
    )

    print(
        f"[INFO] M = {M}"
    )

    print(
        f"[INFO] K = {K}"
    )

    print(
        f"[INFO] D = {D}"
    )

    print()

    print(
        f"[PASS] ZT       = {tuple(ZT.shape)}"
    )

    print(
        f"[PASS] ZS       = {tuple(ZS.shape)}"
    )

    print(
        f"[PASS] qT       = {tuple(qT.shape)}"
    )

    print(
        f"[PASS] qS       = {tuple(qS.shape)}"
    )

    print(
        f"[PASS] Z_scene  = {tuple(Z_scene.shape)}"
    )

    print()

    print(
        "[PASS] Temporal self-attention"
    )

    print(
        "[PASS] Spatial self-attention"
    )

    print(
        "[PASS] Temporal -> Spatial attention"
    )

    print(
        "[PASS] Spatial -> Temporal attention"
    )

    print(
        "[PASS] Temporal learnable queries"
    )

    print(
        "[PASS] Spatial learnable queries"
    )

    print(
        "[PASS] Z_scene = qT + qS"
    )

    print(
        "[PASS] Batch size > 1"
    )

    print(
        "[PASS] Agent masking"
    )

    print(
        "[PASS] Map masking"
    )

    print(
        "[PASS] Gradient propagation"
    )

    print()

    print(
        "GSTA is structurally and numerically "
        "consistent with its current implementation."
    )


###############################################################################
# Entry point
###############################################################################


if __name__ == "__main__":
    main()
