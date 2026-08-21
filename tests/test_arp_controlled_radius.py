"""
tests/test_arp_controlled_radius.py

DSTNet MSPA vs ARP-MSPA
Controlled-Radius Modular Validation

Purpose
-------
Isolate and demonstrate the core novelty of ARP-MSPA:

    B_ij = -d_ij^2 / (2 r_i^2)

The experiment deliberately bypasses the neural radius predictor.

We:
    1. Generate synthetic GSTA-like scene embeddings.
    2. Create a controlled spatial scene.
    3. Use identical Q/K projections.
    4. Directly impose different target-agent radii.
    5. Measure how attention redistribution changes.
    6. Compare ARP-MSPA against original fixed-radius MSPA.

This is a MODULE-LEVEL validation.

It does NOT establish trajectory-prediction superiority.
It establishes that the proposed adaptive spatial mechanism
behaves according to the intended mathematical formulation.

Tensor contract
---------------
Z_scene   : (B,N,H,K,D)
positions : (B,N,2)
mask      : (B,N)

Reference
---------
DSTNet: Dynamic Trajectory Prediction for Autonomous Vehicles
via Spatio-Temporal Attention.

Original MSPA:
    fixed radius per attention head.

Proposed ARP-MSPA:
    learned target-agent radius
    + differentiable spatial bias.
"""

from __future__ import annotations

import math
import os
import sys

import torch
from torch import Tensor


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# ============================================================================
# IMPORTS
# ============================================================================

from models.attention.mspa import MSPA
from models.attention.arp_mspa import ARPMSPA


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 1
NUM_AGENTS = 12
HISTORY_STEPS = 20
NUM_MODES = 6
HIDDEN_DIM = 256
NUM_HEADS = 8

INTERACTION_RADIUS = 30.0

R_MIN = 5.0
R_MAX = 30.0

DROPOUT = 0.0


# ============================================================================
# PRINT HELPERS
# ============================================================================

def section(title: str) -> None:

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def subsection(title: str) -> None:

    print()
    print("-" * 88)
    print(title)
    print("-" * 88)


# ============================================================================
# SYNTHETIC GSTA INPUT
# ============================================================================

def create_synthetic_scene() -> tuple[
    Tensor,
    Tensor,
    Tensor,
]:
    """
    Create synthetic data having the same structural contract
    as the GSTA -> MSPA interface.

    Returns
    -------
    z_scene
        (B,N,H,K,D)

    positions
        (B,N,2)

    agent_mask
        (B,N)
    """

    torch.manual_seed(
        SEED
    )

    z_scene = torch.randn(
        BATCH_SIZE,
        NUM_AGENTS,
        HISTORY_STEPS,
        NUM_MODES,
        HIDDEN_DIM,
        device=DEVICE,
    )

    # ------------------------------------------------------------------------
    # Controlled geometry around target agent 0.
    #
    # Distances:
    #
    # 0, 3, 6, 10, 15, 20, 25, 29, 35, 45 ...
    # ------------------------------------------------------------------------

    positions = torch.tensor(
        [
            [
                [0.0, 0.0],       # 0
                [3.0, 0.0],       # 1
                [6.0, 0.0],       # 2
                [10.0, 0.0],      # 3
                [15.0, 0.0],      # 4
                [20.0, 0.0],      # 5
                [25.0, 0.0],      # 6
                [29.0, 0.0],      # 7
                [35.0, 0.0],      # 8
                [45.0, 0.0],      # 9
                [10.0, 10.0],     # 10 -> 14.14m
                [-10.0, 10.0],    # 11 -> 14.14m
            ]
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    agent_mask = torch.ones(
        BATCH_SIZE,
        NUM_AGENTS,
        dtype=torch.bool,
        device=DEVICE,
    )

    return (
        z_scene,
        positions,
        agent_mask,
    )


# ============================================================================
# PAIRWISE DISTANCES
# ============================================================================

def pairwise_distances(
    positions: Tensor,
) -> Tensor:
    """
    positions:
        (B,N,2)

    returns:
        (B,N,N)
    """

    delta = (
        positions[:, :, None, :]
        - positions[:, None, :, :]
    )

    return torch.linalg.norm(
        delta,
        dim=-1,
    )


# ============================================================================
# ATTENTION METRICS
# ============================================================================

def attention_metrics(
    attention: Tensor,
    distances: Tensor,
    target_agent: int = 0,
) -> dict[str, float]:
    """
    Calculate spatial attention statistics.

    attention:
        (B,N,N,H,K)

    distances:
        (B,N,N)
    """

    # ------------------------------------------------------------------------
    # Select target agent.
    #
    # Result:
    #
    # (N,H,K)
    # ------------------------------------------------------------------------

    weights = attention[
        0,
        target_agent,
    ]

    target_distances = distances[
        0,
        target_agent,
    ]

    # ------------------------------------------------------------------------
    # Average over heads/history/modes.
    # ------------------------------------------------------------------------

    weights = weights.mean(
        dim=(-1, -2)
    )

    # ------------------------------------------------------------------------
    # Expected spatial distance.
    # ------------------------------------------------------------------------

    expected_distance = (
        weights
        * target_distances
    ).sum()

    # ------------------------------------------------------------------------
    # Entropy.
    # ------------------------------------------------------------------------

    entropy = -(
        weights
        * torch.log(
            weights.clamp_min(
                1e-12
            )
        )
    ).sum()

    # ------------------------------------------------------------------------
    # Local mass.
    # ------------------------------------------------------------------------

    local_mass_10 = weights[
        target_distances <= 10.0
    ].sum()

    local_mass_15 = weights[
        target_distances <= 15.0
    ].sum()

    far_mass_20 = weights[
        target_distances > 20.0
    ].sum()

    return {
        "expected_distance":
            float(expected_distance.item()),

        "entropy":
            float(entropy.item()),

        "local_mass_10m":
            float(local_mass_10.item()),

        "local_mass_15m":
            float(local_mass_15.item()),

        "far_mass_20m":
            float(far_mass_20.item()),
    }


# ============================================================================
# ORIGINAL MSPA ATTENTION
# ============================================================================

@torch.no_grad()
def compute_mspa_attention(
    model: MSPA,
    z_scene: Tensor,
    positions: Tensor,
    head_index: int,
) -> Tensor:
    """
    Reconstruct the attention distribution of one original MSPA head.

    Returns
    -------
    attention
        (B,N,N,H,K)
    """

    B, N, H, K, D = z_scene.shape

    Q = model.query_projection(
        z_scene
    )

    Key = model.key_projection(
        z_scene
    )

    Q = Q.reshape(
        B,
        N,
        H,
        K,
        model.num_heads,
        model.head_dim,
    )

    Key = Key.reshape(
        B,
        N,
        H,
        K,
        model.num_heads,
        model.head_dim,
    )

    Q = Q.permute(
        0,
        4,
        1,
        2,
        3,
        5,
    )

    Key = Key.permute(
        0,
        4,
        1,
        2,
        3,
        5,
    )

    q = Q[
        :,
        head_index,
    ]

    k = Key[
        :,
        head_index,
    ]

    scores = torch.einsum(
        "bntkd,bmskd->bnmtk",
        q,
        k,
    )

    scores = (
        scores
        / math.sqrt(
            model.head_dim
        )
    )

    distances = pairwise_distances(
        positions
    )

    # ------------------------------------------------------------------------
    # Original MSPA uses one fixed radius per head.
    # ------------------------------------------------------------------------

    radii = model._interaction_radii()

    radius = radii[
        head_index
    ].to(
        device=positions.device,
        dtype=positions.dtype,
    )

    mask = (
        distances <= radius
    )

    identity = torch.eye(
        N,
        dtype=torch.bool,
        device=positions.device,
    )

    mask = (
        mask
        | identity.unsqueeze(0)
    )

    valid = (
        mask
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

    scores = scores.masked_fill(
        ~valid,
        torch.finfo(
            scores.dtype
        ).min,
    )

    attention = torch.softmax(
        scores,
        dim=2,
    )

    attention = attention.masked_fill(
        ~valid,
        0.0,
    )

    return attention


# ============================================================================
# CONTROLLED ARP-MSPA ATTENTION
# ============================================================================

@torch.no_grad()
def compute_arp_attention(
    model: ARPMSPA,
    z_scene: Tensor,
    positions: Tensor,
    controlled_radius: float,
    head_index: int,
) -> Tensor:
    """
    Compute ARP-MSPA attention while directly controlling r_i.

    This intentionally bypasses the learned radius predictor.

    controlled_radius:
        scalar radius applied to target agent 0.

    Returns
    -------
    attention:
        (B,N,N,H,K)
    """

    B, N, H, K, D = z_scene.shape

    Q = model.query_projection(
        z_scene
    )

    Key = model.key_projection(
        z_scene
    )

    Q = Q.reshape(
        B,
        N,
        H,
        K,
        model.num_heads,
        model.head_dim,
    )

    Key = Key.reshape(
        B,
        N,
        H,
        K,
        model.num_heads,
        model.head_dim,
    )

    Q = Q.permute(
        0,
        4,
        1,
        2,
        3,
        5,
    )

    Key = Key.permute(
        0,
        4,
        1,
        2,
        3,
        5,
    )

    q = Q[
        :,
        head_index,
    ]

    k = Key[
        :,
        head_index,
    ]

    scores = torch.einsum(
        "bntkd,bmskd->bnmtk",
        q,
        k,
    )

    scores = (
        scores
        / math.sqrt(
            model.head_dim
        )
    )

    distances = pairwise_distances(
        positions
    )

    # ------------------------------------------------------------------------
    # Construct controlled radius tensor.
    #
    # Shape:
    #
    # (B,N)
    #
    # Every target gets the controlled radius in this experiment.
    # ------------------------------------------------------------------------

    radius = torch.full(
        (
            B,
            N,
        ),
        float(controlled_radius),
        dtype=positions.dtype,
        device=positions.device,
    )

    # ------------------------------------------------------------------------
    # ARP-MSPA spatial bias.
    #
    # B_ij = -d_ij^2 / (2 r_i^2)
    # ------------------------------------------------------------------------

    bias = -(
        distances.square()
        / (
            2.0
            * radius.square().unsqueeze(-1)
        )
    )

    scores = (
        scores
        + bias.unsqueeze(-1).unsqueeze(-1)
    )

    # ------------------------------------------------------------------------
    # Maximum-radius candidate set.
    # ------------------------------------------------------------------------

    mask = (
        distances <= model.r_max
    )

    identity = torch.eye(
        N,
        dtype=torch.bool,
        device=positions.device,
    )

    mask = (
        mask
        | identity.unsqueeze(0)
    )

    valid = (
        mask
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

    scores = scores.masked_fill(
        ~valid,
        torch.finfo(
            scores.dtype
        ).min,
    )

    attention = torch.softmax(
        scores,
        dim=2,
    )

    attention = attention.masked_fill(
        ~valid,
        0.0,
    )

    return attention


# ============================================================================
# PRINT DISTRIBUTION
# ============================================================================

def print_distribution(
    attention: Tensor,
    distances: Tensor,
    label: str,
    target_agent: int = 0,
) -> None:

    print()
    print(
        label
    )

    weights = attention[
        0,
        target_agent,
    ].mean(
        dim=(-1, -2)
    )

    d = distances[
        0,
        target_agent,
    ]

    print()

    print(
        f"{'Agent':>8}"
        f"{'Distance':>14}"
        f"{'Attention':>16}"
    )

    print(
        "-" * 40
    )

    for agent_index in range(
        NUM_AGENTS
    ):

        print(
            f"{agent_index:>8}"
            f"{float(d[agent_index]):>14.3f}"
            f"{float(weights[agent_index]):>16.6f}"
        )


# ============================================================================
# TEST 1
# ============================================================================

def test_input_structure(
    z_scene: Tensor,
    positions: Tensor,
    agent_mask: Tensor,
) -> None:

    subsection(
        "TEST 1: SYNTHETIC GSTA -> MSPA / ARP-MSPA CONTRACT"
    )

    print(
        f"Z_scene   : {tuple(z_scene.shape)}"
    )

    print(
        f"positions : {tuple(positions.shape)}"
    )

    print(
        f"mask      : {tuple(agent_mask.shape)}"
    )

    assert z_scene.shape == (
        BATCH_SIZE,
        NUM_AGENTS,
        HISTORY_STEPS,
        NUM_MODES,
        HIDDEN_DIM,
    )

    assert positions.shape == (
        BATCH_SIZE,
        NUM_AGENTS,
        2,
    )

    assert agent_mask.shape == (
        BATCH_SIZE,
        NUM_AGENTS,
    )

    print(
        "[PASS] Synthetic data matches the GSTA -> MSPA interface."
    )


# ============================================================================
# TEST 2
# ============================================================================

def test_geometry(
    positions: Tensor,
) -> Tensor:

    subsection(
        "TEST 2: CONTROLLED SPATIAL GEOMETRY"
    )

    distances = pairwise_distances(
        positions
    )

    print(
        "Distances from target agent 0:"
    )

    for i in range(
        NUM_AGENTS
    ):

        print(
            f"  Agent {i:02d}: "
            f"{float(distances[0, 0, i]):8.3f} m"
        )

    print(
        "[PASS] Controlled geometry generated."
    )

    return distances


# ============================================================================
# TEST 3
# ============================================================================

def test_original_mspa(
    original: MSPA,
    z_scene: Tensor,
    positions: Tensor,
    distances: Tensor,
) -> None:

    subsection(
        "TEST 3: ORIGINAL MSPA FIXED-RADIUS BEHAVIOR"
    )

    local_head = 1
    global_head = NUM_HEADS - 1

    local_attention = compute_mspa_attention(
        original,
        z_scene,
        positions,
        local_head,
    )

    global_attention = compute_mspa_attention(
        original,
        z_scene,
        positions,
        global_head,
    )

    local_radius = float(
        original._interaction_radii()[
            local_head
        ].item()
    )

    global_radius = float(
        original._interaction_radii()[
            global_head
        ].item()
    )

    print(
        f"Local head radius  : {local_radius:.3f} m"
    )

    print(
        f"Global head radius : {global_radius:.3f} m"
    )

    local_metrics = attention_metrics(
        local_attention,
        distances,
    )

    global_metrics = attention_metrics(
        global_attention,
        distances,
    )

    print()

    print(
        f"{'Metric':<28}"
        f"{'Local Head':>18}"
        f"{'Global Head':>18}"
    )

    print(
        "-" * 64
    )

    print(
        f"{'Expected distance (m)':<28}"
        f"{local_metrics['expected_distance']:>18.5f}"
        f"{global_metrics['expected_distance']:>18.5f}"
    )

    print(
        f"{'Entropy':<28}"
        f"{local_metrics['entropy']:>18.5f}"
        f"{global_metrics['entropy']:>18.5f}"
    )

    print(
        f"{'Attention <= 10m':<28}"
        f"{local_metrics['local_mass_10m']:>18.5f}"
        f"{global_metrics['local_mass_10m']:>18.5f}"
    )

    print(
        f"{'Attention > 20m':<28}"
        f"{local_metrics['far_mass_20m']:>18.5f}"
        f"{global_metrics['far_mass_20m']:>18.5f}"
    )

    print()

    print_distribution(
        local_attention,
        distances,
        "Original MSPA - Local Head",
    )

    print_distribution(
        global_attention,
        distances,
        "Original MSPA - Global Head",
    )

    print()

    print(
        "[PASS] Original MSPA demonstrates fixed multi-scale spatial coverage."
    )


# ============================================================================
# TEST 4
# ============================================================================

def test_controlled_adaptation(
    proposed: ARPMSPA,
    z_scene: Tensor,
    positions: Tensor,
    distances: Tensor,
) -> None:

    subsection(
        "TEST 4: ARP-MSPA CONTROLLED RADIUS ADAPTATION"
    )

    # ------------------------------------------------------------------------
    # These are deliberately chosen to demonstrate the complete range.
    # ------------------------------------------------------------------------

    controlled_radii = [
        5.0,
        10.0,
        20.0,
        30.0,
    ]

    head_index = NUM_HEADS - 1

    results: dict[
        float,
        dict[str, float],
    ] = {}

    attentions: dict[
        float,
        Tensor,
    ] = {}

    for radius in controlled_radii:

        attention = compute_arp_attention(
            proposed,
            z_scene,
            positions,
            radius,
            head_index,
        )

        metrics = attention_metrics(
            attention,
            distances,
        )

        results[
            radius
        ] = metrics

        attentions[
            radius
        ] = attention

    print(
        "Controlled target-agent radii:"
    )

    print()

    print(
        f"{'Radius':>12}"
        f"{'Expected d':>18}"
        f"{'Entropy':>16}"
        f"{'<=10m':>16}"
        f"{'>20m':>16}"
    )

    print(
        "-" * 80
    )

    for radius in controlled_radii:

        metrics = results[
            radius
        ]

        print(
            f"{radius:>12.1f}"
            f"{metrics['expected_distance']:>18.5f}"
            f"{metrics['entropy']:>16.5f}"
            f"{metrics['local_mass_10m']:>16.5f}"
            f"{metrics['far_mass_20m']:>16.5f}"
        )

    # ------------------------------------------------------------------------
    # Print detailed distributions for the extreme cases.
    # ------------------------------------------------------------------------

    print_distribution(
        attentions[5.0],
        distances,
        "ARP-MSPA - r_i = 5m",
    )

    print_distribution(
        attentions[30.0],
        distances,
        "ARP-MSPA - r_i = 30m",
    )

    # ------------------------------------------------------------------------
    # Behavioral assertions.
    #
    # Smaller radius should produce:
    #   - lower expected distance
    #   - more local mass
    #
    # Larger radius should produce:
    #   - broader attention
    #   - more far-neighbour mass
    # ------------------------------------------------------------------------

    expected_5 = results[5.0][
        "expected_distance"
    ]

    expected_30 = results[30.0][
        "expected_distance"
    ]

    local_5 = results[5.0][
        "local_mass_10m"
    ]

    local_30 = results[30.0][
        "local_mass_10m"
    ]

    far_5 = results[5.0][
        "far_mass_20m"
    ]

    far_30 = results[30.0][
        "far_mass_20m"
    ]

    assert expected_5 < expected_30, (
        "Expected spatial distance should increase "
        "as radius increases."
    )

    assert local_5 > local_30, (
        "Small radius should produce more local attention."
    )

    assert far_5 < far_30, (
        "Large radius should produce more far-neighbour attention."
    )

    print()

    print(
        "[PASS] ARP-MSPA adapts spatial attention as r_i changes."
    )


# ============================================================================
# TEST 5
# ============================================================================

def test_spatial_bias_formula(
    proposed: ARPMSPA,
    positions: Tensor,
) -> None:

    subsection(
        "TEST 5: DIRECT VALIDATION OF ADAPTIVE SPATIAL BIAS"
    )

    distances = pairwise_distances(
        positions
    )

    target_agent = 0

    test_radii = [
        5.0,
        10.0,
        20.0,
        30.0,
    ]

    print(
        "Validating:"
    )

    print(
        "    B_ij = -d_ij^2 / (2 r_i^2)"
    )

    print()

    print(
        f"{'d (m)':>10}"
        f"{'r=5m':>16}"
        f"{'r=10m':>16}"
        f"{'r=20m':>16}"
        f"{'r=30m':>16}"
    )

    print(
        "-" * 76
    )

    test_distances = [
        0.0,
        3.0,
        6.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
    ]

    for d in test_distances:

        values = []

        for radius in test_radii:

            expected = -(
                d * d
            ) / (
                2.0
                * radius
                * radius
            )

            values.append(
                expected
            )

        print(
            f"{d:>10.2f}"
            f"{values[0]:>16.5f}"
            f"{values[1]:>16.5f}"
            f"{values[2]:>16.5f}"
            f"{values[3]:>16.5f}"
        )

    # ------------------------------------------------------------------------
    # Check actual implementation against analytical formula.
    # ------------------------------------------------------------------------

    radius_tensor = torch.full(
        (
            BATCH_SIZE,
            NUM_AGENTS,
        ),
        10.0,
        dtype=positions.dtype,
        device=positions.device,
    )

    actual_bias = proposed._adaptive_spatial_bias(
        positions,
        radius_tensor,
    )

    expected_bias = -(
        distances.square()
        / (
            2.0
            * radius_tensor.square().unsqueeze(-1)
        )
    )

    max_error = (
        actual_bias
        - expected_bias
    ).abs().max().item()

    print()

    print(
        f"Maximum implementation error: "
        f"{max_error:.10e}"
    )

    assert max_error < 1e-6

    print(
        "[PASS] ARP-MSPA implements the intended spatial-bias equation."
    )


# ============================================================================
# TEST 6
# ============================================================================

def test_same_geometry_different_radius(
    proposed: ARPMSPA,
    z_scene: Tensor,
    positions: Tensor,
) -> None:

    subsection(
        "TEST 6: SAME SCENE -> DIFFERENT ADAPTIVE REPRESENTATIONS"
    )

    attention_small = compute_arp_attention(
        proposed,
        z_scene,
        positions,
        controlled_radius=5.0,
        head_index=NUM_HEADS - 1,
    )

    attention_large = compute_arp_attention(
        proposed,
        z_scene,
        positions,
        controlled_radius=30.0,
        head_index=NUM_HEADS - 1,
    )

    difference = (
        attention_small
        - attention_large
    ).abs()

    mean_difference = (
        difference.mean().item()
    )

    max_difference = (
        difference.max().item()
    )

    print(
        f"Mean attention difference : "
        f"{mean_difference:.8f}"
    )

    print(
        f"Maximum attention difference : "
        f"{max_difference:.8f}"
    )

    assert mean_difference > 1e-5

    assert max_difference > 1e-4

    print()

    print(
        "[PASS] Same scene geometry produces different attention "
        "distributions when r_i changes."
    )


# ============================================================================
# TEST 7
# ============================================================================

def test_full_module_forward(
    original: MSPA,
    proposed: ARPMSPA,
    z_scene: Tensor,
    positions: Tensor,
    agent_mask: Tensor,
) -> None:

    subsection(
        "TEST 7: FULL MODULE FORWARD COMPATIBILITY"
    )

    original.eval()
    proposed.eval()

    with torch.no_grad():

        original_output = original(
            z_scene,
            positions,
            agent_mask=agent_mask,
        )

        proposed_output = proposed(
            z_scene,
            positions,
            agent_mask=agent_mask,
        )

    print(
        f"Original output : "
        f"{tuple(original_output.shape)}"
    )

    print(
        f"ARP-MSPA output : "
        f"{tuple(proposed_output.shape)}"
    )

    expected_shape = (
        BATCH_SIZE,
        NUM_AGENTS,
        HISTORY_STEPS,
        NUM_MODES,
        HIDDEN_DIM,
    )

    assert original_output.shape == expected_shape

    assert proposed_output.shape == expected_shape

    assert torch.isfinite(
        original_output
    ).all()

    assert torch.isfinite(
        proposed_output
    ).all()

    print()

    print(
        "[PASS] Both modules preserve the same tensor contract."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    section(
        "DSTNet MSPA vs ARP-MSPA"
    )

    print(
        "CONTROLLED-RADIUS MODULAR VALIDATION"
    )

    print()

    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Batch size           : {BATCH_SIZE}"
    )

    print(
        f"Agents               : {NUM_AGENTS}"
    )

    print(
        f"GSTA history H       : {HISTORY_STEPS}"
    )

    print(
        f"Modes K              : {NUM_MODES}"
    )

    print(
        f"Hidden dimension D  : {HIDDEN_DIM}"
    )

    print(
        f"Attention heads     : {NUM_HEADS}"
    )

    print(
        f"Interaction R       : {INTERACTION_RADIUS:.1f} m"
    )

    print(
        f"ARP range           : "
        f"[{R_MIN:.1f}, {R_MAX:.1f}] m"
    )

    # ------------------------------------------------------------------------
    # Reproducibility.
    # ------------------------------------------------------------------------

    torch.manual_seed(
        SEED
    )

    # ------------------------------------------------------------------------
    # Synthetic input.
    # ------------------------------------------------------------------------

    z_scene, positions, agent_mask = (
        create_synthetic_scene()
    )

    test_input_structure(
        z_scene,
        positions,
        agent_mask,
    )

    distances = test_geometry(
        positions
    )

    # ------------------------------------------------------------------------
    # Instantiate modules.
    # ------------------------------------------------------------------------

    subsection(
        "MODULE INITIALIZATION"
    )

    original = MSPA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        dropout=DROPOUT,
    ).to(
        DEVICE
    )

    proposed = ARPMSPA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        r_min=R_MIN,
        r_max=R_MAX,
        radius_hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(
        DEVICE
    )

    original.eval()
    proposed.eval()

    print(
        f"Original MSPA parameters : "
        f"{sum(p.numel() for p in original.parameters()):,}"
    )

    print(
        f"ARP-MSPA parameters       : "
        f"{sum(p.numel() for p in proposed.parameters()):,}"
    )

    # ------------------------------------------------------------------------
    # Run tests.
    # ------------------------------------------------------------------------

    test_original_mspa(
        original,
        z_scene,
        positions,
        distances,
    )

    test_controlled_adaptation(
        proposed,
        z_scene,
        positions,
        distances,
    )

    test_spatial_bias_formula(
        proposed,
        positions,
    )

    test_same_geometry_different_radius(
        proposed,
        z_scene,
        positions,
    )

    test_full_module_forward(
        original,
        proposed,
        z_scene,
        positions,
        agent_mask,
    )

    # ------------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------------

    section(
        "CONTROLLED-RADIUS VALIDATION COMPLETE"
    )

    print(
        "[PASS] Synthetic GSTA-compatible input accepted."
    )

    print(
        "[PASS] Original MSPA retains fixed per-head spatial ranges."
    )

    print(
        "[PASS] ARP-MSPA accepts target-agent-specific radius."
    )

    print(
        "[PASS] Spatial bias follows -d²/(2r²)."
    )

    print(
        "[PASS] Smaller radius produces more local attention."
    )

    print(
        "[PASS] Larger radius produces broader attention."
    )

    print(
        "[PASS] Changing radius changes attention distribution."
    )

    print(
        "[PASS] Both modules preserve the same output contract."
    )

    print()

    print(
        "RESEARCH INTERPRETATION:"
    )

    print(
        "  The controlled experiment validates the proposed"
    )

    print(
        "  adaptive spatial mechanism independently of training."
    )

    print()

    print(
        "  This is evidence that ARP-MSPA provides a genuinely"
    )

    print(
        "  different spatial-processing mechanism from fixed MSPA."
    )

    print()

    print(
        "  It is NOT yet evidence of lower ADE/FDE."
    )

    print(
        "  Prediction improvement must be demonstrated through"
    )

    print(
        "  controlled ablation and end-to-end Argoverse evaluation."
    )

    print()

    print(
        "Reference: DSTNet original MSPA + current ARP-MSPA implementation."
    )


if __name__ == "__main__":
    main()
