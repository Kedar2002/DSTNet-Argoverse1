"""
tests/test_arp_mspa_behavior.py

Behavioral validation for the proposed ARP-MSPA module.

Tests
-----
1. Radius prediction bounds
2. Radius predictor learning
3. Adaptive spatial-bias behavior
4. Attention redistribution for small vs. large radius
5. Full ARP-MSPA radius sensitivity
6. End-to-end gradient flow

The central ARP-MSPA equation is:

    B_ij = -d_ij^2 / (2 r_i^2)

and:

    A_ij =
        softmax_j(
            Q_i K_j^T / sqrt(d_k)
            + B_ij
        )

This test does NOT train the complete DSTNet model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

# ============================================================================
# Repository root
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.attention.arp_mspa import ARPMSPA


# ============================================================================
# Configuration
# ============================================================================

SEED: Final[int] = 42

DEVICE: Final[torch.device] = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

HIDDEN_DIM: Final[int] = 256
NUM_HEADS: Final[int] = 8
HISTORY_STEPS: Final[int] = 20
NUM_MODES: Final[int] = 6
NUM_AGENTS: Final[int] = 12

INTERACTION_RADIUS: Final[float] = 30.0

R_MIN: Final[float] = 5.0
R_MAX: Final[float] = 30.0

DROPOUT: Final[float] = 0.0


# ============================================================================
# Utility functions
# ============================================================================


def print_header(title: str) -> None:
    """Print a formatted test section header."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def assert_close(
    actual: Tensor,
    expected: Tensor,
    atol: float = 1e-5,
    message: str = "",
) -> None:
    """Assert two tensors are numerically close."""

    if not torch.allclose(
        actual,
        expected,
        atol=atol,
    ):
        max_error = (
            actual - expected
        ).abs().max().item()

        raise AssertionError(
            message
            + f"\nMaximum error: {max_error:.8f}"
        )


def create_model() -> ARPMSPA:
    """Create a deterministic ARP-MSPA instance."""

    torch.manual_seed(SEED)

    model = ARPMSPA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        r_min=R_MIN,
        r_max=R_MAX,
        radius_hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    return model.to(DEVICE)


# ============================================================================
# Synthetic scene creation
# ============================================================================


def create_structural_sample() -> tuple[
    Tensor,
    Tensor,
    Tensor,
]:
    """
    Create tensors with the same structural contract as DSTNet MSPA.

    Returns
    -------
    scene_embeddings:
        (B,N,H,K,D)

    positions:
        (B,N,2)

    agent_mask:
        (B,N)
    """

    torch.manual_seed(SEED)

    batch_size = 2

    scene_embeddings = torch.randn(
        batch_size,
        NUM_AGENTS,
        HISTORY_STEPS,
        NUM_MODES,
        HIDDEN_DIM,
        device=DEVICE,
    )

    ###########################################################################
    # Structured spatial arrangement.
    #
    # Distances from agent 0:
    #
    # 0, 3, 6, 10, 15, 20, 25, 29, 35, 45,
    # sqrt(16^2 + 5^2), sqrt(15^2 + 5^2)
    ###########################################################################

    positions = torch.tensor(
        [
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [6.0, 0.0],
                [10.0, 0.0],
                [15.0, 0.0],
                [20.0, 0.0],
                [25.0, 0.0],
                [29.0, 0.0],
                [35.0, 0.0],
                [45.0, 0.0],
                [16.0, 5.0],
                [15.0, 5.0],
            ]
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    positions = positions.repeat(
        batch_size,
        1,
        1,
    )

    agent_mask = torch.ones(
        batch_size,
        NUM_AGENTS,
        dtype=torch.bool,
        device=DEVICE,
    )

    return (
        scene_embeddings,
        positions,
        agent_mask,
    )


# ============================================================================
# TEST 1
# Radius prediction bounds
# ============================================================================


def test_radius_bounds() -> None:
    """
    Verify:

        r_min <= r_i <= r_max
    """

    print_header(
        "TEST 1: ADAPTIVE RADIUS BOUNDS"
    )

    model = create_model()

    scene_embeddings, _, _ = (
        create_structural_sample()
    )

    with torch.no_grad():

        agent_features: Tensor = (
            model._agent_features(
                scene_embeddings
            )
        )

        radii: Tensor = (
            model.predict_radius(
                agent_features
            )
        )

    min_radius = float(
        radii.min().item()
    )

    max_radius = float(
        radii.max().item()
    )

    print(
        f"Configured range : "
        f"[{R_MIN:.2f}, {R_MAX:.2f}] m"
    )

    print(
        f"Observed range   : "
        f"[{min_radius:.4f}, {max_radius:.4f}] m"
    )

    if min_radius < R_MIN - 1e-6:
        raise AssertionError(
            "Predicted radius fell below r_min."
        )

    if max_radius > R_MAX + 1e-6:
        raise AssertionError(
            "Predicted radius exceeded r_max."
        )

    print(
        "[PASS] All predicted radii remain bounded."
    )


# ============================================================================
# TEST 2
# Radius predictor learning
# ============================================================================


def test_radius_predictor_learning() -> None:
    """
    Verify that the radius predictor can learn different radii for
    different agent representations.

    Synthetic targets:

        Agent 0 -> 6 m
        Agent 1 -> 17 m
        Agent 2 -> 28 m
    """

    print_header(
        "TEST 2: RADIUS PREDICTOR LEARNING"
    )

    torch.manual_seed(SEED)

    model = create_model()

    ###########################################################################
    # Freeze the entire model first.
    ###########################################################################

    for parameter in model.parameters():
        parameter.requires_grad = False

    ###########################################################################
    # Enable gradients ONLY for radius predictor.
    ###########################################################################

    for parameter in (
        model.radius_predictor.parameters()
    ):
        parameter.requires_grad = True

    ###########################################################################
    # Synthetic agent features.
    ###########################################################################

    num_test_agents: Final[int] = 3

    agent_features = torch.zeros(
        1,
        num_test_agents,
        HIDDEN_DIM,
        device=DEVICE,
    )

    agent_features[0, 0, 0] = -3.0
    agent_features[0, 1, 0] = 0.0
    agent_features[0, 2, 0] = 3.0

    ###########################################################################
    # Target radii.
    ###########################################################################

    target_radii = torch.tensor(
        [[6.0, 17.0, 28.0]],
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print("Target radii:")

    for index in range(
        num_test_agents
    ):
        target_value = float(
            target_radii[
                0,
                index,
            ].item()
        )

        print(
            f"  Agent {index}: "
            f"{target_value:.2f} m"
        )

    ###########################################################################
    # IMPORTANT:
    #
    # Production ARP-MSPA initializes the final radius layer to zero so that
    # all agents initially start at the midpoint:
    #
    #     (r_min + r_max) / 2
    #
    # For this isolated learning test, we give the final layer a tiny random
    # initialization so that we explicitly test feature-dependent learning.
    #
    # This does NOT modify arp_mspa.py.
    ###########################################################################

    final_layer = model.radius_predictor[-1]

    if not isinstance(
        final_layer,
        torch.nn.Linear,
    ):
        raise TypeError(
            "Expected final radius predictor "
            "layer to be nn.Linear."
        )

    with torch.no_grad():

        final_layer.weight.normal_(
            mean=0.0,
            std=0.01,
        )

        final_layer.bias.zero_()

    ###########################################################################
    # Optimizer.
    ###########################################################################

    optimizer = torch.optim.Adam(
        model.radius_predictor.parameters(),
        lr=0.01,
    )

    ###########################################################################
    # Initial prediction.
    ###########################################################################

    with torch.no_grad():

        initial_radii: Tensor = (
            model.predict_radius(
                agent_features
            )
        )

    print()
    print("Initial predicted radii:")

    for index in range(
        num_test_agents
    ):

        predicted_value = float(
            initial_radii[
                0,
                index,
            ].item()
        )

        print(
            f"  Agent {index}: "
            f"{predicted_value:.4f} m"
        )

    ###########################################################################
    # Explicitly typed scalar.
    #
    # This fixes the Pylance warning caused by:
    #
    #     initial_loss = None
    #
    # followed by:
    #
    #     final_loss < initial_loss
    ###########################################################################

    initial_loss: float = 0.0

    num_steps: Final[int] = 800

    ###########################################################################
    # Optimization loop.
    ###########################################################################

    for step in range(
        num_steps
    ):

        optimizer.zero_grad()

        predicted_radii: Tensor = (
            model.predict_radius(
                agent_features
            )
        )

        loss: Tensor = torch.mean(
            (
                predicted_radii
                - target_radii
            ).square()
        )

        if step == 0:
            initial_loss = float(
                loss.detach().item()
            )

        loss.backward()

        optimizer.step()

    ###########################################################################
    # Final prediction.
    ###########################################################################

    with torch.no_grad():

        final_radii: Tensor = (
            model.predict_radius(
                agent_features
            )
        )

        final_loss_tensor: Tensor = (
            torch.mean(
                (
                    final_radii
                    - target_radii
                ).square()
            )
        )

        final_loss: float = float(
            final_loss_tensor.item()
        )

    print()
    print(
        f"Initial MSE : "
        f"{initial_loss:.6f}"
    )

    print(
        f"Final MSE   : "
        f"{final_loss:.6f}"
    )

    print()
    print("Learned radii:")

    for index in range(
        num_test_agents
    ):

        target_value = float(
            target_radii[
                0,
                index,
            ].item()
        )

        predicted_value = float(
            final_radii[
                0,
                index,
            ].item()
        )

        print(
            f"  Agent {index}: "
            f"target={target_value:.2f} m, "
            f"predicted={predicted_value:.4f} m"
        )

    ###########################################################################
    # Verify learning.
    ###########################################################################

    if not (
        final_loss < initial_loss
    ):
        raise AssertionError(
            "Radius predictor loss did not decrease."
        )

    ###########################################################################
    # Maximum fitting error.
    ###########################################################################

    max_error = float(
        (
            final_radii
            - target_radii
        ).abs().max().item()
    )

    if max_error > 1.0:
        raise AssertionError(
            "Radius predictor failed to learn "
            "the synthetic target radii accurately. "
            f"Maximum error = {max_error:.4f} m"
        )

    ###########################################################################
    # Check agent dependence.
    ###########################################################################

    radius_range = float(
        (
            final_radii.max()
            - final_radii.min()
        ).item()
    )

    if radius_range < 5.0:
        raise AssertionError(
            "Learned radii are not sufficiently "
            "agent-dependent."
        )

    print()
    print(
        f"Learned radius spread: "
        f"{radius_range:.4f} m"
    )

    print(
        "[PASS] Radius predictor learned "
        "agent-dependent spatial ranges."
    )


# ============================================================================
# TEST 3
# Analytical adaptive spatial bias
# ============================================================================


def test_spatial_bias_behavior() -> None:
    """
    Verify:

        B_ij = -d_ij^2 / (2 r_i^2)

    Smaller r_i produces a more negative bias.
    """

    print_header(
        "TEST 3: ADAPTIVE SPATIAL BIAS"
    )

    model = create_model()

    distances = torch.tensor(
        [
            0.0,
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    positions = torch.zeros(
        1,
        len(distances),
        2,
        device=DEVICE,
    )

    positions[0, :, 0] = distances

    radius_values = [
        5.0,
        10.0,
        20.0,
        30.0,
    ]

    print()
    print(
        "Spatial bias:"
    )

    header = (
        f"{'Distance':>10}"
        f"{'r=5m':>12}"
        f"{'r=10m':>12}"
        f"{'r=20m':>12}"
        f"{'r=30m':>12}"
    )

    print(header)
    print("-" * len(header))

    bias_rows: list[list[float]] = []

    for distance_tensor in distances:

        distance = float(
            distance_tensor.item()
        )

        row = [distance]

        for radius in radius_values:

            bias_value = -(
                distance**2
            ) / (
                2.0
                * radius**2
            )

            row.append(
                bias_value
            )

        bias_rows.append(row)

        print(
            f"{row[0]:10.2f}"
            f"{row[1]:12.4f}"
            f"{row[2]:12.4f}"
            f"{row[3]:12.4f}"
            f"{row[4]:12.4f}"
        )

    ###########################################################################
    # Smaller radius must create a more negative bias.
    ###########################################################################

    for row in bias_rows:

        distance = row[0]

        if distance == 0.0:
            continue

        bias_small = row[1]
        bias_large = row[4]

        if not (
            bias_small < bias_large
        ):
            raise AssertionError(
                "Smaller radius should produce "
                "a more negative spatial bias."
            )

    ###########################################################################
    # Verify module implementation against formula.
    ###########################################################################

    radii = torch.full(
        (
            1,
            len(distances),
        ),
        20.0,
        device=DEVICE,
    )

    bias: Tensor = (
        model._adaptive_spatial_bias(
            positions,
            radii,
        )
    )

    expected_target: Tensor = (
        -distances.square()
        / (
            2.0
            * 20.0**2
        )
    )

    actual_target: Tensor = (
        bias[0, 0]
    )

    assert_close(
        actual_target,
        expected_target,
        atol=1e-5,
        message=(
            "Module spatial bias does not match "
            "the analytical formula."
        ),
    )

    print()
    print(
        "[PASS] Spatial bias follows "
        "B_ij = -d_ij² / (2r_i²)."
    )


# ============================================================================
# Spatial-only attention helper
# ============================================================================


def compute_spatial_only_attention(
    distances: Tensor,
    radius: float,
) -> Tensor:
    """
    Compute attention using ONLY adaptive spatial bias.

    QK term is intentionally zero.

        A_ij =
            softmax(
                -d_ij² / (2r_i²)
            )
    """

    spatial_bias: Tensor = -(
        distances.square()
        / (
            2.0
            * radius**2
        )
    )

    attention: Tensor = torch.softmax(
        spatial_bias,
        dim=0,
    )

    return attention


# ============================================================================
# TEST 4
# Attention redistribution
# ============================================================================


def test_attention_redistribution() -> None:
    """
    Demonstrate:

        smaller r_i -> local attention

        larger r_i -> broader attention
    """

    print_header(
        "TEST 4: ATTENTION REDISTRIBUTION"
    )

    distances = torch.tensor(
        [
            0.0,
            3.0,
            6.0,
            10.0,
            15.0,
            20.0,
            25.0,
            29.0,
            35.0,
            45.0,
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    ###########################################################################
    # r_max defines the hard candidate boundary.
    ###########################################################################

    candidate_mask = (
        distances <= R_MAX
    )

    candidate_mask[0] = True

    candidate_distances: Tensor = (
        distances[candidate_mask]
    )

    print()
    print(
        "Candidate distances within "
        f"r_max={R_MAX:.1f}m:"
    )

    for index in range(
        candidate_distances.shape[0]
    ):

        distance = float(
            candidate_distances[index].item()
        )

        print(
            f"  Candidate {index:02d}: "
            f"{distance:.2f} m"
        )

    radii_to_test = [
        5.0,
        10.0,
        20.0,
        30.0,
    ]

    results: dict[
        float,
        Tensor,
    ] = {}

    print()
    print(
        "Spatial-only attention distribution:"
    )

    for radius in radii_to_test:

        attention: Tensor = (
            compute_spatial_only_attention(
                candidate_distances,
                radius,
            )
        )

        results[radius] = attention

        expected_distance = float(
            (
                attention
                * candidate_distances
            ).sum().item()
        )

        entropy_tensor: Tensor = -(
            attention
            * torch.log(
                attention.clamp_min(
                    1e-12
                )
            )
        ).sum()

        entropy = float(
            entropy_tensor.item()
        )

        print()
        print(
            f"r = {radius:5.1f} m"
        )

        print(
            f"  Expected distance : "
            f"{expected_distance:.4f} m"
        )

        print(
            f"  Attention entropy : "
            f"{entropy:.4f}"
        )

        for index in range(
            candidate_distances.shape[0]
        ):

            distance = float(
                candidate_distances[
                    index
                ].item()
            )

            weight = float(
                attention[index].item()
            )

            print(
                f"    d={distance:5.1f} m"
                f" -> A={weight:.6f}"
            )

    ###########################################################################
    # Calculate expected distances and entropy.
    ###########################################################################

    expected_distances: list[float] = []
    entropies: list[float] = []

    for radius in radii_to_test:

        attention = results[radius]

        expected_distance = float(
            (
                attention
                * candidate_distances
            ).sum().item()
        )

        entropy_tensor = -(
            attention
            * torch.log(
                attention.clamp_min(
                    1e-12
                )
            )
        ).sum()

        entropy = float(
            entropy_tensor.item()
        )

        expected_distances.append(
            expected_distance
        )

        entropies.append(
            entropy
        )

    ###########################################################################
    # Larger radius should produce broader interaction.
    ###########################################################################

    for previous, current in zip(
        expected_distances,
        expected_distances[1:],
    ):

        if current <= previous:
            raise AssertionError(
                "Expected interaction distance "
                "did not increase with radius."
            )

    ###########################################################################
    # Entropy should increase as attention becomes broader.
    ###########################################################################

    for previous, current in zip(
        entropies,
        entropies[1:],
    ):

        if current <= previous:
            raise AssertionError(
                "Attention entropy did not increase "
                "with radius."
            )

    ###########################################################################
    # Compare 3m and 20m attention.
    ###########################################################################

    index_3m = int(
        (
            candidate_distances == 3.0
        ).nonzero(
            as_tuple=True
        )[0][0].item()
    )

    index_20m = int(
        (
            candidate_distances == 20.0
        ).nonzero(
            as_tuple=True
        )[0][0].item()
    )

    attention_small: Tensor = (
        results[5.0]
    )

    ratio_small_radius = float(
        (
            attention_small[index_3m]
            / attention_small[index_20m]
        ).item()
    )

    print()
    print(
        "Small-radius locality check:"
    )

    print(
        f"  A(3m) / A(20m) at r=5m "
        f"= {ratio_small_radius:.4f}"
    )

    if ratio_small_radius <= 10.0:
        raise AssertionError(
            "Small radius did not sufficiently "
            "favor nearby agents."
        )

    ###########################################################################
    # Large radius.
    ###########################################################################

    attention_large: Tensor = (
        results[30.0]
    )

    ratio_large_radius = float(
        (
            attention_large[index_3m]
            / attention_large[index_20m]
        ).item()
    )

    print()
    print(
        "Large-radius locality check:"
    )

    print(
        f"  A(3m) / A(20m) at r=30m "
        f"= {ratio_large_radius:.4f}"
    )

    if ratio_large_radius >= ratio_small_radius:
        raise AssertionError(
            "Large radius should reduce the "
            "relative suppression of distant agents."
        )

    print()
    print(
        "[PASS] Attention becomes broader as "
        "the adaptive radius increases."
    )


# ============================================================================
# TEST 5
# Full module radius sensitivity
# ============================================================================


def test_full_module_radius_sensitivity() -> None:
    """
    Verify that changing r_i changes the actual ARP-MSPA output.
    """

    print_header(
        "TEST 5: FULL ARP-MSPA RADIUS SENSITIVITY"
    )

    model = create_model()

    model.eval()

    (
        scene_embeddings,
        positions,
        agent_mask,
    ) = create_structural_sample()

    with torch.no_grad():

        output_normal: Tensor = model(
            scene_embeddings,
            positions,
            agent_mask=agent_mask,
        )

    del output_normal

    ###########################################################################
    # Radius configurations.
    ###########################################################################

    batch_size = (
        scene_embeddings.shape[0]
    )

    num_agents = (
        scene_embeddings.shape[1]
    )

    small_radii = torch.full(
        (
            batch_size,
            num_agents,
        ),
        5.0,
        device=DEVICE,
    )

    large_radii = torch.full(
        (
            batch_size,
            num_agents,
        ),
        30.0,
        device=DEVICE,
    )

    ###########################################################################
    # Q / K / V.
    ###########################################################################

    Q: Tensor = model.query_projection(
        scene_embeddings
    )

    K: Tensor = model.key_projection(
        scene_embeddings
    )

    V: Tensor = model.value_projection(
        scene_embeddings
    )

    (
        batch_size,
        num_agents,
        history,
        modes,
        _,
    ) = scene_embeddings.shape

    Q = Q.reshape(
        batch_size,
        num_agents,
        history,
        modes,
        NUM_HEADS,
        model.head_dim,
    ).permute(
        0,
        4,
        1,
        2,
        3,
        5,
    )

    K = K.reshape(
        batch_size,
        num_agents,
        history,
        modes,
        NUM_HEADS,
        model.head_dim,
    ).permute(
        0,
        4,
        1,
        2,
        3,
        5,
    )

    V = V.reshape(
        batch_size,
        num_agents,
        history,
        modes,
        NUM_HEADS,
        model.head_dim,
    ).permute(
        0,
        4,
        1,
        2,
        3,
        5,
    )

    candidate_mask: Tensor = (
        model._build_candidate_mask(
            positions,
            agent_mask,
        )
    )

    ###########################################################################
    # Helper.
    ###########################################################################

    def compute_output(
        radii: Tensor,
    ) -> Tensor:

        spatial_bias: Tensor = (
            model._adaptive_spatial_bias(
                positions,
                radii,
            )
        )

        head_outputs: list[Tensor] = []

        for head_index in range(
            NUM_HEADS
        ):

            head_output: Tensor = (
                model._single_head_attention(
                    query=Q[:, head_index],
                    key=K[:, head_index],
                    value=V[:, head_index],
                    candidate_mask=candidate_mask,
                    spatial_bias=spatial_bias,
                )
            )

            head_output = (
                model.alpha[head_index]
                * head_output
            )

            head_outputs.append(
                head_output
            )

        result: Tensor = torch.cat(
            head_outputs,
            dim=-1,
        )

        result = model.output_projection(
            result
        )

        return result

    ###########################################################################
    # Compute both representations.
    ###########################################################################

    with torch.no_grad():

        output_small: Tensor = (
            compute_output(
                small_radii
            )
        )

        output_large: Tensor = (
            compute_output(
                large_radii
            )
        )

    difference: Tensor = (
        output_small
        - output_large
    )

    mean_absolute_difference = float(
        difference.abs()
        .mean()
        .item()
    )

    maximum_difference = float(
        difference.abs()
        .max()
        .item()
    )

    print()
    print(
        f"Mean absolute difference : "
        f"{mean_absolute_difference:.8f}"
    )

    print(
        f"Maximum absolute difference : "
        f"{maximum_difference:.8f}"
    )

    if mean_absolute_difference <= 1e-6:
        raise AssertionError(
            "Changing radius did not change "
            "the ARP-MSPA representation."
        )

    print()
    print(
        "[PASS] Adaptive radius changes "
        "the actual ARP-MSPA representation."
    )


# ============================================================================
# TEST 6
# End-to-end gradient
# ============================================================================


def test_end_to_end_gradient() -> None:
    """
    Verify gradient flow through:

        radius predictor
             ↓
            r_i
             ↓
       spatial bias
             ↓
         attention
             ↓
          output
             ↓
           loss
    """

    print_header(
        "TEST 6: END-TO-END GRADIENT THROUGH ADAPTIVE RADIUS"
    )

    model = create_model()

    model.train()

    (
        scene_embeddings,
        positions,
        agent_mask,
    ) = create_structural_sample()

    scene_embeddings.requires_grad_(True)

    output: Tensor = model(
        scene_embeddings,
        positions,
        agent_mask=agent_mask,
    )

    loss: Tensor = (
        output.square().mean()
    )

    loss.backward()

    ###########################################################################
    # Radius predictor gradients.
    ###########################################################################

    total_radius_gradient = 0.0

    for (
        name,
        parameter,
    ) in model.radius_predictor.named_parameters():

        if parameter.grad is None:
            raise AssertionError(
                "No gradient received by "
                f"radius predictor parameter: {name}"
            )

        gradient_norm = float(
            parameter.grad.abs()
            .sum()
            .item()
        )

        total_radius_gradient += (
            gradient_norm
        )

        print(
            f"  {name:<20} "
            f"gradient L1 = "
            f"{gradient_norm:.8f}"
        )

    print()
    print(
        "Total radius-predictor "
        f"gradient L1: "
        f"{total_radius_gradient:.8f}"
    )

    if total_radius_gradient <= 0.0:
        raise AssertionError(
            "No gradient propagated through "
            "the adaptive radius predictor."
        )

    print(
        "[PASS] Gradient flows through "
        "the adaptive radius mechanism."
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:

    print("=" * 80)
    print(
        "ARP-MSPA BEHAVIORAL VALIDATION"
    )
    print("=" * 80)

    print(
        f"Device         : {DEVICE}"
    )

    print(
        f"Hidden dim     : {HIDDEN_DIM}"
    )

    print(
        f"Attention heads: {NUM_HEADS}"
    )

    print(
        f"History        : {HISTORY_STEPS}"
    )

    print(
        f"Modes          : {NUM_MODES}"
    )

    print(
        f"Agents         : {NUM_AGENTS}"
    )

    print(
        f"Interaction R  : "
        f"{INTERACTION_RADIUS:.1f} m"
    )

    print(
        f"ARP range      : "
        f"[{R_MIN:.1f}, {R_MAX:.1f}] m"
    )

    # ========================================================================
    # Run all tests
    # ========================================================================

    test_radius_bounds()

    test_radius_predictor_learning()

    test_spatial_bias_behavior()

    test_attention_redistribution()

    test_full_module_radius_sensitivity()

    test_end_to_end_gradient()

    # ========================================================================
    # Final status
    # ========================================================================

    print()
    print("=" * 80)
    print(
        "ALL ARP-MSPA BEHAVIORAL TESTS PASSED"
    )
    print("=" * 80)

    print()
    print("Validated:")

    print(
        "  [PASS] Radius is bounded"
    )

    print(
        "  [PASS] Radius predictor can learn "
        "agent-dependent ranges"
    )

    print(
        "  [PASS] Spatial bias follows "
        "-d²/(2r²)"
    )

    print(
        "  [PASS] Small radius produces "
        "more local attention"
    )

    print(
        "  [PASS] Large radius produces "
        "broader attention"
    )

    print(
        "  [PASS] Changing radius changes "
        "the full ARP-MSPA representation"
    )

    print(
        "  [PASS] End-to-end gradients reach "
        "the radius predictor"
    )

    print()
    print(
        "ARP-MSPA core behavioral mechanism "
        "is validated."
    )


if __name__ == "__main__":
    main()
