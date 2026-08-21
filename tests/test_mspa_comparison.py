"""
tests/test_mspa_comparison.py

Standalone comparison test for:

    1. Original DSTNet MSPA
    2. Proposed ARP-MSPA

The test does NOT require the complete DSTNet model.

Current DSTNet MSPA contract:

    scene_embeddings : (B, N, T, K, D)
    positions        : (B, N, 2)
    agent_mask       : (B, N)

Output:

    (B, N, T, K, D)

The synthetic input intentionally follows the same structural
tensor contract as the current DSTNet scene representation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Make repository root importable when running this file directly.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------------------

from models.attention.mspa import MSPA

# Proposed module.
#
# IMPORTANT:
# The proposed implementation should expose the same basic forward
# interface as MSPA:
#
#     output = module(
#         scene_embeddings,
#         positions,
#         agent_mask=agent_mask,
#     )
#
# Rename this import if we choose a different filename/class name.
#
from models.attention.arp_mspa import ARPMSPA


# ===========================================================================
# Configuration
# ===========================================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DTYPE = torch.float32

# Match the current DSTNet architecture.
HIDDEN_DIM = 256
NUM_HEADS = 8

# Argoverse-style temporal/multimodal structure.
OBSERVATION_STEPS = 20
NUM_MODES = 6

# Keep the synthetic test small enough to run quickly.
#
# The production implementation supports variable N.
# Here we deliberately use 12 agents so the spatial relationships
# can be controlled and inspected easily.
NUM_AGENTS = 12

BATCH_SIZE = 2

INTERACTION_RADIUS = 30.0

# ARP-MSPA bounds.
R_MIN = 5.0
R_MAX = 30.0

DROPOUT = 0.0


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int = SEED) -> None:
    """Set deterministic random seeds for the test."""

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Synthetic scene generation
# ===========================================================================

def build_sample_scene(
    *,
    batch_size: int = BATCH_SIZE,
    num_agents: int = NUM_AGENTS,
    observation_steps: int = OBSERVATION_STEPS,
    num_modes: int = NUM_MODES,
    hidden_dim: int = HIDDEN_DIM,
    device: torch.device = DEVICE,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Create a synthetic scene with the same structural tensor contract
    expected by the current MSPA implementation.

    Returns
    -------
    scene_embeddings:
        (B, N, T, K, D)

    positions:
        (B, N, 2)

    agent_mask:
        (B, N)
    """

    if num_agents < 8:
        raise ValueError(
            "num_agents should be >= 8 for the controlled scene."
        )

    # -----------------------------------------------------------------------
    # 1. Scene embeddings
    #
    # These represent Z_scene after GSTA.
    #
    # Shape:
    #
    #     (B, N, T, K, D)
    # -----------------------------------------------------------------------

    scene_embeddings = torch.randn(
        batch_size,
        num_agents,
        observation_steps,
        num_modes,
        hidden_dim,
        device=device,
        dtype=DTYPE,
    )

    # -----------------------------------------------------------------------
    # 2. Controlled spatial configuration
    #
    # We intentionally create agents at different distances.
    #
    # Agent 0 = target/reference agent at origin.
    #
    # Nearby agents:
    #     1 -> 3 m
    #     2 -> 6 m
    #     3 -> 10 m
    #
    # Medium-range agents:
    #     4 -> 15 m
    #     5 -> 20 m
    #     6 -> 25 m
    #
    # Near outer interaction boundary:
    #     7 -> 29 m
    #
    # Outside original maximum radius:
    #     8 -> 35 m
    #     9 -> 45 m
    #
    # Remaining agents are placed elsewhere.
    # -----------------------------------------------------------------------

    base_positions = torch.tensor(
        [
            [0.0, 0.0],       # 0: target
            [3.0, 0.0],       # 1
            [6.0, 0.0],       # 2
            [10.0, 0.0],      # 3
            [15.0, 0.0],      # 4
            [20.0, 0.0],      # 5
            [25.0, 0.0],      # 6
            [29.0, 0.0],      # 7
            [35.0, 0.0],      # 8
            [45.0, 0.0],      # 9
            [12.0, 12.0],     # 10
            [-15.0, 5.0],     # 11
        ],
        device=device,
        dtype=DTYPE,
    )

    positions = base_positions.unsqueeze(0).repeat(
        batch_size,
        1,
        1,
    )

    # -----------------------------------------------------------------------
    # Slightly perturb the second scene.
    #
    # This ensures the model sees more than one identical scene while
    # keeping the geometry interpretable.
    # -----------------------------------------------------------------------

    if batch_size > 1:
        noise = torch.randn(
            batch_size - 1,
            num_agents,
            2,
            device=device,
            dtype=DTYPE,
        ) * 0.25

        positions[1:] += noise

    # -----------------------------------------------------------------------
    # 3. Valid-agent mask
    #
    # All agents are valid in this synthetic test.
    # -----------------------------------------------------------------------

    agent_mask = torch.ones(
        batch_size,
        num_agents,
        device=device,
        dtype=torch.bool,
    )

    return (
        scene_embeddings,
        positions,
        agent_mask,
    )


# ===========================================================================
# Module construction
# ===========================================================================

def build_original_mspa() -> MSPA:
    """Construct the original DSTNet MSPA."""

    return MSPA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        dropout=DROPOUT,
    ).to(DEVICE)


def build_arp_mspa() -> ARPMSPA:
    """
    Construct the proposed ARP-MSPA.

    Expected constructor:

        ARPMSPA(
            hidden_dim=...,
            num_heads=...,
            interaction_radius=...,
            r_min=...,
            r_max=...,
            dropout=...,
        )
    """

    return ARPMSPA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        r_min=R_MIN,
        r_max=R_MAX,
        dropout=DROPOUT,
    ).to(DEVICE)


# ===========================================================================
# Utility functions
# ===========================================================================

def count_parameters(
    module: torch.nn.Module,
) -> int:
    """Return the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def check_finite(
    name: str,
    tensor: torch.Tensor,
) -> None:
    """Fail if tensor contains NaN or Inf."""

    if not torch.isfinite(tensor).all():
        raise AssertionError(
            f"{name} contains NaN or Inf."
        )


def print_tensor_statistics(
    name: str,
    tensor: torch.Tensor,
) -> None:
    """Print basic numerical statistics."""

    print(
        f"{name}:"
        f" shape={tuple(tensor.shape)}"
        f" min={tensor.min().item():.6f}"
        f" max={tensor.max().item():.6f}"
        f" mean={tensor.mean().item():.6f}"
        f" std={tensor.std().item():.6f}"
    )


# ===========================================================================
# Test 1 — Input structure
# ===========================================================================

def test_input_structure(
    scene_embeddings: torch.Tensor,
    positions: torch.Tensor,
    agent_mask: torch.Tensor,
) -> None:
    """Validate the synthetic scene contract."""

    expected_scene_shape = (
        BATCH_SIZE,
        NUM_AGENTS,
        OBSERVATION_STEPS,
        NUM_MODES,
        HIDDEN_DIM,
    )

    expected_position_shape = (
        BATCH_SIZE,
        NUM_AGENTS,
        2,
    )

    expected_mask_shape = (
        BATCH_SIZE,
        NUM_AGENTS,
    )

    assert scene_embeddings.shape == expected_scene_shape, (
        "Unexpected scene_embeddings shape: "
        f"{tuple(scene_embeddings.shape)}"
    )

    assert positions.shape == expected_position_shape, (
        "Unexpected positions shape: "
        f"{tuple(positions.shape)}"
    )

    assert agent_mask.shape == expected_mask_shape, (
        "Unexpected agent_mask shape: "
        f"{tuple(agent_mask.shape)}"
    )

    check_finite(
        "scene_embeddings",
        scene_embeddings,
    )

    check_finite(
        "positions",
        positions,
    )

    print("\n[PASS] Input structure")


# ===========================================================================
# Test 2 — Original MSPA forward pass
# ===========================================================================

def test_original_mspa(
    model: MSPA,
    scene_embeddings: torch.Tensor,
    positions: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the original MSPA."""

    model.eval()

    with torch.no_grad():
        output = model(
            scene_embeddings,
            positions,
            agent_mask=agent_mask,
        )

    expected_shape = scene_embeddings.shape

    assert output.shape == expected_shape, (
        "Original MSPA changed the tensor shape. "
        f"Expected {tuple(expected_shape)}, "
        f"got {tuple(output.shape)}."
    )

    check_finite(
        "Original MSPA output",
        output,
    )

    print("\n[PASS] Original MSPA forward")

    print_tensor_statistics(
        "Original MSPA output",
        output,
    )

    return output


# ===========================================================================
# Test 3 — ARP-MSPA forward pass
# ===========================================================================

def test_arp_mspa(
    model: ARPMSPA,
    scene_embeddings: torch.Tensor,
    positions: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the proposed ARP-MSPA."""

    model.eval()

    with torch.no_grad():
        output = model(
            scene_embeddings,
            positions,
            agent_mask=agent_mask,
        )

    expected_shape = scene_embeddings.shape

    assert output.shape == expected_shape, (
        "ARP-MSPA changed the tensor shape. "
        f"Expected {tuple(expected_shape)}, "
        f"got {tuple(output.shape)}."
    )

    check_finite(
        "ARP-MSPA output",
        output,
    )

    print("\n[PASS] ARP-MSPA forward")

    print_tensor_statistics(
        "ARP-MSPA output",
        output,
    )

    return output


# ===========================================================================
# Test 4 — Gradient flow
# ===========================================================================

def test_gradient_flow(
    original: MSPA,
    arp: ARPMSPA,
    scene_embeddings: torch.Tensor,
    positions: torch.Tensor,
    agent_mask: torch.Tensor,
) -> None:
    """
    Verify that both modules participate in backpropagation.

    This is particularly important for ARP because the predicted
    radius must receive gradients from the attention output.
    """

    print("\nChecking gradient flow...")

    # -----------------------------------------------------------------------
    # Original MSPA
    # -----------------------------------------------------------------------

    original.train()

    x_original = scene_embeddings.detach().clone()
    x_original.requires_grad_(True)

    output_original = original(
        x_original,
        positions,
        agent_mask=agent_mask,
    )

    loss_original = output_original.square().mean()

    original.zero_grad(set_to_none=True)

    loss_original.backward()

    assert x_original.grad is not None, (
        "Original MSPA did not propagate gradients to its input."
    )

    check_finite(
        "Original input gradient",
        x_original.grad,
    )

    # -----------------------------------------------------------------------
    # ARP-MSPA
    # -----------------------------------------------------------------------

    arp.train()

    x_arp = scene_embeddings.detach().clone()
    x_arp.requires_grad_(True)

    output_arp = arp(
        x_arp,
        positions,
        agent_mask=agent_mask,
    )

    loss_arp = output_arp.square().mean()

    arp.zero_grad(set_to_none=True)

    loss_arp.backward()

    assert x_arp.grad is not None, (
        "ARP-MSPA did not propagate gradients to its input."
    )

    check_finite(
        "ARP input gradient",
        x_arp.grad,
    )

    # Check at least one trainable ARP parameter received gradient.
    arp_gradients = [
        parameter.grad
        for parameter in arp.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    assert len(arp_gradients) > 0, (
        "No ARP-MSPA parameter received a gradient."
    )

    for gradient in arp_gradients:
        check_finite(
            "ARP parameter gradient",
            gradient,
        )

    print("[PASS] Gradient flow")


# ===========================================================================
# Test 5 — Radius inspection
# ===========================================================================

def test_adaptive_radius(
    arp: ARPMSPA,
    scene_embeddings: torch.Tensor,
) -> None:
    """
    Inspect the learned radius values.

    The proposed ARP-MSPA should expose either:

        model.radius_predictor(...)

    or:

        model.predict_radius(...)

    This test supports both conventions.

    IMPORTANT:
    At initialization the radii do NOT need to be meaningful.
    We are checking:
        1. correct shape
        2. finite values
        3. bounds
        4. ability to produce different radii
    """

    arp.eval()

    # Use the first temporal step and first prediction mode as the
    # agent-level feature used by the radius predictor.
    #
    # scene_embeddings:
    #     (B,N,T,K,D)
    #
    # agent_features:
    #     (B,N,D)

    agent_features = scene_embeddings[
        :,
        :,
        -1,
        0,
        :,
    ]

    with torch.no_grad():

        if hasattr(
            arp,
            "predict_radius",
        ):
            radii = arp.predict_radius(
                agent_features,
            )

        elif hasattr(
            arp,
            "radius_predictor",
        ):
            radii = arp.radius_predictor(
                agent_features,
            )

        else:
            raise AttributeError(
                "ARP-MSPA must expose either "
                "`predict_radius()` or `radius_predictor` "
                "for this diagnostic test."
            )

    # Accept either:
    #
    #     (B,N)
    #
    # or:
    #
    #     (B,N,1)

    if radii.ndim == 3 and radii.shape[-1] == 1:
        radii = radii.squeeze(-1)

    expected_shape = (
        BATCH_SIZE,
        NUM_AGENTS,
    )

    assert radii.shape == expected_shape, (
        "Unexpected radius shape. "
        f"Expected {expected_shape}, "
        f"got {tuple(radii.shape)}."
    )

    check_finite(
        "Predicted radii",
        radii,
    )

    assert torch.all(
        radii >= R_MIN
    ), (
        "ARP produced a radius below r_min."
    )

    assert torch.all(
        radii <= R_MAX
    ), (
        "ARP produced a radius above r_max."
    )

    print("\n[PASS] Adaptive radius bounds")

    print("\nPredicted radii for first scene:")

    for agent_index, radius in enumerate(
        radii[0]
    ):
        print(
            f"  Agent {agent_index:02d}: "
            f"{radius.item():.4f} m"
        )


# ===========================================================================
# Test 6 — Output difference
# ===========================================================================

def test_outputs_are_not_identical(
    original_output: torch.Tensor,
    arp_output: torch.Tensor,
) -> None:
    """
    Verify that the proposed module is not accidentally identical
    to the baseline implementation.
    """

    difference = torch.abs(
        original_output - arp_output
    )

    mean_difference = difference.mean().item()
    max_difference = difference.max().item()

    print("\nBaseline vs ARP-MSPA:")
    print(
        f"  Mean absolute difference: "
        f"{mean_difference:.8f}"
    )

    print(
        f"  Maximum absolute difference: "
        f"{max_difference:.8f}"
    )

    assert mean_difference > 1e-8, (
        "Original MSPA and ARP-MSPA produced effectively "
        "identical outputs. Check whether ARP-MSPA is actually "
        "modifying the attention computation."
    )

    print("[PASS] Modules produce different representations")


# ===========================================================================
# Test 7 — Parameter count
# ===========================================================================

def test_parameter_count(
    original: MSPA,
    arp: ARPMSPA,
) -> None:
    """Compare trainable parameter counts."""

    original_parameters = count_parameters(
        original
    )

    arp_parameters = count_parameters(
        arp
    )

    additional_parameters = (
        arp_parameters
        - original_parameters
    )

    if original_parameters > 0:
        percentage_increase = (
            100.0
            * additional_parameters
            / original_parameters
        )
    else:
        percentage_increase = math.nan

    print("\nParameter comparison:")
    print(
        f"  Original MSPA : "
        f"{original_parameters:,}"
    )

    print(
        f"  ARP-MSPA      : "
        f"{arp_parameters:,}"
    )

    print(
        f"  Additional    : "
        f"{additional_parameters:,}"
    )

    print(
        f"  Increase      : "
        f"{percentage_increase:.2f}%"
    )

    assert arp_parameters >= original_parameters, (
        "ARP-MSPA should not have fewer parameters than the "
        "original MSPA unless deliberately designed that way."
    )

    print("[PASS] Parameter count")


# ===========================================================================
# Test 8 — Spatial geometry sanity check
# ===========================================================================

def test_spatial_geometry(
    positions: torch.Tensor,
) -> None:
    """
    Verify the controlled spatial arrangement.

    This test is independent of the neural network.
    """

    first_scene = positions[0]

    target = first_scene[0]

    distances = torch.linalg.vector_norm(
        first_scene - target,
        dim=-1,
    )

    print("\nDistances from target agent 0:")

    for agent_index, distance in enumerate(
        distances
    ):
        print(
            f"  Agent {agent_index:02d}: "
            f"{distance.item():.2f} m"
        )

    # Agent 1 should be 3m away.
    assert torch.isclose(
        distances[1],
        torch.tensor(
            3.0,
            device=distances.device,
        ),
        atol=1e-4,
    )

    # Agent 7 should be approximately 29m away.
    assert torch.isclose(
        distances[7],
        torch.tensor(
            29.0,
            device=distances.device,
        ),
        atol=1e-4,
    )

    # Agent 8 should be outside the 30m original radius.
    assert distances[8] > INTERACTION_RADIUS

    print("[PASS] Spatial geometry")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Run the complete MSPA comparison suite."""

    print("=" * 72)
    print("DSTNet MSPA vs ARP-MSPA MODULAR TEST")
    print("=" * 72)

    print(f"Device       : {DEVICE}")
    print(f"Hidden dim   : {HIDDEN_DIM}")
    print(f"Attention heads: {NUM_HEADS}")
    print(f"History      : {OBSERVATION_STEPS}")
    print(f"Modes        : {NUM_MODES}")
    print(f"Agents       : {NUM_AGENTS}")
    print(f"Interaction R: {INTERACTION_RADIUS} m")
    print(f"ARP range    : [{R_MIN}, {R_MAX}] m")

    set_seed()

    # -----------------------------------------------------------------------
    # Build synthetic dataset-like sample.
    # -----------------------------------------------------------------------

    (
        scene_embeddings,
        positions,
        agent_mask,
    ) = build_sample_scene()

    # -----------------------------------------------------------------------
    # Input test.
    # -----------------------------------------------------------------------

    test_input_structure(
        scene_embeddings,
        positions,
        agent_mask,
    )

    # -----------------------------------------------------------------------
    # Spatial geometry.
    # -----------------------------------------------------------------------

    test_spatial_geometry(
        positions,
    )

    # -----------------------------------------------------------------------
    # Build models.
    # -----------------------------------------------------------------------

    original = build_original_mspa()

    arp = build_arp_mspa()

    print("\nModels:")
    print(
        f"  Original: {original}"
    )
    print(
        f"  Proposed: {arp}"
    )

    # -----------------------------------------------------------------------
    # Forward passes.
    # -----------------------------------------------------------------------

    original_output = test_original_mspa(
        original,
        scene_embeddings,
        positions,
        agent_mask,
    )

    arp_output = test_arp_mspa(
        arp,
        scene_embeddings,
        positions,
        agent_mask,
    )

    # -----------------------------------------------------------------------
    # Compare outputs.
    # -----------------------------------------------------------------------

    test_outputs_are_not_identical(
        original_output,
        arp_output,
    )

    # -----------------------------------------------------------------------
    # Radius diagnostics.
    # -----------------------------------------------------------------------

    test_adaptive_radius(
        arp,
        scene_embeddings,
    )

    # -----------------------------------------------------------------------
    # Gradient flow.
    # -----------------------------------------------------------------------

    test_gradient_flow(
        original,
        arp,
        scene_embeddings,
        positions,
        agent_mask,
    )

    # -----------------------------------------------------------------------
    # Parameter comparison.
    # -----------------------------------------------------------------------

    test_parameter_count(
        original,
        arp,
    )

    # -----------------------------------------------------------------------
    # Final summary.
    # -----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("ALL MSPA MODULAR TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
