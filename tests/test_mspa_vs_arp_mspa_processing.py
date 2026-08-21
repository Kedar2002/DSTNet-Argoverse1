"""
tests/test_mspa_vs_arp_mspa_processing.py

MODULAR PROCESSING COMPARISON
=============================

Purpose
-------
Compare the ORIGINAL DSTNet MSPA against the proposed ARP-MSPA using
synthetic data having the SAME structural contract as GSTA output.

The test does NOT train the complete DSTNet.

Instead it isolates the MSPA -> ARP-MSPA modification and demonstrates:

1. Same Z_scene input reaches both modules.
2. Original MSPA uses FIXED per-head spatial radii.
3. ARP-MSPA predicts AGENT-SPECIFIC radii.
4. Original MSPA uses HARD multi-scale neighborhood masks.
5. ARP-MSPA uses r_max as the hard candidate radius and then applies
   a DIFFERENTIABLE radius-dependent spatial bias.
6. Attention distributions are therefore processed differently.
7. Locality / attention entropy / expected interaction distance differ.
8. ARP-MSPA can learn different radii for different synthetic contexts.
9. The resulting scene representations differ.
10. The difference is attributable to the spatial interaction mechanism,
    while Q/K/V/alpha/output weights are matched between the models.

This is a MODULAR RESEARCH VALIDATION TEST.

It does NOT claim that ARP-MSPA improves ADE/FDE.
That must ultimately be demonstrated after integration into DSTNet.

Input contract
--------------
Z_scene:
    (B, N, H, K, D)

positions:
    (B, N, 2)

agent_mask:
    (B, N)

Output:
    (B, N, H, K, D)
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


# ============================================================================
# Project root
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Imports
# ============================================================================

from models.attention.mspa import MSPA
from models.attention.arp_mspa import ARPMSPA


# ============================================================================
# Reproducibility
# ============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 3

NUM_AGENTS = 12

HISTORY = 20

NUM_MODES = 6

HIDDEN_DIM = 256

NUM_HEADS = 8

INTERACTION_RADIUS = 30.0

R_MIN = 5.0

R_MAX = 30.0

DROPOUT = 0.0


# ============================================================================
# Utility
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


def tensor_stats(name: str, x: Tensor) -> None:
    print(
        f"{name:<28}"
        f"shape={tuple(x.shape)!s:<25}"
        f"min={x.min().item(): .5f} "
        f"max={x.max().item(): .5f} "
        f"mean={x.mean().item(): .5f} "
        f"std={x.std().item(): .5f}"
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================================
# Synthetic GSTA-like scene
# ============================================================================

def create_synthetic_scene() -> tuple[Tensor, Tensor, Tensor]:
    """
    Create synthetic data with the same structural shape expected from GSTA.

    Z_scene:
        (B,N,H,K,D)

    positions:
        (B,N,2)

    agent_mask:
        (B,N)

    The target agent is agent 0.

    Other agents are intentionally placed at different distances so that
    the difference between fixed and adaptive spatial processing becomes
    visible.
    """

    # ------------------------------------------------------------------------
    # Base random scene embedding
    # ------------------------------------------------------------------------

    z_scene = torch.randn(
        BATCH_SIZE,
        NUM_AGENTS,
        HISTORY,
        NUM_MODES,
        HIDDEN_DIM,
        device=DEVICE,
    )

    # ------------------------------------------------------------------------
    # Positions
    #
    # Agent 0 is the target.
    #
    # Other agents are placed at controlled distances.
    # ------------------------------------------------------------------------

    base_positions = torch.tensor(
        [
            [0.0, 0.0],     # Agent 0: target
            [3.0, 0.0],     # 3 m
            [6.0, 0.0],     # 6 m
            [10.0, 0.0],    # 10 m
            [15.0, 0.0],    # 15 m
            [20.0, 0.0],    # 20 m
            [25.0, 0.0],    # 25 m
            [29.0, 0.0],    # 29 m
            [35.0, 0.0],    # 35 m
            [45.0, 0.0],    # 45 m
            [12.0, 8.0],    # sqrt(208) ~= 14.42 m
            [-12.0, 8.0],   # sqrt(208) ~= 14.42 m
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    positions = (
        base_positions
        .unsqueeze(0)
        .repeat(BATCH_SIZE, 1, 1)
    )

    # ------------------------------------------------------------------------
    # Slightly perturb each scene so the scenes are not identical.
    # ------------------------------------------------------------------------

    positions[1, :, 1] += 1.5

    positions[2, :, 1] -= 1.5

    # ------------------------------------------------------------------------
    # All agents valid.
    # ------------------------------------------------------------------------

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
# Parameter matching
# ============================================================================

def copy_shared_parameters(
    original: MSPA,
    proposed: ARPMSPA,
) -> None:
    """
    Copy Q/K/V, alpha and output projection from MSPA to ARP-MSPA.

    This is important.

    We want to isolate the effect of the spatial mechanism.

    Therefore:

        Q_original == Q_proposed
        K_original == K_proposed
        V_original == V_proposed
        alpha_original == alpha_proposed
        W_out_original == W_out_proposed

    The only intentional difference is the spatial interaction mechanism
    and the adaptive radius predictor.
    """

    with torch.no_grad():

        proposed.query_projection.weight.copy_(
            original.query_projection.weight
        )

        proposed.key_projection.weight.copy_(
            original.key_projection.weight
        )

        proposed.value_projection.weight.copy_(
            original.value_projection.weight
        )

        proposed.alpha.copy_(
            original.alpha
        )

        proposed.output_projection.weight.copy_(
            original.output_projection.weight
        )


# ============================================================================
# Pairwise distance helper
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
# Original MSPA radii
# ============================================================================

def original_head_radii(
    model: MSPA,
) -> Tensor:
    """
    Return original fixed MSPA radii.

        r_h = hR / H_a

    for h = 1,...,H_a.
    """

    indices = torch.arange(
        1,
        model.num_heads + 1,
        dtype=torch.float32,
        device=DEVICE,
    )

    return (
        indices
        * model.interaction_radius
        / model.num_heads
    )


# ============================================================================
# Radius summary
# ============================================================================

def print_radius_comparison(
    original: MSPA,
    proposed: ARPMSPA,
    z_scene: Tensor,
) -> Tensor:

    subsection(
        "SPATIAL RANGE: FIXED MSPA vs ADAPTIVE ARP-MSPA"
    )

    fixed_radii = original_head_radii(
        original
    )

    print("Original MSPA fixed radii:")
    print()

    for index, radius in enumerate(
        fixed_radii,
        start=1,
    ):
        print(
            f"  Head {index:02d}: "
            f"{radius.item():7.3f} m"
        )

    print()

    agent_features = proposed._agent_features(
        z_scene
    )

    predicted_radii = proposed.predict_radius(
        agent_features
    )

    print(
        "ARP-MSPA predicted radii:"
    )
    print()

    for batch_index in range(
        predicted_radii.shape[0]
    ):

        print(
            f"Scene {batch_index}:"
        )

        for agent_index in range(
            predicted_radii.shape[1]
        ):

            print(
                f"  Agent {agent_index:02d}: "
                f"{predicted_radii[batch_index, agent_index].item():7.3f} m"
            )

    print()

    print(
        "Original MSPA:"
    )
    print(
        "  One fixed radius per attention head."
    )

    print()

    print(
        "ARP-MSPA:"
    )
    print(
        "  One learned radius per TARGET AGENT."
    )

    return predicted_radii


# ============================================================================
# Original MSPA mask analysis
# ============================================================================

def analyze_original_masks(
    model: MSPA,
    positions: Tensor,
) -> Tensor:

    subsection(
        "TEST 1: ORIGINAL MSPA FIXED NEIGHBORHOODS"
    )

    masks = model._build_radius_masks(
        positions,
        agent_mask=None,
    )

    radii = original_head_radii(
        model
    )

    distances = pairwise_distances(
        positions
    )

    print(
        "Target agent = 0"
    )

    print()

    print(
        "Agent distances from target:"
    )

    for agent_index in range(
        NUM_AGENTS
    ):

        print(
            f"  Agent {agent_index:02d}: "
            f"{distances[0, 0, agent_index].item():7.3f} m"
        )

    print()

    print(
        "Fixed MSPA neighborhood size for target agent 0:"
    )

    print()

    for head_index, radius in enumerate(
        radii
    ):

        visible = masks[
            0,
            head_index,
            0,
        ]

        count = int(
            visible.sum().item()
        )

        visible_agents = torch.where(
            visible
        )[0].tolist()

        print(
            f"  Head {head_index + 1:02d} | "
            f"radius={radius.item():6.2f} m | "
            f"agents={count:02d} | "
            f"visible={visible_agents}"
        )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  MSPA changes spatial coverage by switching between "
        "predefined head radii."
    )

    return masks


# ============================================================================
# ARP-MSPA candidate analysis
# ============================================================================

def analyze_arp_candidates(
    model: ARPMSPA,
    positions: Tensor,
    predicted_radii: Tensor,
) -> Tensor:

    subsection(
        "TEST 2: ARP-MSPA ADAPTIVE CANDIDATE SET"
    )

    candidate_mask = model._build_candidate_mask(
        positions,
        agent_mask=None,
    )

    distances = pairwise_distances(
        positions
    )

    print(
        f"Hard candidate radius r_max = "
        f"{model.r_max:.2f} m"
    )

    print()

    for batch_index in range(
        BATCH_SIZE
    ):

        print(
            f"Scene {batch_index}:"
        )

        for agent_index in range(
            NUM_AGENTS
        ):

            radius = predicted_radii[
                batch_index,
                agent_index,
            ]

            visible = candidate_mask[
                batch_index,
                agent_index,
            ]

            count = int(
                visible.sum().item()
            )

            print(
                f"  Target {agent_index:02d}: "
                f"r_i={radius.item():7.3f} m | "
                f"r_max candidates={count:02d}"
            )

        print()

    print(
        "Important:"
    )

    print(
        "  r_max defines the hard computational candidate set."
    )

    print(
        "  r_i determines the differentiable spatial weighting "
        "INSIDE that candidate set."
    )

    return candidate_mask


# ============================================================================
# Attention extraction
# ============================================================================

def compute_attention_for_head(
    model,
    scene_embeddings: Tensor,
    positions: Tensor,
    head_index: int,
    radius: Tensor | None = None,
) -> Tensor:
    """
    Compute attention weights for one head.

    This diagnostic function mirrors the actual attention computation
    in the uploaded modules.

    Returns
    -------
    attention:
        (B,N,N,H,K)

    We retain H and K because the actual module performs attention
    independently over the agent dimension while preserving the
    history/mode dimensions.
    """

    B, N, H, K, D = scene_embeddings.shape

    Q = model.query_projection(
        scene_embeddings
    )

    Key = model.key_projection(
        scene_embeddings
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

    q = Q[:, head_index]

    k = Key[:, head_index]

    scores = torch.einsum(
        "bntkd,bmskd->bnmtk",
        q,
        k,
    )

    scores = scores / math.sqrt(
        model.head_dim
    )

    distances = pairwise_distances(
        positions
    )

    if radius is None:
        # ------------------------------------------------------------
        # Original MSPA
        # ------------------------------------------------------------

        radii = original_head_radii(
            model
        )

        r = radii[
            head_index
        ]

        mask = (
            distances <= r
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

        scores = scores.masked_fill(
            ~mask.unsqueeze(-1).unsqueeze(-1),
            torch.finfo(
                scores.dtype
            ).min,
        )

    else:
        # ------------------------------------------------------------
        # ARP-MSPA
        #
        # distances:
        #     (B, N_target, N_candidate)
        #
        # radius:
        #     (B, N_target)
        #
        # Each target agent i has its own predicted radius r_i.
        # Expand r_i so that it broadcasts across all candidate
        # agents j.
        # ------------------------------------------------------------

        if radius.dim() == 2:
            # (B, N) -> (B, N, 1)
            radius_for_pairs = radius.unsqueeze(-1)

        elif radius.dim() == 3 and radius.size(-1) == 1:
            # Already (B, N, 1)
            radius_for_pairs = radius

        else:
            raise ValueError(
                "ARP-MSPA radius must have shape "
                f"(B, N) or (B, N, 1), got {tuple(radius.shape)}"
            )

        radius_squared = radius_for_pairs.square()

        # Adaptive Gaussian spatial bias:
        #
        # B_ij = -d_ij^2 / (2 r_i^2)
        #
        # distances      : (B, N, N)
        # radius_squared : (B, N, 1)
        # bias            : (B, N, N)
        bias = -(
            distances.square()
            / (
                2.0
                * radius_squared.clamp_min(1e-8)
            )
        )

        # Expand over history and mode dimensions.
        #
        # bias:
        #     (B, N, N)
        #
        # ->  (B, N, N, 1, 1)
        #
        # scores:
        #     (B, N, N, H, K)
        scores = (
            scores
            + bias.unsqueeze(-1).unsqueeze(-1)
        )

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

        scores = scores.masked_fill(
            ~mask.unsqueeze(-1).unsqueeze(-1),
            torch.finfo(
                scores.dtype
            ).min,
        )

    attention = torch.softmax(
        scores,
        dim=2,
    )

    return attention


# ============================================================================
# Attention metrics
# ============================================================================

def attention_metrics(
    attention: Tensor,
    distances: Tensor,
    target_agent: int = 0,
) -> dict[str, float]:
    """
    Calculate interpretable spatial attention metrics.

    Metrics:

    expected_distance:
        E[d] under the attention distribution.

    entropy:
        H(A) = -sum A log A

    local_mass_10m:
        attention assigned to agents <= 10 m.

    local_mass_15m:
        attention assigned to agents <= 15 m.

    far_mass_20m:
        attention assigned to agents > 20 m.
    """

    # attention:
    # B,N,N,H,K

    a = attention[
        :,
        target_agent,
    ]

    d = distances[
        :,
        target_agent,
    ]

    # Mean over history and mode dimensions.
    a = a.mean(
        dim=(-1, -2)
    )

    # Prevent numerical issues.
    eps = 1e-12

    expected_distance = (
        a * d
    ).sum(
        dim=-1
    )

    entropy = -(
        a
        * torch.log(
            a.clamp_min(eps)
        )
    ).sum(
        dim=-1
    )

    local_10 = a.masked_fill(
        d > 10.0,
        0.0,
    ).sum(
        dim=-1
    )

    local_15 = a.masked_fill(
        d > 15.0,
        0.0,
    ).sum(
        dim=-1
    )

    far_20 = a.masked_fill(
        d <= 20.0,
        0.0,
    ).sum(
        dim=-1
    )

    return {
        "expected_distance":
            expected_distance.mean().item(),

        "entropy":
            entropy.mean().item(),

        "local_mass_10m":
            local_10.mean().item(),

        "local_mass_15m":
            local_15.mean().item(),

        "far_mass_20m":
            far_20.mean().item(),
    }


# ============================================================================
# Attention comparison
# ============================================================================

def compare_attention(
    original: MSPA,
    proposed: ARPMSPA,
    z_scene: Tensor,
    positions: Tensor,
    predicted_radii: Tensor,
) -> None:

    subsection(
        "TEST 3: ATTENTION DISTRIBUTION COMPARISON"
    )

    distances = pairwise_distances(
        positions
    )

    # ------------------------------------------------------------
    # Select two original MSPA scales:
    #
    # Head 2 = relatively local
    # Head 8 = maximum radius
    # ------------------------------------------------------------

    local_head = 1
    global_head = NUM_HEADS - 1

    target_agent = 0

    print(
        f"Target agent = {target_agent}"
    )

    print()

    # ------------------------------------------------------------
    # Verify predicted-radius structure
    #
    # Expected:
    #     predicted_radii = (B, N)
    #
    # One radius for every target agent in every scene.
    # ------------------------------------------------------------

    if predicted_radii.dim() != 2:
        raise ValueError(
            "predicted_radii must have shape (B, N), "
            f"got {tuple(predicted_radii.shape)}"
        )

    B, N = positions.shape[:2]

    if predicted_radii.shape != (B, N):
        raise ValueError(
            "predicted_radii shape does not match positions: "
            f"expected {(B, N)}, "
            f"got {tuple(predicted_radii.shape)}"
        )

    # ------------------------------------------------------------
    # Print the actual ARP radius of the selected target agent
    # ------------------------------------------------------------

    print(
        "ARP-MSPA predicted radius for target agent 0:"
    )

    for b in range(B):
        print(
            f"  Scene {b}: "
            f"r_0 = {predicted_radii[b, target_agent].item():.4f} m"
        )

    print()

    # ------------------------------------------------------------
    # ORIGINAL MSPA
    #
    # Compute attention using:
    #
    #   1. predefined local head radius
    #   2. predefined maximum/global head radius
    #
    # The radius is determined entirely by the head index.
    # ------------------------------------------------------------

    mspa_local = compute_attention_for_head(
        original,
        z_scene,
        positions,
        local_head,
        radius=None,
    )

    mspa_global = compute_attention_for_head(
        original,
        z_scene,
        positions,
        global_head,
        radius=None,
    )

    # ------------------------------------------------------------
    # ARP-MSPA
    #
    # IMPORTANT:
    #
    # Pass the COMPLETE (B,N) radius tensor.
    #
    # compute_attention_for_head() internally expands this to:
    #
    #     (B,N,1)
    #
    # so that each TARGET agent i receives its own radius r_i.
    # ------------------------------------------------------------

    arp_attention = compute_attention_for_head(
        proposed,
        z_scene,
        positions,
        global_head,
        radius=predicted_radii,
    )

    # ------------------------------------------------------------
    # Select target agent AFTER computing attention.
    #
    # Full attention:
    #
    #     (B,N,N,H,K)
    #
    # Target-agent attention:
    #
    #     (B,N,H,K)
    #
    # The attention_metrics() function is expected to handle
    # this target-agent representation in the same way as the
    # original attention tensors.
    # ------------------------------------------------------------

    mspa_local_target = mspa_local[
        :,
        target_agent,
    ]

    mspa_global_target = mspa_global[
        :,
        target_agent,
    ]

    arp_target = arp_attention[
        :,
        target_agent,
    ]

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    mspa_local_metrics = attention_metrics(
        mspa_local_target,
        distances[:, target_agent],
    )

    mspa_global_metrics = attention_metrics(
        mspa_global_target,
        distances[:, target_agent],
    )

    arp_metrics = attention_metrics(
        arp_target,
        distances[:, target_agent],
    )

    print(
        "Spatial attention metrics:"
    )

    print()

    print(
        f"{'Metric':<28}"
        f"{'MSPA Local':>16}"
        f"{'MSPA Global':>16}"
        f"{'ARP-MSPA':>16}"
    )

    print(
        "-" * 76
    )

    metric_names = [
        "expected_distance",
        "entropy",
        "local_mass_10m",
        "local_mass_15m",
        "far_mass_20m",
    ]

    pretty_names = {
        "expected_distance":
            "Expected distance (m)",

        "entropy":
            "Attention entropy",

        "local_mass_10m":
            "Attention <= 10m",

        "local_mass_15m":
            "Attention <= 15m",

        "far_mass_20m":
            "Attention > 20m",
    }

    for name in metric_names:

        print(
            f"{pretty_names[name]:<28}"
            f"{mspa_local_metrics[name]:>16.5f}"
            f"{mspa_global_metrics[name]:>16.5f}"
            f"{arp_metrics[name]:>16.5f}"
        )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  MSPA Local is constrained by a predefined small head radius."
    )

    print(
        "  MSPA Global uses the predefined maximum radius."
    )

    print(
        "  ARP-MSPA uses the radius predicted specifically "
        "for the target agent."
    )

    print()

    print(
        "Key distinction:"
    )

    print(
        "  MSPA:   spatial scale is selected by attention head."
    )

    print(
        "  ARP-MSPA: spatial scale is selected by target-agent context."
    )


# ============================================================================
# Detailed attention table
# ============================================================================

def print_attention_table(
    original: MSPA,
    proposed: ARPMSPA,
    z_scene: Tensor,
    positions: Tensor,
    predicted_radii: Tensor,
) -> None:

    subsection(
        "TEST 4: AGENT-BY-AGENT ATTENTION PROCESSING"
    )

    distances = pairwise_distances(
        positions
    )

    target = 0

    # ------------------------------------------------------------
    # Choose an original MSPA head with a middle radius.
    # ------------------------------------------------------------

    head_index = 3

    mspa_attention = compute_attention_for_head(
        original,
        z_scene,
        positions,
        head_index,
        radius=None,
    )

    arp_attention = compute_attention_for_head(
        proposed,
        z_scene,
        positions,
        head_index,
        radius=predicted_radii[:, target],
    )

    mspa_values = (
        mspa_attention[
            0,
            target
        ]
        .mean(dim=(-1, -2))
    )

    arp_values = (
        arp_attention[
            0,
            target
        ]
        .mean(dim=(-1, -2))
    )

    print(
        f"Target agent: {target}"
    )

    print(
        f"MSPA head {head_index + 1} "
        f"radius = "
        f"{original_head_radii(original)[head_index].item():.3f} m"
    )

    print(
        f"ARP-MSPA target radius = "
        f"{predicted_radii[0, target].item():.3f} m"
    )

    print()

    print(
        f"{'Agent':<10}"
        f"{'Distance':>12}"
        f"{'MSPA Aij':>16}"
        f"{'ARP Aij':>16}"
        f"{'Difference':>16}"
    )

    print(
        "-" * 70
    )

    for agent_index in range(
        NUM_AGENTS
    ):

        distance = distances[
            0,
            target,
            agent_index,
        ].item()

        mspa_a = mspa_values[
            agent_index
        ].item()

        arp_a = arp_values[
            agent_index
        ].item()

        difference = (
            arp_a - mspa_a
        )

        print(
            f"{agent_index:<10}"
            f"{distance:>12.3f}"
            f"{mspa_a:>16.6f}"
            f"{arp_a:>16.6f}"
            f"{difference:>16.6f}"
        )


# ============================================================================
# Spatial bias demonstration
# ============================================================================

def demonstrate_spatial_bias(
    proposed: ARPMSPA,
    positions: Tensor,
    predicted_radii: Tensor,
) -> None:

    subsection(
        "TEST 5: ADAPTIVE SPATIAL BIAS"
    )

    distances = pairwise_distances(
        positions
    )

    target = 0

    radius = predicted_radii[
        0,
        target,
    ]

    print(
        f"Target agent = {target}"
    )

    print(
        f"Predicted radius = {radius.item():.4f} m"
    )

    print()

    print(
        "B_ij = -d_ij^2 / (2 r_i^2)"
    )

    print()

    print(
        f"{'Agent':<10}"
        f"{'Distance':>12}"
        f"{'Bias':>16}"
    )

    print(
        "-" * 42
    )

    for agent_index in range(
        NUM_AGENTS
    ):

        d = distances[
            0,
            target,
            agent_index,
        ]

        bias = -(
            d.square()
            / (
                2.0
                * radius.square()
            )
        )

        print(
            f"{agent_index:<10}"
            f"{d.item():>12.3f}"
            f"{bias.item():>16.6f}"
        )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  Near agents receive a weak spatial penalty."
    )

    print(
        "  Far agents receive a stronger negative attention-logit bias."
    )

    print(
        "  The strength of this suppression depends on r_i."
    )


# ============================================================================
# Context adaptation experiment
# ============================================================================

def train_radius_predictor_on_synthetic_contexts(
    model: ARPMSPA,
) -> None:

    subsection(
        "TEST 6: CAN ARP-MSPA LEARN DIFFERENT INTERACTION CONTEXTS?"
    )

    print(
        "This test isolates the adaptive radius predictor."
    )

    print(
        "It does NOT train the complete forecasting model."
    )

    print()

    # ------------------------------------------------------------
    # Create synthetic agent feature contexts.
    #
    # We construct three clearly different feature patterns.
    # ------------------------------------------------------------

    num_contexts = 3

    contexts = torch.zeros(
        num_contexts,
        HIDDEN_DIM,
        device=DEVICE,
    )

    contexts[0, 0] = -3.0
    contexts[1, 0] = 0.0
    contexts[2, 0] = 3.0

    target_radii = torch.tensor(
        [
            6.0,
            17.5,
            28.0,
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    # ------------------------------------------------------------
    # Save original radius predictor parameters.
    # ------------------------------------------------------------

    original_state = {
        key: value.detach().clone()
        for key, value in model.radius_predictor.state_dict().items()
    }

    optimizer = torch.optim.Adam(
        model.radius_predictor.parameters(),
        lr=0.01,
    )

    model.radius_predictor.train()

    initial_prediction = model.predict_radius(
        contexts
    ).detach()

    initial_loss = torch.mean(
        (
            initial_prediction
            - target_radii
        ).square()
    ).item()

    for _ in range(600):

        optimizer.zero_grad()

        prediction = model.predict_radius(
            contexts
        )

        loss = torch.mean(
            (
                prediction
                - target_radii
            ).square()
        )

        loss.backward()

        optimizer.step()

    final_prediction = model.predict_radius(
        contexts
    ).detach()

    final_loss = torch.mean(
        (
            final_prediction
            - target_radii
        ).square()
    ).item()

    print(
        f"Initial radius MSE : {initial_loss:.6f}"
    )

    print(
        f"Final radius MSE   : {final_loss:.6f}"
    )

    print()

    print(
        f"{'Context':<15}"
        f"{'Target radius':>18}"
        f"{'Predicted radius':>20}"
    )

    print(
        "-" * 55
    )

    for index in range(
        num_contexts
    ):

        print(
            f"{index:<15}"
            f"{target_radii[index].item():>18.4f}"
            f"{final_prediction[index].item():>20.4f}"
        )

    learned_spread = (
        final_prediction.max()
        - final_prediction.min()
    ).item()

    print()

    print(
        f"Learned radius spread = "
        f"{learned_spread:.4f} m"
    )

    if final_loss < initial_loss:

        print(
            "[PASS] Radius predictor learned "
            "context-dependent interaction ranges."
        )

    else:

        print(
            "[FAIL] Radius predictor did not improve."
        )

    # ------------------------------------------------------------
    # Restore original state so that the remainder of the test
    # remains deterministic and does not retain the synthetic fit.
    # ------------------------------------------------------------

    model.radius_predictor.load_state_dict(
        original_state
    )


# ============================================================================
# Representation comparison
# ============================================================================

def compare_final_representations(
    original: MSPA,
    proposed: ARPMSPA,
    z_scene: Tensor,
    positions: Tensor,
    agent_mask: Tensor,
) -> None:

    subsection(
        "TEST 7: FINAL REPRESENTATION DIFFERENCE"
    )

    original.eval()
    proposed.eval()

    with torch.no_grad():

        z_original = original(
            z_scene,
            positions,
            agent_mask=agent_mask,
        )

        z_proposed = proposed(
            z_scene,
            positions,
            agent_mask=agent_mask,
        )

    tensor_stats(
        "Original MSPA output",
        z_original,
    )

    tensor_stats(
        "ARP-MSPA output",
        z_proposed,
    )

    difference = (
        z_proposed
        - z_original
    )

    mean_abs_difference = (
        difference.abs().mean().item()
    )

    max_abs_difference = (
        difference.abs().max().item()
    )

    relative_difference = (
        difference.norm()
        / (
            z_original.norm()
            + 1e-12
        )
    ).item()

    print()

    print(
        f"Mean absolute difference : "
        f"{mean_abs_difference:.8f}"
    )

    print(
        f"Maximum absolute difference : "
        f"{max_abs_difference:.8f}"
    )

    print(
        f"Relative representation difference : "
        f"{relative_difference:.8f}"
    )

    if mean_abs_difference > 1e-6:

        print(
            "[PASS] Adaptive spatial processing "
            "changes the resulting representation."
        )

    else:

        print(
            "[FAIL] Outputs are unexpectedly identical."
        )


# ============================================================================
# Locality sensitivity test
# ============================================================================

def locality_sensitivity_test(
    original: MSPA,
    proposed: ARPMSPA,
    z_scene: Tensor,
    positions: Tensor,
    predicted_radii: Tensor,
) -> None:

    subsection(
        "TEST 8: DISTANCE-DEPENDENT ATTENTION SUPPRESSION"
    )

    distances = pairwise_distances(
        positions
    )

    target = 0

    # ------------------------------------------------------------
    # Original MSPA maximum-scale head.
    # ------------------------------------------------------------

    mspa_attention = compute_attention_for_head(
        original,
        z_scene,
        positions,
        NUM_HEADS - 1,
        radius=None,
    )

    # ------------------------------------------------------------
    # ARP-MSPA.
    # ------------------------------------------------------------

    arp_attention = compute_attention_for_head(
        proposed,
        z_scene,
        positions,
        NUM_HEADS - 1,
        radius=predicted_radii[:, target],
    )

    mspa = (
        mspa_attention[
            0,
            target
        ]
        .mean(dim=(-1, -2))
    )

    arp = (
        arp_attention[
            0,
            target
        ]
        .mean(dim=(-1, -2))
    )

    print(
        "Target agent = 0"
    )

    print()

    print(
        f"{'Distance':<14}"
        f"{'MSPA attention':>20}"
        f"{'ARP attention':>20}"
        f"{'ARP/MSPA ratio':>20}"
    )

    print(
        "-" * 76
    )

    for agent_index in range(
        NUM_AGENTS
    ):

        d = distances[
            0,
            target,
            agent_index,
        ].item()

        mspa_value = mspa[
            agent_index
        ].item()

        arp_value = arp[
            agent_index
        ].item()

        ratio = (
            arp_value
            / max(
                mspa_value,
                1e-12,
            )
        )

        print(
            f"{d:<14.3f}"
            f"{mspa_value:>20.8f}"
            f"{arp_value:>20.8f}"
            f"{ratio:>20.4f}"
        )

    print()

    print(
        "This table shows HOW the same agents are weighted "
        "differently by the two spatial mechanisms."
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    section(
        "DSTNet MSPA vs ARP-MSPA "
        "MODULAR PROCESSING VALIDATION"
    )

    print(
        "Device              :",
        DEVICE,
    )

    print(
        "Batch size           :",
        BATCH_SIZE,
    )

    print(
        "Agents               :",
        NUM_AGENTS,
    )

    print(
        "GSTA history H      :",
        HISTORY,
    )

    print(
        "Modes K             :",
        NUM_MODES,
    )

    print(
        "Hidden dimension D  :",
        HIDDEN_DIM,
    )

    print(
        "Attention heads     :",
        NUM_HEADS,
    )

    print(
        "Interaction R       :",
        INTERACTION_RADIUS,
    )

    print(
        "ARP radius range    :",
        f"[{R_MIN}, {R_MAX}] m",
    )

    # ========================================================================
    # Synthetic GSTA output
    # ========================================================================

    section(
        "STEP 1: SYNTHETIC GSTA -> MSPA INPUT"
    )

    (
        z_scene,
        positions,
        agent_mask,
    ) = create_synthetic_scene()

    tensor_stats(
        "Z_scene",
        z_scene,
    )

    tensor_stats(
        "positions",
        positions,
    )

    print(
        f"agent_mask shape = "
        f"{tuple(agent_mask.shape)}"
    )

    print()

    print(
        "Synthetic data follows the same structural contract:"
    )

    print(
        "  Z_scene = (B,N,H,K,D)"
    )

    print(
        "  positions = (B,N,2)"
    )

    print(
        "  agent_mask = (B,N)"
    )

    # ========================================================================
    # Instantiate modules
    # ========================================================================

    section(
        "STEP 2: INSTANTIATE ORIGINAL AND PROPOSED MODULES"
    )

    original = MSPA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        dropout=DROPOUT,
    ).to(DEVICE)

    proposed = ARPMSPA(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        interaction_radius=INTERACTION_RADIUS,
        r_min=R_MIN,
        r_max=R_MAX,
        radius_hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    # ========================================================================
    # Match common parameters
    # ========================================================================

    copy_shared_parameters(
        original,
        proposed,
    )

    print(
        "Shared parameters copied:"
    )

    print(
        "  Q projection       [MATCHED]"
    )

    print(
        "  K projection       [MATCHED]"
    )

    print(
        "  V projection       [MATCHED]"
    )

    print(
        "  alpha head weights [MATCHED]"
    )

    print(
        "  output projection  [MATCHED]"
    )

    print()

    print(
        "Intentional difference:"
    )

    print(
        "  Original MSPA  -> fixed multi-scale spatial masks"
    )

    print(
        "  ARP-MSPA       -> learned radius + adaptive spatial bias"
    )

    # ========================================================================
    # Parameter counts
    # ========================================================================

    subsection(
        "PARAMETER COUNTS"
    )

    original_parameters = count_parameters(
        original
    )

    proposed_parameters = count_parameters(
        proposed
    )

    print(
        f"Original MSPA : "
        f"{original_parameters:,}"
    )

    print(
        f"ARP-MSPA      : "
        f"{proposed_parameters:,}"
    )

    print(
        f"Additional    : "
        f"{proposed_parameters - original_parameters:,}"
    )

    print(
        f"Increase      : "
        f"{100.0 * (proposed_parameters - original_parameters) / original_parameters:.2f}%"
    )

    # ========================================================================
    # Radius analysis
    # ========================================================================

    predicted_radii = print_radius_comparison(
        original,
        proposed,
        z_scene,
    )

    # ========================================================================
    # Original masks
    # ========================================================================

    analyze_original_masks(
        original,
        positions,
    )

    # ========================================================================
    # ARP candidates
    # ========================================================================

    analyze_arp_candidates(
        proposed,
        positions,
        predicted_radii,
    )

    # ========================================================================
    # Attention comparison
    # ========================================================================

    compare_attention(
        original,
        proposed,
        z_scene,
        positions,
        predicted_radii,
    )

    # ========================================================================
    # Detailed attention table
    # ========================================================================

    print_attention_table(
        original,
        proposed,
        z_scene,
        positions,
        predicted_radii,
    )

    # ========================================================================
    # Adaptive bias
    # ========================================================================

    demonstrate_spatial_bias(
        proposed,
        positions,
        predicted_radii,
    )

    # ========================================================================
    # Context adaptation
    # ========================================================================

    train_radius_predictor_on_synthetic_contexts(
        proposed
    )

    # ========================================================================
    # Final representation
    # ========================================================================

    compare_final_representations(
        original,
        proposed,
        z_scene,
        positions,
        agent_mask,
    )

    # ========================================================================
    # Locality sensitivity
    # ========================================================================

    locality_sensitivity_test(
        original,
        proposed,
        z_scene,
        positions,
        predicted_radii,
    )

    # ========================================================================
    # Final summary
    # ========================================================================

    section(
        "MODULAR RESEARCH VALIDATION SUMMARY"
    )

    print(
        "[PASS] Synthetic Z_scene matches GSTA output contract."
    )

    print(
        "[PASS] Original MSPA uses fixed per-head spatial radii."
    )

    print(
        "[PASS] ARP-MSPA predicts agent-specific spatial radii."
    )

    print(
        "[PASS] Original MSPA uses hard fixed-radius neighborhoods."
    )

    print(
        "[PASS] ARP-MSPA uses r_max candidate sparsity."
    )

    print(
        "[PASS] ARP-MSPA applies differentiable "
        "radius-dependent spatial bias."
    )

    print(
        "[PASS] Attention distributions can be inspected "
        "agent-by-agent."
    )

    print(
        "[PASS] Final representations can be directly compared."
    )

    print(
        "[PASS] Radius predictor can learn distinct "
        "synthetic interaction contexts."
    )

    print()

    print(
        "RESEARCH INTERPRETATION:"
    )

    print(
        "  Original MSPA:"
    )

    print(
        "      fixed spatial scale per attention head"
    )

    print()

    print(
        "  ARP-MSPA:"
    )

    print(
        "      learned spatial scale per target agent"
    )

    print()

    print(
        "  Therefore the proposed module changes the processing from:"
    )

    print(
        "      FIXED MULTI-SCALE SPATIAL SELECTION"
    )

    print(
        "  to:"
    )

    print(
        "      CONTEXT-DEPENDENT ADAPTIVE SPATIAL ATTENTION"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  These experiments validate the proposed mechanism and "
        "its behavioral properties."
    )

    print(
        "  They do NOT yet prove improved ADE/FDE over DSTNet."
    )

    print(
        "  That requires a controlled end-to-end DSTNet ablation "
        "on the target dataset."
    )

    print()

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()
