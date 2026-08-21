"""
scripts.test_losses

Targeted verification for the first DSTNet Phase-7 losses.

Tests
-----
1. WTA endpoint target construction
2. ProposalLoss
3. ClassificationLoss
4. Shape validation
5. Numerical stability
6. Gradient propagation
7. B > 1
8. H > 1
9. Correct endpoint-based mode selection

Current DSTNet tensor contract
------------------------------

Coarse trajectories:

    (B,N,H,K,T,2)

Ground truth:

    (B,N,T,2)

WTA target:

    (B,N,H)

Classification scores:

    (B,N,H,K)
"""

from __future__ import annotations

import argparse
import math

import torch

from models.model_types import Prediction

from losses.targets import (
    best_mode_from_endpoint,
    validate_trajectory_shapes,
)

from losses.proposal_loss import ProposalLoss
from losses.classification_loss import ClassificationLoss


###############################################################################
# Utilities
###############################################################################


def check(
    condition: bool,
    message: str,
) -> None:
    """
    Assertion helper.
    """

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
    """
    Verify all tensor values are finite.
    """

    check(
        bool(torch.isfinite(tensor).all().item()),
        f"{name}: all values finite",
    )


def check_nonzero(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Verify tensor contains at least one non-zero value.
    """

    check(
        bool(
            tensor.detach()
            .abs()
            .sum()
            .item()
            > 0.0
        ),
        f"{name}: contains non-zero values",
    )


def print_shape(
    name: str,
    tensor: torch.Tensor,
) -> None:
    """
    Print tensor shape and dtype.
    """

    print(
        f"[INFO] {name}: "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}"
    )


###############################################################################
# Synthetic data
###############################################################################


def create_synthetic_data(
    *,
    batch_size: int,
    num_agents: int,
    history_steps: int,
    num_modes: int,
    prediction_steps: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Create deterministic synthetic trajectories and ground truth.

    The data is constructed so that mode 2 has the smallest
    endpoint error for every (B,N,H) entry.
    """

    torch.manual_seed(42)

    trajectories = torch.randn(
        batch_size,
        num_agents,
        history_steps,
        num_modes,
        prediction_steps,
        2,
        device=device,
    )

    ground_truth = torch.randn(
        batch_size,
        num_agents,
        prediction_steps,
        2,
        device=device,
    )

    ###########################################################################
    # Make mode 2 the known WTA mode.
    #
    # All modes receive relatively large endpoint errors.
    # Mode 2 receives a very small endpoint error.
    ###########################################################################

    target_endpoint = ground_truth[
        ...,
        -1,
        :,
    ]

    for k in range(num_modes):

        trajectories[
            ...,
            k,
            -1,
            :,
        ] = (
            target_endpoint.unsqueeze(2)
            + (k + 1) * 10.0
        )

    trajectories[
        ...,
        2,
        -1,
        :,
    ] = (
        target_endpoint.unsqueeze(2)
        + 0.01
    )

    return (
        trajectories,
        ground_truth,
    )


###############################################################################
# 1. Target construction
###############################################################################


def test_targets(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:

    print()
    print("=" * 79)
    print("1. WTA TARGET CONSTRUCTION")
    print("=" * 79)

    validate_trajectory_shapes(
        trajectories,
        ground_truth,
    )

    print_shape(
        "trajectories",
        trajectories,
    )

    print_shape(
        "ground_truth",
        ground_truth,
    )

    check(
        trajectories.ndim == 6,
        "Trajectory tensor has 6 dimensions",
    )

    check(
        ground_truth.ndim == 4,
        "Ground-truth tensor has 4 dimensions",
    )

    ###########################################################################
    # Endpoint target
    ###########################################################################

    endpoint_error, best_mode = (
        best_mode_from_endpoint(
            trajectories,
            ground_truth,
        )
    )

    print_shape(
        "endpoint_error",
        endpoint_error,
    )

    print_shape(
        "best_mode",
        best_mode,
    )

    B, N, H, K, T, C = trajectories.shape

    check(
        endpoint_error.shape
        == (B, N, H, K),
        "Endpoint error shape = (B,N,H,K)",
    )

    check(
        best_mode.shape
        == (B, N, H),
        "Best-mode shape = (B,N,H)",
    )

    check_finite(
        endpoint_error,
        "Endpoint error",
    )

    check_finite(
        best_mode.float(),
        "Best-mode tensor",
    )

    ###########################################################################
    # Known winner
    ###########################################################################

    expected_mode = 2

    check(
        bool(
            torch.all(
                best_mode == expected_mode
            ).item()
        ),
        "Endpoint-based WTA selects the known best mode",
    )

    ###########################################################################
    # Verify mode 2 actually has minimum endpoint error
    ###########################################################################

    selected_error = torch.gather(
        endpoint_error,
        dim=-1,
        index=best_mode.unsqueeze(-1),
    ).squeeze(-1)

    minimum_error = endpoint_error.min(
        dim=-1,
    ).values

    check(
        bool(
            torch.allclose(
                selected_error,
                minimum_error,
            )
        ),
        "Selected endpoint error equals minimum endpoint error",
    )

    ###########################################################################
    # Verify historical dimension is preserved
    ###########################################################################

    check(
        best_mode.shape[2] == H,
        "Historical dimension H is preserved",
    )

    return best_mode


###############################################################################
# 2. Proposal loss
###############################################################################


def test_proposal_loss(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    print()
    print("=" * 79)
    print("2. PROPOSAL LOSS")
    print("=" * 79)

    trajectories_for_loss = (
        trajectories.clone()
        .detach()
        .requires_grad_(True)
    )

    prediction = Prediction(
        trajectories=trajectories_for_loss,
        probabilities=None,
        scores=None,
    )

    loss_fn = ProposalLoss(
        delta=1.0,
    )

    check(
        isinstance(
            loss_fn,
            ProposalLoss,
        ),
        "ProposalLoss instantiated",
    )

    ###########################################################################
    # Return best mode as well
    ###########################################################################

    loss, best_mode = loss_fn(
        prediction,
        ground_truth,
        return_best_mode=True,
    )

    print(
        f"[INFO] Proposal loss = "
        f"{loss.detach().item():.8f}"
    )

    check(
        loss.ndim == 0,
        "Proposal loss is scalar",
    )

    check_finite(
        loss,
        "Proposal loss",
    )

    check(
        bool(
            torch.all(
                best_mode == 2
            ).item()
        ),
        "ProposalLoss uses endpoint-based WTA target",
    )

    ###########################################################################
    # Backward
    ###########################################################################

    loss.backward()

    check(
        trajectories_for_loss.grad is not None,
        "Gradient reaches coarse trajectories",
    )

    check_finite(
        trajectories_for_loss.grad,
        "Proposal trajectory gradient",
    )

    check_nonzero(
        trajectories_for_loss.grad,
        "Proposal trajectory gradient",
    )

    return (
        loss.detach(),
        best_mode.detach(),
    )


###############################################################################
# 3. Classification loss
###############################################################################


def test_classification_loss(
    trajectories: torch.Tensor,
    best_mode: torch.Tensor,
    *,
    num_modes: int,
) -> torch.Tensor:

    print()
    print("=" * 79)
    print("3. CLASSIFICATION LOSS")
    print("=" * 79)

    B, N, H, K, _, _ = trajectories.shape

    ###########################################################################
    # Construct learnable logits.
    #
    # Make the known WTA mode initially preferred.
    ###########################################################################

    scores = torch.randn(
        B,
        N,
        H,
        K,
        device=trajectories.device,
        requires_grad=True,
    )

    scores.data[..., 2] += 2.0

    prediction = Prediction(
        trajectories=trajectories.detach(),
        probabilities=None,
        scores=scores,
    )

    loss_fn = ClassificationLoss()

    check(
        isinstance(
            loss_fn,
            ClassificationLoss,
        ),
        "ClassificationLoss instantiated",
    )

    ###########################################################################
    # Loss
    ###########################################################################

    loss = loss_fn(
        prediction,
        best_mode,
    )

    print(
        f"[INFO] Classification loss = "
        f"{loss.detach().item():.8f}"
    )

    check(
        loss.ndim == 0,
        "Classification loss is scalar",
    )

    check_finite(
        loss,
        "Classification loss",
    )

    ###########################################################################
    # Backward
    ###########################################################################

    loss.backward()

    check(
        scores.grad is not None,
        "Gradient reaches classification scores",
    )

    check_finite(
        scores.grad,
        "Classification score gradient",
    )

    check_nonzero(
        scores.grad,
        "Classification score gradient",
    )

    ###########################################################################
    # Verify number of classes
    ###########################################################################

    check(
        scores.shape[-1] == num_modes,
        "Classification operates over K prediction modes",
    )

    return loss.detach()


###############################################################################
# 4. Numerical sanity
###############################################################################


def test_numerical_sanity(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
) -> None:

    print()
    print("=" * 79)
    print("4. NUMERICAL SANITY")
    print("=" * 79)

    ###########################################################################
    # Perfect trajectory
    ###########################################################################

    perfect = (
        ground_truth
        .unsqueeze(2)
        .unsqueeze(2)
        .expand(
            trajectories.shape[0],
            trajectories.shape[1],
            trajectories.shape[2],
            trajectories.shape[3],
            trajectories.shape[4],
            2,
        )
        .clone()
    )

    prediction = Prediction(
        trajectories=perfect,
        probabilities=None,
        scores=None,
    )

    proposal_loss = ProposalLoss()

    loss = proposal_loss(
        prediction,
        ground_truth,
    )

    print(
        f"[INFO] Perfect proposal loss = "
        f"{loss.item():.8f}"
    )

    check(
        math.isfinite(
            loss.item()
        ),
        "Perfect proposal loss is finite",
    )

    check(
        abs(loss.item()) < 1e-7,
        "Perfect proposal loss is approximately zero",
    )

    ###########################################################################
    # Very large values
    ###########################################################################

    large = torch.full_like(
        trajectories,
        1e3,
    )

    large_prediction = Prediction(
        trajectories=large,
        probabilities=None,
        scores=None,
    )

    large_loss = proposal_loss(
        large_prediction,
        ground_truth,
    )

    check_finite(
        large_loss,
        "Large-value proposal loss",
    )


###############################################################################
# 5. Input validation
###############################################################################


def test_validation(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
) -> None:

    print()
    print("=" * 79)
    print("5. INPUT VALIDATION")
    print("=" * 79)

    ###########################################################################
    # Wrong trajectory dimensions
    ###########################################################################

    try:

        validate_trajectory_shapes(
            trajectories[..., 0, :],
            ground_truth,
        )

        raise AssertionError(
            "[FAIL] Invalid trajectory rank was accepted"
        )

    except ValueError:

        print(
            "[PASS] Invalid trajectory rank rejected."
        )

    ###########################################################################
    # Wrong ground-truth dimensions
    ###########################################################################

    try:

        validate_trajectory_shapes(
            trajectories,
            ground_truth[..., 0],
        )

        raise AssertionError(
            "[FAIL] Invalid ground-truth rank was accepted"
        )

    except ValueError:

        print(
            "[PASS] Invalid ground-truth rank rejected."
        )

    ###########################################################################
    # Wrong coordinate dimension
    ###########################################################################

    bad_coordinates = trajectories[
        ...,
        :1,
    ]

    try:

        validate_trajectory_shapes(
            bad_coordinates,
            ground_truth,
        )

        raise AssertionError(
            "[FAIL] Invalid coordinate dimension was accepted"
        )

    except ValueError:

        print(
            "[PASS] Invalid trajectory coordinate dimension rejected."
        )


###############################################################################
# Main
###############################################################################


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Test DSTNet Phase-7 target, proposal, "
            "and classification losses."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--history",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--modes",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--prediction-steps",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    ###########################################################################
    # Device
    ###########################################################################

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 79)
    print("DSTNET PHASE-7 LOSS TARGETED VERIFICATION")
    print("=" * 79)

    print(
        f"[INFO] Device = {device}"
    )

    print(
        f"[INFO] PyTorch = {torch.__version__}"
    )

    print(
        f"[INFO] B = {args.batch_size}"
    )

    print(
        f"[INFO] N = {args.agents}"
    )

    print(
        f"[INFO] H = {args.history}"
    )

    print(
        f"[INFO] K = {args.modes}"
    )

    print(
        f"[INFO] T = {args.prediction_steps}"
    )

    ###########################################################################
    # Synthetic data
    ###########################################################################

    trajectories, ground_truth = (
        create_synthetic_data(
            batch_size=args.batch_size,
            num_agents=args.agents,
            history_steps=args.history,
            num_modes=args.modes,
            prediction_steps=args.prediction_steps,
            device=device,
        )
    )

    print()
    print("=" * 79)
    print("SYNTHETIC INPUT")
    print("=" * 79)

    print_shape(
        "Y^(0)",
        trajectories,
    )

    print_shape(
        "ground_truth",
        ground_truth,
    )

    check_finite(
        trajectories,
        "Y^(0)",
    )

    check_finite(
        ground_truth,
        "ground_truth",
    )

    ###########################################################################
    # Tests
    ###########################################################################

    best_mode = test_targets(
        trajectories,
        ground_truth,
    )

    proposal_loss, proposal_best_mode = (
        test_proposal_loss(
            trajectories,
            ground_truth,
        )
    )

    classification_loss = (
        test_classification_loss(
            trajectories,
            best_mode,
            num_modes=args.modes,
        )
    )

    test_numerical_sanity(
        trajectories,
        ground_truth,
    )

    test_validation(
        trajectories,
        ground_truth,
    )

    ###########################################################################
    # Final summary
    ###########################################################################

    print()
    print("=" * 79)
    print("FINAL RESULTS")
    print("=" * 79)

    print(
        f"[INFO] Y^(0)          = "
        f"{tuple(trajectories.shape)}"
    )

    print(
        f"[INFO] Ground truth   = "
        f"{tuple(ground_truth.shape)}"
    )

    print(
        f"[INFO] k_best         = "
        f"{tuple(best_mode.shape)}"
    )

    print(
        f"[INFO] Proposal loss  = "
        f"{proposal_loss.item():.8f}"
    )

    print(
        f"[INFO] Classification = "
        f"{classification_loss.item():.8f}"
    )

    print()

    check(
        torch.equal(
            best_mode,
            proposal_best_mode,
        ),
        "ProposalLoss and target construction produce identical k_best",
    )

    check(
        args.batch_size > 1,
        "Test uses batch size greater than one",
    )

    check(
        args.history > 1,
        "Test preserves historical dimension H",
    )

    check(
        args.modes > 1,
        "Test uses multiple prediction modes",
    )

    print()
    print("=" * 79)
    print("PHASE-7 INITIAL LOSSES PASSED")
    print("=" * 79)

    print(
        """
Verified:

[PASS] Endpoint-based Winner-Takes-All
[PASS] k_best shape = (B,N,H)
[PASS] Endpoint distance selection
[PASS] Proposal Huber loss
[PASS] Proposal gradient propagation
[PASS] Classification cross-entropy
[PASS] Classification gradient propagation
[PASS] Historical dimension H preservation
[PASS] Prediction-mode dimension K preservation
[PASS] Batch size > 1
[PASS] Numerical stability
[PASS] Input validation
[PASS] Proposal/target consistency
"""
    )


if __name__ == "__main__":
    main()
