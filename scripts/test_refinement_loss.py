"""
scripts/test_refinement_loss.py

Targeted verification for DSTNet RefinementLoss.

Current contract
----------------

Refined trajectories:

    (B,N,H,K,T,2)

Ground truth:

    (B,N,T,2)

The test verifies:

1. RefinementLoss construction
2. Current DSTNet tensor contract
3. WTA mode selection
4. Refined trajectory regression
5. Endpoint loss
6. Bezier smoothness regularization
7. Total weighted loss
8. Numerical stability
9. Gradient propagation
10. Batch size > 1
11. Input validation
"""

from __future__ import annotations

import argparse
import sys

import torch

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.refinement_loss import RefinementLoss
from losses.targets import best_mode_from_endpoint

from models.model_types import RefinedPrediction


###############################################################################
# Configuration
###############################################################################


DEFAULT_BATCH_SIZE = 2
DEFAULT_AGENTS = 4
DEFAULT_HISTORY = 20
DEFAULT_MODES = 6
DEFAULT_PREDICTION_STEPS = 30


###############################################################################
# Output helpers
###############################################################################


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def info(message: str) -> None:
    print(f"[INFO] {message}")


def passed(message: str) -> None:
    print(f"[PASS] {message}")


def failed(message: str) -> None:
    print(f"[FAIL] {message}")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(
            f"[FAIL] {message}"
        )

    passed(message)


def check_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:

    check(
        isinstance(tensor, torch.Tensor),
        f"{name}: is a torch.Tensor",
    )

    check(
        bool(
            torch.isfinite(tensor)
            .all()
            .item()
        ),
        f"{name}: all values finite",
    )

    info(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}"
    )


###############################################################################
# Argument parser
###############################################################################


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Targeted verification of DSTNet "
            "RefinementLoss."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=DEFAULT_AGENTS,
    )

    parser.add_argument(
        "--history",
        type=int,
        default=DEFAULT_HISTORY,
    )

    parser.add_argument(
        "--modes",
        type=int,
        default=DEFAULT_MODES,
    )

    parser.add_argument(
        "--prediction-steps",
        type=int,
        default=DEFAULT_PREDICTION_STEPS,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    return parser.parse_args()


###############################################################################
# Main test
###############################################################################


def main() -> int:

    args = parse_args()

    device = torch.device(
        args.device
    )

    B = args.batch_size
    N = args.agents
    H = args.history
    K = args.modes
    T = args.prediction_steps

    ###########################################################################
    # Header
    ###########################################################################

    section(
        "DSTNet REFINEMENT LOSS TARGETED VERIFICATION"
    )

    info(
        f"Device = {device}"
    )

    info(
        f"PyTorch = {torch.__version__}"
    )

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
    # Basic configuration checks
    ###########################################################################

    check(
        B > 1,
        "Test uses batch size greater than one.",
    )

    check(
        N > 0,
        "Number of agents is positive.",
    )

    check(
        H > 0,
        "Historical dimension is positive.",
    )

    check(
        K > 1,
        "Prediction modes are greater than one.",
    )

    check(
        T > 1,
        "Prediction horizon is greater than one.",
    )

    ###########################################################################
    # Synthetic input
    ###########################################################################

    section(
        "1. SYNTHETIC INPUT"
    )

    torch.manual_seed(
        42
    )

    ###########################################################################
    # Ground truth
    #
    # (B,N,T,2)
    ###########################################################################

    ground_truth = torch.randn(
        B,
        N,
        T,
        2,
        dtype=torch.float32,
        device=device,
    )

    ###########################################################################
    # Refined trajectories
    #
    # (B,N,H,K,T,2)
    ###########################################################################

    refined_trajectories = (
        torch.randn(
            B,
            N,
            H,
            K,
            T,
            2,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
    )

    check_tensor(
        "refined trajectories",
        refined_trajectories,
    )

    check_tensor(
        "ground truth",
        ground_truth,
    )

    check(
        tuple(
            refined_trajectories.shape
        )
        == (
            B,
            N,
            H,
            K,
            T,
            2,
        ),
        (
            "Refined trajectory shape = "
            f"{(B,N,H,K,T,2)}."
        ),
    )

    check(
        tuple(
            ground_truth.shape
        )
        == (
            B,
            N,
            T,
            2,
        ),
        (
            "Ground-truth shape = "
            f"{(B,N,T,2)}."
        ),
    )

    ###########################################################################
    # Synthetic probabilities
    #
    # RefinedPrediction requires probabilities in the current model contract.
    ###########################################################################

    probabilities = torch.softmax(
        torch.randn(
            B,
            N,
            H,
            K,
            dtype=torch.float32,
            device=device,
        ),
        dim=-1,
    )

    check_tensor(
        "probabilities",
        probabilities,
    )

    check(
        tuple(
            probabilities.shape
        )
        == (
            B,
            N,
            H,
            K,
        ),
        (
            "Probability shape = "
            f"{(B,N,H,K)}."
        ),
    )

    check(
        torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones(
                B,
                N,
                H,
                device=device,
            ),
            atol=1e-5,
        ),
        "Prediction probabilities sum to one.",
    )

    ###########################################################################
    # Synthetic refinement outputs
    ###########################################################################

    offsets = (
        refined_trajectories
        - torch.zeros_like(
            refined_trajectories
        )
    )

    scores = torch.sigmoid(
        torch.randn(
            B,
            N,
            H,
            K,
            dtype=torch.float32,
            device=device,
        )
    )

    ###########################################################################
    # RefinedPrediction
    ###########################################################################

    prediction = RefinedPrediction(
        trajectories=refined_trajectories,
        probabilities=probabilities,
        refinement_scores=scores,
        offsets=offsets,
    )

    passed(
        "RefinedPrediction instantiated."
    )

    ###############################################################################
    # WTA target construction
    ###############################################################################

    section(
        "2. WTA TARGET CONSTRUCTION"
    )
    endpoint_distance_per_mode = torch.linalg.norm(
        refined_trajectories[..., -1, :]
        - ground_truth[:, :, None, None, -1, :],
        dim=-1,
    )

    check_tensor(
        "endpoint distance per mode",
        endpoint_distance_per_mode,
    )

    check(
        tuple(endpoint_distance_per_mode.shape)
        == (B, N, H, K),
        (
            "Endpoint distance per mode shape = "
            f"{(B, N, H, K)}."
        ),
    )

    ###############################################################################
    # Paper Eq. (28): winner-takes-all mode
    ###############################################################################

    best_endpoint_distance, best_mode = (
        endpoint_distance_per_mode.min(
            dim=-1
        )
    )

    check_tensor(
        "best endpoint distance",
        best_endpoint_distance,
    )

    check_tensor(
        "best_mode",
        best_mode,
    )

    check(
        tuple(best_endpoint_distance.shape)
        == (B, N, H),
        (
            "Best endpoint distance shape = "
            f"{(B, N, H)}."
        ),
    )

    check(
        tuple(best_mode.shape)
        == (B, N, H),
        (
            "best_mode shape = "
            f"{(B, N, H)}."
        ),
    )

    check(
        best_mode.dtype == torch.int64,
        "best_mode has dtype torch.int64.",
    )

    check(
        bool(
            (
                (best_mode >= 0)
                & (best_mode < K)
            )
            .all()
            .item()
        ),
        "best_mode indices are inside [0,K).",
    )

    ###############################################################################
    # Verify project helper
    ###############################################################################

    helper_endpoint, helper_best_mode = (
        best_mode_from_endpoint(
            refined_trajectories,
            ground_truth,
        )
    )

    check(
        tuple(helper_endpoint.shape)
        == (B, N, H),
        (
            "best_mode_from_endpoint endpoint "
            "output shape = (B,N,H)."
        ),
    )

    check(
        torch.equal(
            helper_best_mode,
            best_mode,
        ),
        (
            "best_mode_from_endpoint agrees with "
            "reference argmin calculation."
        ),
    )

    passed(
        "best_mode_from_endpoint matches the reference WTA calculation."
    )

    ###############################################################################
    # Verify argmin explicitly
    ###############################################################################

    check(
        torch.equal(
            best_mode,
            endpoint_distance_per_mode.argmin(
                dim=-1
            ),
        ),
        (
            "best_mode selects the minimum endpoint "
            "distance for every (B,N,H)."
        ),
    )

    info(
        f"Endpoint distance per mode = "
        f"{tuple(endpoint_distance_per_mode.shape)}"
    )

    info(
        f"best endpoint distance    = "
        f"{tuple(best_endpoint_distance.shape)}"
    )

    info(
        f"best_mode                 = "
        f"{tuple(best_mode.shape)}"
    )

    ###########################################################################
    # Instantiate loss
    ###########################################################################

    section(
        "3. REFINEMENT LOSS CONSTRUCTION"
    )

    loss_fn = RefinementLoss(
        trajectory_weight=1.0,
        endpoint_weight=0.5,
        smoothness_weight=0.05,
    )

    passed(
        "RefinementLoss instantiated."
    )

    info(
        f"Loss = {loss_fn}"
    )

    ###########################################################################
    # Forward
    ###########################################################################

    section(
        "4. REFINEMENT LOSS FORWARD"
    )

    result = loss_fn(
        prediction,
        ground_truth,
    )

    check(
        isinstance(
            result,
            dict,
        ),
        "RefinementLoss returns a dictionary.",
    )

    ###########################################################################
    # Required components
    ###########################################################################

    required_keys = {
        "loss",
        "regression",
        "endpoint",
        "smoothness",
    }

    check(
        required_keys.issubset(
            result.keys()
        ),
        (
            "Loss dictionary contains "
            "loss/regression/endpoint/smoothness."
        ),
    )

    ###########################################################################
    # Component checks
    ###########################################################################

    total_loss = result[
        "loss"
    ]

    regression_loss = result[
        "regression"
    ]

    endpoint_loss = result[
        "endpoint"
    ]

    smoothness_loss = result[
        "smoothness"
    ]

    check_tensor(
        "total refinement loss",
        total_loss,
    )

    check_tensor(
        "trajectory regression loss",
        regression_loss,
    )

    check_tensor(
        "endpoint loss",
        endpoint_loss,
    )

    check_tensor(
        "Bezier smoothness loss",
        smoothness_loss,
    )

    ###########################################################################
    # Scalar checks
    ###########################################################################

    check(
        total_loss.ndim == 0,
        "Total refinement loss is scalar.",
    )

    check(
        regression_loss.ndim == 0,
        "Regression loss is scalar.",
    )

    check(
        endpoint_loss.ndim == 0,
        "Endpoint loss is scalar.",
    )

    check(
        smoothness_loss.ndim == 0,
        "Smoothness loss is scalar.",
    )

    ###########################################################################
    # Non-negative checks
    ###########################################################################

    check(
        bool(
            regression_loss
            .detach()
            .item()
            >= 0.0
        ),
        "Regression loss is non-negative.",
    )

    check(
        bool(
            endpoint_loss
            .detach()
            .item()
            >= 0.0
        ),
        "Endpoint loss is non-negative.",
    )

    check(
        bool(
            smoothness_loss
            .detach()
            .item()
            >= 0.0
        ),
        "Smoothness loss is non-negative.",
    )

    check(
        bool(
            total_loss
            .detach()
            .item()
            >= 0.0
        ),
        "Total refinement loss is non-negative.",
    )

    ###########################################################################
    # Weighted total verification
    ###########################################################################

    expected_total = (
        1.0 * regression_loss
        + 0.5 * endpoint_loss
        + 0.05 * smoothness_loss
    )

    check(
        torch.allclose(
            total_loss,
            expected_total,
            atol=1e-6,
            rtol=1e-5,
        ),
        "Total loss equals the configured weighted components.",
    )

    ###########################################################################
    # Gradient test
    ###########################################################################

    section(
        "5. GRADIENT PROPAGATION"
    )

    refined_trajectories.grad = None

    total_loss.backward()

    check(
        refined_trajectories.grad is not None,
        "Gradient reaches refined trajectories.",
    )

    if refined_trajectories.grad is not None:

        check_tensor(
            "refined trajectory gradient",
            refined_trajectories.grad,
        )

        check(
            bool(
                refined_trajectories.grad
                .abs()
                .sum()
                .item()
                > 0.0
            ),
            "Refined trajectory gradient is non-zero.",
        )

    ###########################################################################
    # Gradient numerical stability
    ###########################################################################

    if refined_trajectories.grad is not None:

        check(
            bool(
                torch.isfinite(
                    refined_trajectories.grad
                )
                .all()
                .item()
            ),
            "Refined trajectory gradient is finite.",
        )

    ###########################################################################
    # Zero-error sanity test
    ###########################################################################

    section(
        "6. ZERO-ERROR SANITY CHECK"
    )

    ###########################################################################
    # Construct trajectories where every mode exactly follows GT.
    ###########################################################################

    zero_error_trajectories = (
        ground_truth
        .unsqueeze(2)
        .unsqueeze(2)
        .expand(
            B,
            N,
            H,
            K,
            T,
            2,
        )
        .clone()
        .detach()
        .requires_grad_(True)
    )

    zero_probabilities = torch.full(
        (
            B,
            N,
            H,
            K,
        ),
        1.0 / K,
        dtype=torch.float32,
        device=device,
    )

    zero_scores = torch.ones(
        B,
        N,
        H,
        K,
        dtype=torch.float32,
        device=device,
    )

    zero_offsets = torch.zeros_like(
        zero_error_trajectories
    )

    zero_prediction = RefinedPrediction(
        trajectories=zero_error_trajectories,
        probabilities=zero_probabilities,
        refinement_scores=zero_scores,
        offsets=zero_offsets,
    )

    zero_result = loss_fn(
        zero_prediction,
        ground_truth,
    )

    check_tensor(
        "zero-error total loss",
        zero_result["loss"],
    )

    check(
        bool(
            zero_result["regression"]
            .detach()
            .item()
            < 1e-6
        ),
        "Zero-error regression loss is approximately zero.",
    )

    check(
        bool(
            zero_result["endpoint"]
            .detach()
            .item()
            < 1e-6
        ),
        "Zero-error endpoint loss is approximately zero.",
    )

    ###########################################################################
    # Batch > 1
    ###########################################################################

    section(
        "7. BATCH SIZE > 1"
    )

    check(
        B > 1,
        "Test explicitly uses B > 1.",
    )

    check(
        tuple(
            prediction.trajectories.shape
        )[0]
        == B,
        "RefinedPrediction preserves batch dimension.",
    )

    ###########################################################################
    # Invalid input tests
    ###########################################################################

    section(
        "8. INPUT VALIDATION"
    )

    ###############################################################################
    # Wrong trajectory rank
    #
    # RefinedPrediction validates its own trajectory shape in __post_init__.
    # Therefore the invalid object must be constructed inside the try block.
    ###############################################################################

    try:

        invalid_prediction = RefinedPrediction(
            trajectories=torch.randn(
                B,
                N,
                K,
                T,
                2,
                device=device,
            ),
            probabilities=probabilities,
            refinement_scores=scores,
            offsets=torch.randn(
                B,
                N,
                K,
                T,
                2,
                device=device,
            ),
        )

        # This line should not be reached because RefinedPrediction
        # validates the trajectory shape during construction.
        loss_fn(
            invalid_prediction,
            ground_truth,
        )

    except (
        ValueError,
        TypeError,
        RuntimeError,
    ):

        passed(
            "Invalid refined trajectory shape is rejected."
        )

    else:

        failed(
            "Invalid refined trajectory shape was accepted."
        )

        return 1

    ###########################################################################
    # Wrong ground-truth rank
    ###########################################################################

    try:

        loss_fn(
            prediction,
            torch.randn(
                B,
                N,
                H,
                T,
                2,
                device=device,
            ),
        )

    except (
        ValueError,
        TypeError,
        RuntimeError,
    ):

        passed(
            "RefinementLoss rejects invalid "
            "ground-truth shape."
        )

    else:

        failed(
            "RefinementLoss accepted invalid "
            "ground-truth shape."
        )

        return 1

    ###########################################################################
    # Final summary
    ###########################################################################

    section(
        "9. REFINEMENT LOSS SUMMARY"
    )

    print(
        f"[PASS] Refined trajectories = "
        f"{tuple(refined_trajectories.shape)}"
    )

    print(
        f"[PASS] Ground truth         = "
        f"{tuple(ground_truth.shape)}"
    )

    print(
        f"[PASS] k_best              = "
        f"{tuple(best_mode.shape)}"
    )

    print(
        f"[PASS] Regression loss     = "
        f"{regression_loss.detach().item():.6f}"
    )

    print(
        f"[PASS] Endpoint loss       = "
        f"{endpoint_loss.detach().item():.6f}"
    )

    print(
        f"[PASS] Smoothness loss     = "
        f"{smoothness_loss.detach().item():.6f}"
    )

    print(
        f"[PASS] Total loss          = "
        f"{total_loss.detach().item():.6f}"
    )

    print()
    print(
        "[PASS] WTA target construction"
    )

    print(
        "[PASS] Refined trajectory regression"
    )

    print(
        "[PASS] Endpoint loss"
    )

    print(
        "[PASS] Bezier smoothness regularization"
    )

    print(
        "[PASS] Weighted total loss"
    )

    print(
        "[PASS] Numerical stability"
    )

    print(
        "[PASS] Gradient propagation"
    )

    print(
        "[PASS] Zero-error sanity check"
    )

    print(
        "[PASS] Batch size > 1"
    )

    print(
        "[PASS] Input validation"
    )

    print()
    print(
        "RefinementLoss is structurally and numerically "
        "consistent with the current DSTNet refinement contract."
    )

    return 0


###############################################################################
# Entry point
###############################################################################


if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "[FAIL] Test interrupted by user."
        )

        raise

    except Exception as exc:

        print()
        print(
            f"[FAIL] Unexpected test failure: "
            f"{type(exc).__name__}: {exc}"
        )

        raise
