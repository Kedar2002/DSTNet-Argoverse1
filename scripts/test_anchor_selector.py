"""
scripts/test_anchor_selector.py

Targeted verification for:

    models.refinement.anchor_selector.AnchorSelector

Tests:

    trajectories
        (B,N,H,K,T,2)

        ↓

    AnchorSelector

        ↓

    anchors
        (B,N,H,K,2,2)

    radii
        (2,)

The two anchors are:

    anchor 0 = midpoint
    anchor 1 = endpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


###############################################################################
# Repository import setup
###############################################################################

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from models.refinement.anchor_selector import AnchorSelector


###############################################################################
# Helpers
###############################################################################


def check(
    condition: bool,
    message: str,
) -> None:

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

    check(
        bool(torch.isfinite(tensor).all().item()),
        f"{name}: all values finite",
    )


###############################################################################
# Main
###############################################################################


def main() -> None:

    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "--radius-start",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--radius-end",
        type=float,
        default=10.0,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 79
    )

    print(
        "ANCHOR SELECTOR TARGETED VERIFICATION"
    )

    print(
        "=" * 79
    )

    print(
        f"[INFO] Device = {device}"
    )

    print(
        f"[INFO] PyTorch = {torch.__version__}"
    )

    B = args.batch_size
    N = args.agents
    H = args.history
    K = args.modes
    T = args.prediction_steps

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
        f"[INFO] K = {K}"
    )

    print(
        f"[INFO] T = {T}"
    )

    print(
        f"[INFO] radius_start = {args.radius_start}"
    )

    print(
        f"[INFO] radius_end = {args.radius_end}"
    )

    ###########################################################################
    # Synthetic trajectories
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "1. SYNTHETIC INPUT"
    )

    print(
        "=" * 79
    )

    torch.manual_seed(42)

    trajectories = torch.randn(
        B,
        N,
        H,
        K,
        T,
        2,
        device=device,
        dtype=torch.float32,
    )

    print(
        f"[PASS] trajectories: shape={tuple(trajectories.shape)}"
    )

    check(
        tuple(trajectories.shape)
        == (B, N, H, K, T, 2),
        "Trajectory shape is correct",
    )

    check_finite(
        trajectories,
        "trajectories",
    )

    ###########################################################################
    # Instantiate selector
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "2. ANCHOR SELECTION"
    )

    print(
        "=" * 79
    )

    selector = AnchorSelector(
        radius_start=args.radius_start,
        radius_end=args.radius_end,
    ).to(device)

    print(
        "[PASS] AnchorSelector instantiated."
    )

    ###########################################################################
    # Forward
    ###########################################################################

    selection = selector(
        trajectories,
    )

    anchors = selection.anchors
    radii = selection.radii

    print(
        f"[INFO] anchors = {tuple(anchors.shape)}"
    )

    print(
        f"[INFO] radii   = {tuple(radii.shape)}"
    )

    ###########################################################################
    # Anchor shape
    ###########################################################################

    check(
        tuple(anchors.shape)
        == (B, N, H, K, 2, 2),
        "Anchor shape = (B,N,H,2,2)",
    )

    check_finite(
        anchors,
        "anchors",
    )

    ###########################################################################
    # Radius shape
    ###########################################################################

    check(
        tuple(radii.shape)
        == (2,),
        "Radius shape = (2,)",
    )

    check_finite(
        radii,
        "radii",
    )

    ###########################################################################
    # Midpoint index
    ###########################################################################

    expected_midpoint = (
        T - 1
    ) // 2

    check(
        selection.midpoint_index
        == expected_midpoint,
        "Midpoint index is correct",
    )

    check(
        selection.endpoint_index
        == T - 1,
        "Endpoint index is correct",
    )

    ###########################################################################
    # Exact anchor values
    ###########################################################################

    expected_midpoint_anchor = (
        trajectories[
            ...,
            expected_midpoint,
            :,
        ]
    )

    expected_endpoint_anchor = (
        trajectories[
            ...,
            T - 1,
            :,
        ]
    )

    check(
        torch.allclose(
            anchors[
                ...,
                0,
                :,
            ],
            expected_midpoint_anchor,
        ),
        "Anchor 0 equals trajectory midpoint",
    )

    check(
        torch.allclose(
            anchors[
                ...,
                1,
                :,
            ],
            expected_endpoint_anchor,
        ),
        "Anchor 1 equals trajectory endpoint",
    )

    ###########################################################################
    # Radius values
    ###########################################################################

    check(
        torch.allclose(
            radii[0],
            torch.tensor(
                args.radius_start,
                device=device,
            ),
        ),
        "First anchor radius equals radius_start",
    )

    check(
        torch.allclose(
            radii[1],
            torch.tensor(
                args.radius_end,
                device=device,
            ),
        ),
        "Second anchor radius equals radius_end",
    )

    check(
        bool(
            torch.all(
                radii[1:]
                <= radii[:-1]
            ).item()
        ),
        "Anchor radii are non-increasing",
    )

    ###########################################################################
    # Historical dimension preservation
    ###########################################################################

    check(
        anchors.shape[2] == H,
        "Historical dimension H is preserved",
    )

    ###########################################################################
    # Agent dimension preservation
    ###########################################################################

    check(
        anchors.shape[1] == N,
        "Agent dimension N is preserved",
    )

    ###########################################################################
    # Mode dimension preservation
    ###########################################################################

    check(
        anchors.shape[3] == K,
        "Prediction-mode dimension K is preserved",
    )

    ###########################################################################
    # Gradient test
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "3. GRADIENT VERIFICATION"
    )

    print(
        "=" * 79
    )

    trainable_input = trajectories.clone().detach()
    trainable_input.requires_grad_(True)

    trainable_selection = selector(
        trainable_input,
    )

    loss = (
        trainable_selection.anchors
        .square()
        .mean()
    )

    check(
        bool(torch.isfinite(loss).item()),
        "Anchor loss is finite",
    )

    loss.backward()

    check(
        trainable_input.grad is not None,
        "Gradient reaches trajectories",
    )

    if trainable_input.grad is not None:

        check_finite(
            trainable_input.grad,
            "trajectory gradient",
        )

        check(
            bool(
                trainable_input.grad
                .abs()
                .sum()
                .item()
                > 0.0
            ),
            "Trajectory gradient is non-zero",
        )

    ###########################################################################
    # Invalid input tests
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "4. INPUT VALIDATION"
    )

    print(
        "=" * 79
    )

    try:

        selector(
            torch.randn(
                B,
                N,
                H,
                K,
                T,
            ).to(device)
        )

    except (TypeError, ValueError):

        print(
            "[PASS] Selector rejects missing coordinate dimension."
        )

    else:

        raise AssertionError(
            "[FAIL] Selector accepted invalid trajectory rank."
        )

    try:

        selector(
            torch.randn(
                B,
                N,
                H,
                K,
                T,
                3,
            ).to(device)
        )

    except ValueError:

        print(
            "[PASS] Selector rejects non-2D coordinates."
        )

    else:

        raise AssertionError(
            "[FAIL] Selector accepted non-2D coordinates."
        )

    ###########################################################################
    # Final summary
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "ANCHOR SELECTOR VERIFICATION PASSED"
    )

    print(
        "=" * 79
    )

    print(
        f"[INFO] trajectories = {tuple(trajectories.shape)}"
    )

    print(
        f"[INFO] anchors      = {tuple(anchors.shape)}"
    )

    print(
        f"[INFO] radii        = {tuple(radii.shape)}"
    )


if __name__ == "__main__":
    main()
