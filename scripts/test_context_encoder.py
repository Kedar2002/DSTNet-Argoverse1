"""
scripts/test_context_encoder.py

Targeted verification for:

    models.refinement.context_encoder.ContextEncoder

Input:

    scene_features
        (B,N,H,K,D)

    anchors
        (B,N,H,K,2,2)

    radii
        (2,)

Output:

    context
        (B,N,H,K,2,D)
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


from models.refinement.context_encoder import ContextEncoder


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
        "--hidden-dim",
        type=int,
        default=256,
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
        "CONTEXT ENCODER TARGETED VERIFICATION"
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
    D = args.hidden_dim
    A = 2

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
        f"[INFO] A = {A}"
    )

    print(
        f"[INFO] D = {D}"
    )

    ###########################################################################
    # Synthetic input
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

    scene_features = torch.randn(
        B,
        N,
        H,
        K,
        D,
        device=device,
        dtype=torch.float32,
    )

    anchors = torch.randn(
        B,
        N,
        H,
        K,
        A,
        2,
        device=device,
        dtype=torch.float32,
    )

    radii = torch.tensor(
        [30.0, 10.0],
        device=device,
        dtype=torch.float32,
    )

    print(
        f"[PASS] scene_features: "
        f"shape={tuple(scene_features.shape)}"
    )

    print(
        f"[PASS] anchors: "
        f"shape={tuple(anchors.shape)}"
    )

    print(
        f"[PASS] radii: "
        f"shape={tuple(radii.shape)}"
    )

    check_finite(
        scene_features,
        "scene_features",
    )

    check_finite(
        anchors,
        "anchors",
    )

    check_finite(
        radii,
        "radii",
    )

    ###########################################################################
    # Instantiate
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "2. CONTEXT ENCODER"
    )

    print(
        "=" * 79
    )

    encoder = ContextEncoder(
        hidden_dim=D,
        dropout=0.0,
    ).to(device)

    print(
        "[PASS] ContextEncoder instantiated."
    )

    ###########################################################################
    # Forward
    ###########################################################################

    context = encoder(
        scene_features=scene_features,
        anchors=anchors,
        radii=radii,
    )

    print(
        f"[INFO] context = {tuple(context.shape)}"
    )

    ###########################################################################
    # Output shape
    ###########################################################################

    expected_shape = (
        B,
        N,
        H,
        K,
        A,
        D,
    )

    check(
        tuple(context.shape)
        == expected_shape,
        "Context shape = (B,N,H,K,2,D)",
    )

    check_finite(
        context,
        "context",
    )

    ###########################################################################
    # Non-zero output
    ###########################################################################

    check(
        bool(
            context.abs()
            .sum()
            .item()
            > 0.0
        ),
        "Context contains non-zero values",
    )

    ###########################################################################
    # Anchor separation
    ###########################################################################

    midpoint_context = context[
        ...,
        0,
        :,
    ]

    endpoint_context = context[
        ...,
        1,
        :,
    ]

    check(
        not torch.allclose(
            midpoint_context,
            endpoint_context,
        ),
        "Midpoint and endpoint contexts are not identical",
    )

    ###########################################################################
    # Historical dimension preservation
    ###########################################################################

    check(
        context.shape[2] == H,
        "Historical dimension H is preserved",
    )

    ###########################################################################
    # Mode dimension preservation
    ###########################################################################

    check(
        context.shape[3] == K,
        "Prediction-mode dimension K is preserved",
    )

    ###########################################################################
    # Agent dimension preservation
    ###########################################################################

    check(
        context.shape[1] == N,
        "Agent dimension N is preserved",
    )

    ###########################################################################
    # Anchor radius sensitivity
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "3. ANCHOR RADIUS SENSITIVITY"
    )

    print(
        "=" * 79
    )

    changed_radii = torch.tensor(
        [5.0, 50.0],
        device=device,
        dtype=torch.float32,
    )

    changed_context = encoder(
        scene_features=scene_features,
        anchors=anchors,
        radii=changed_radii,
    )

    check(
        not torch.allclose(
            context,
            changed_context,
        ),
        "Changing anchor radii changes contextual encoding",
    )

    ###########################################################################
    # Anchor geometry sensitivity
    ###########################################################################

    changed_anchors = anchors.clone()

    changed_anchors[
        ...,
        0,
        :,
    ] += 10.0

    geometry_context = encoder(
        scene_features=scene_features,
        anchors=changed_anchors,
        radii=radii,
    )

    check(
        not torch.allclose(
            context,
            geometry_context,
        ),
        "Changing anchor geometry changes contextual encoding",
    )

    ###########################################################################
    # Scene feature sensitivity
    ###########################################################################

    changed_scene = scene_features.clone()

    changed_scene[
        ...,
        0,
    ] += 1.0

    scene_context = encoder(
        scene_features=changed_scene,
        anchors=anchors,
        radii=radii,
    )

    check(
        not torch.allclose(
            context,
            scene_context,
        ),
        "Changing scene features changes contextual encoding",
    )

    ###########################################################################
    # Gradient propagation
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "4. GRADIENT VERIFICATION"
    )

    print(
        "=" * 79
    )

    train_scene = (
        scene_features.clone()
        .detach()
    )

    train_scene.requires_grad_(True)

    train_anchors = (
        anchors.clone()
        .detach()
    )

    train_anchors.requires_grad_(True)

    train_radii = (
        radii.clone()
        .detach()
    )

    train_radii.requires_grad_(True)

    train_context = encoder(
        scene_features=train_scene,
        anchors=train_anchors,
        radii=train_radii,
    )

    loss = (
        train_context.square()
        .mean()
    )

    check(
        bool(torch.isfinite(loss).item()),
        "Context encoder loss is finite",
    )

    loss.backward()

    check(
        train_scene.grad is not None,
        "Gradient reaches scene features",
    )

    if train_scene.grad is not None:

        check_finite(
            train_scene.grad,
            "scene feature gradient",
        )

        check(
            bool(
                train_scene.grad
                .abs()
                .sum()
                .item()
                > 0.0
            ),
            "Scene feature gradient is non-zero",
        )

    check(
        train_anchors.grad is not None,
        "Gradient reaches anchors",
    )

    if train_anchors.grad is not None:

        check_finite(
            train_anchors.grad,
            "anchor gradient",
        )

        check(
            bool(
                train_anchors.grad
                .abs()
                .sum()
                .item()
                > 0.0
            ),
            "Anchor gradient is non-zero",
        )

    ###########################################################################
    # Parameter gradients
    ###########################################################################

    parameter_gradient_found = False

    for name, parameter in (
        encoder.named_parameters()
    ):

        if parameter.grad is not None:

            parameter_gradient_found = True

            check_finite(
                parameter.grad,
                f"gradient: {name}",
            )

    check(
        parameter_gradient_found,
        "At least one ContextEncoder parameter receives gradient",
    )

    ###########################################################################
    # Batch-size > 1
    ###########################################################################

    check(
        B > 1,
        "Test uses batch size greater than one",
    )

    ###########################################################################
    # Invalid shape
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "5. INPUT VALIDATION"
    )

    print(
        "=" * 79
    )

    try:

        encoder(
            scene_features=scene_features,
            anchors=anchors[
                ...,
                :,
                :,
                :,
                :,
                :1,
            ],
            radii=radii,
        )

    except (TypeError, ValueError):

        print(
            "[PASS] ContextEncoder rejects invalid anchor coordinate dimension."
        )

    else:

        raise AssertionError(
            "[FAIL] ContextEncoder accepted invalid anchor shape."
        )

    ###########################################################################
    # Final summary
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "CONTEXT ENCODER VERIFICATION PASSED"
    )

    print(
        "=" * 79
    )

    print(
        f"[INFO] scene_features = {tuple(scene_features.shape)}"
    )

    print(
        f"[INFO] anchors        = {tuple(anchors.shape)}"
    )

    print(
        f"[INFO] context        = {tuple(context.shape)}"
    )


if __name__ == "__main__":
    main()
