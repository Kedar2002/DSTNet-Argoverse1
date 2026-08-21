"""
scripts/test_refinement.py

Targeted verification for the complete DSTNet refinement stage.

Pipeline:

    Z_STM
        +
    Prediction
        |
        v
    Refinement
        |
        v
    RefinedPrediction

Default synthetic configuration:

    B = 2
    N = 4
    H = 20
    K = 6
    D = 256
    T = 30
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


from models.model_types import Prediction
from models.refinement.refinement import Refinement


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
# Prediction construction
###############################################################################


def make_prediction(
    *,
    batch_size: int,
    agents: int,
    history: int,
    modes: int,
    prediction_steps: int,
    device: torch.device,
) -> Prediction:

    trajectories = torch.randn(
        batch_size,
        agents,
        history,
        modes,
        prediction_steps,
        2,
        device=device,
        dtype=torch.float32,
    )

    ###########################################################################
    # Valid multimodal probabilities
    ###########################################################################

    logits = torch.randn(
        batch_size,
        agents,
        history,
        modes,
        device=device,
        dtype=torch.float32,
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    return Prediction(
        trajectories=trajectories,
        probabilities=probabilities,
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

    parser.add_argument(
        "--prediction-steps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
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
        "DSTNet REFINEMENT PHASE-5 TARGETED VERIFICATION"
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
        f"[INFO] D = {D}"
    )

    print(
        f"[INFO] T = {T}"
    )

    print(
        f"[INFO] heads = {args.heads}"
    )

    print(
        f"[INFO] refinement_iterations = "
        f"{args.iterations}"
    )

    ###########################################################################
    # Synthetic Z_STM
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

    z_stm = torch.randn(
        B,
        N,
        H,
        K,
        D,
        device=device,
        dtype=torch.float32,
    )

    prediction = make_prediction(
        batch_size=B,
        agents=N,
        history=H,
        modes=K,
        prediction_steps=T,
        device=device,
    )

    print(
        f"[PASS] Z_STM: "
        f"shape={tuple(z_stm.shape)}"
    )

    print(
        f"[PASS] Prediction trajectories: "
        f"shape={tuple(prediction.trajectories.shape)}"
    )

    print(
        f"[PASS] Prediction probabilities: "
        f"shape={tuple(prediction.probabilities.shape)}"
    )

    check_finite(
        z_stm,
        "Z_STM",
    )

    check_finite(
        prediction.trajectories,
        "prediction trajectories",
    )

    check_finite(
        prediction.probabilities,
        "prediction probabilities",
    )

    ###########################################################################
    # Prediction probability normalization
    ###########################################################################

    check(
        torch.allclose(
            prediction.probabilities.sum(dim=-1),
            torch.ones(
                B,
                N,
                H,
                device=device,
            ),
            atol=1e-5,
            rtol=1e-5,
        ),
        "Input trajectory probabilities sum to one",
    )

    ###########################################################################
    # Instantiate refinement
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "2. REFINEMENT CONSTRUCTION"
    )

    print(
        "=" * 79
    )

    refinement = Refinement(
        hidden_dim=D,
        num_heads=args.heads,
        prediction_steps=T,
        refinement_iterations=args.iterations,
        radius_start=30.0,
        radius_end=10.0,
        dropout=0.0,
    ).to(device)

    print(
        "[PASS] Refinement instantiated."
    )

    ###########################################################################
    # Evaluation forward
    ###########################################################################

    refinement.eval()

    with torch.no_grad():

        refined_prediction = refinement(
            z_stm=z_stm,
            prediction=prediction,
        )

    print(
        "[PASS] Refinement forward pass completed."
    )

    ###########################################################################
    # Output shapes
    ###########################################################################

    expected_trajectory_shape = (
        B,
        N,
        H,
        K,
        T,
        2,
    )

    expected_probability_shape = (
        B,
        N,
        H,
        K,
    )

    expected_score_shape = (
        B,
        N,
        H,
        K,
    )

    check(
        tuple(
            refined_prediction.trajectories.shape
        )
        == expected_trajectory_shape,
        "Refined trajectories shape is correct",
    )

    check(
        tuple(
            refined_prediction.probabilities.shape
        )
        == expected_probability_shape,
        "Refined probabilities shape is correct",
    )

    check(
        tuple(
            refined_prediction.refinement_scores.shape
        )
        == expected_score_shape,
        "Refinement scores shape is correct",
    )

    check(
        tuple(
            refined_prediction.offsets.shape
        )
        == expected_trajectory_shape,
        "Refinement offsets shape is correct",
    )

    ###########################################################################
    # Numerical validation
    ###########################################################################

    check_finite(
        refined_prediction.trajectories,
        "refined trajectories",
    )

    check_finite(
        refined_prediction.probabilities,
        "refined probabilities",
    )

    check_finite(
        refined_prediction.refinement_scores,
        "refinement scores",
    )

    check_finite(
        refined_prediction.offsets,
        "refinement offsets",
    )

    ###########################################################################
    # Probability preservation
    ###########################################################################

    check(
        torch.allclose(
            refined_prediction.probabilities,
            prediction.probabilities,
        ),
        "Refinement preserves decoder trajectory probabilities",
    )

    ###########################################################################
    # Probability normalization
    ###########################################################################

    check(
        torch.allclose(
            refined_prediction.probabilities.sum(
                dim=-1
            ),
            torch.ones(
                B,
                N,
                H,
                device=device,
            ),
            atol=1e-5,
            rtol=1e-5,
        ),
        "Refined trajectory probabilities sum to one",
    )

    ###########################################################################
    # Offset consistency
    ###########################################################################

    calculated_offsets = (
        refined_prediction.trajectories
        - prediction.trajectories
    )

    check(
        torch.allclose(
            refined_prediction.offsets,
            calculated_offsets,
            atol=1e-5,
            rtol=1e-5,
        ),
        "Offsets equal refined - coarse trajectories",
    )

    ###########################################################################
    # Refinement actually changes trajectory
    ###########################################################################

    check(
        bool(
            refined_prediction.offsets
            .abs()
            .sum()
            .item()
            > 0.0
        ),
        "Refinement produces non-zero trajectory offsets",
    )

    ###########################################################################
    # Refinement score range
    ###########################################################################

    check(
        bool(
            (
                refined_prediction.refinement_scores
                >= 0.0
            ).all().item()
        ),
        "Refinement scores are >= 0",
    )

    check(
        bool(
            (
                refined_prediction.refinement_scores
                <= 1.0
            ).all().item()
        ),
        "Refinement scores are <= 1",
    )

    ###########################################################################
    # Historical dimension
    ###########################################################################

    check(
        refined_prediction.trajectories.shape[2]
        == H,
        "Historical dimension H is preserved",
    )

    ###########################################################################
    # Mode dimension
    ###########################################################################

    check(
        refined_prediction.trajectories.shape[3]
        == K,
        "Prediction-mode dimension K is preserved",
    )

    ###########################################################################
    # Anchor/context internal test
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "3. INTERNAL ANCHOR/CONTEXT VERIFICATION"
    )

    print(
        "=" * 79
    )

    with torch.no_grad():

        selection = (
            refinement.anchor_selector(
                prediction.trajectories,
            )
        )

        check(
            tuple(
                selection.anchors.shape
            )
            == (
                B,
                N,
                H,
                K,
                2,
                2,
            ),
            "Internal anchor tensor shape is correct",
        )

        anchor_context = (
            refinement.context_encoder(
                scene_features=z_stm,
                anchors=selection.anchors,
                radii=selection.radii,
            )
        )

    check(
        tuple(
            anchor_context.shape
        )
        == (
            B,
            N,
            H,
            K,
            2,
            D,
        ),
        "Internal anchor context shape is correct",
    )

    check_finite(
        anchor_context,
        "anchor context",
    )

    ###########################################################################
    # Training-mode forward
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "4. TRAINING / GRADIENT VERIFICATION"
    )

    print(
        "=" * 79
    )

    refinement.train()

    train_z = (
        z_stm.clone()
        .detach()
    )

    train_z.requires_grad_(True)

    train_prediction = Prediction(
        trajectories=(
            prediction.trajectories
            .clone()
            .detach()
        ),
        probabilities=(
            prediction.probabilities
            .clone()
            .detach()
        ),
    )

    train_prediction.trajectories.requires_grad_(
        True
    )

    train_output = refinement(
        z_stm=train_z,
        prediction=train_prediction,
    )

    train_loss = (
        train_output.trajectories.square().mean()
        + train_output.refinement_scores.mean()
    )

    check(
        bool(torch.isfinite(train_loss).item()),
        "Refinement training loss is finite",
    )

    train_loss.backward()

    ###########################################################################
    # Z_STM gradient
    ###########################################################################

    check(
        train_z.grad is not None,
        "Gradient reaches Z_STM",
    )

    if train_z.grad is not None:

        check_finite(
            train_z.grad,
            "Z_STM gradient",
        )

        check(
            bool(
                train_z.grad
                .abs()
                .sum()
                .item()
                > 0.0
            ),
            "Z_STM gradient is non-zero",
        )

    ###########################################################################
    # Coarse trajectory gradient
    ###########################################################################

    coarse_grad = (
        train_prediction
        .trajectories
        .grad
    )

    check(
        coarse_grad is not None,
        "Gradient reaches coarse trajectories",
    )

    if coarse_grad is not None:

        check_finite(
            coarse_grad,
            "coarse trajectory gradient",
        )

    ###########################################################################
    # Parameter gradients
    ###########################################################################

    parameter_gradient_found = False

    for name, parameter in (
        refinement.named_parameters()
    ):

        if parameter.grad is not None:

            parameter_gradient_found = True

            check_finite(
                parameter.grad,
                f"gradient: {name}",
            )

    check(
        parameter_gradient_found,
        "At least one Refinement parameter receives gradient",
    )

    ###########################################################################
    # Batch size
    ###########################################################################

    check(
        B > 1,
        "Test uses batch size greater than one",
    )

    ###########################################################################
    # Multiple iterations
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "5. ITERATIVE REFINEMENT"
    )

    print(
        "=" * 79
    )

    iterative_refinement = Refinement(
        hidden_dim=D,
        num_heads=args.heads,
        prediction_steps=T,
        refinement_iterations=2,
        radius_start=30.0,
        radius_end=10.0,
        dropout=0.0,
    ).to(device)

    iterative_refinement.eval()

    with torch.no_grad():

        iterative_output = (
            iterative_refinement(
                z_stm=z_stm,
                prediction=prediction,
            )
        )

    check(
        tuple(
            iterative_output.trajectories.shape
        )
        == expected_trajectory_shape,
        "Two-iteration refinement preserves output shape",
    )

    check_finite(
        iterative_output.trajectories,
        "two-iteration trajectories",
    )

    check_finite(
        iterative_output.offsets,
        "two-iteration offsets",
    )

    ###########################################################################
    # Refinement history
    ###########################################################################

    expected_history_shape = (
        B,
        N,
        H,
        K,
        args.iterations + 1,
        T,
        2,
    )

    check(
        refined_prediction.trajectory_history is not None,
        "Trajectory refinement history is present",
    )

    if refined_prediction.trajectory_history is not None:

        check(
            tuple(
                refined_prediction.trajectory_history.shape
            )
            == expected_history_shape,
            "Trajectory refinement history shape is correct",
        )

        check_finite(
            refined_prediction.trajectory_history,
            "trajectory refinement history",
        )

        check(
            torch.allclose(
                refined_prediction.trajectory_history[
                    ..., 0, :, :
                ],
                prediction.trajectories,
                atol=1e-5,
                rtol=1e-5,
            ),
            "History Y^(0) equals coarse prediction",
        )

        check(
            torch.allclose(
                refined_prediction.trajectory_history[
                    ..., -1, :, :
                ],
                refined_prediction.trajectories,
                atol=1e-5,
                rtol=1e-5,
            ),
            "History Y^(C) equals final refined prediction",
        )

    ###########################################################################
    # Refinement score history
    ###########################################################################

    expected_score_history_shape = (
        B,
        N,
        H,
        K,
        args.iterations + 1,
    )

    check(
        refined_prediction.refinement_score_history is not None,
        "Refinement score history is present",
    )

    if (
        refined_prediction.refinement_score_history
        is not None
    ):

        check(
            tuple(
                refined_prediction
                .refinement_score_history
                .shape
            )
            == expected_score_history_shape,
            "Refinement score history shape is correct",
        )

        check_finite(
            refined_prediction.refinement_score_history,
            "refinement score history",
        )

        check(
            torch.allclose(
                refined_prediction
                .refinement_score_history[..., -1],
                refined_prediction.refinement_scores,
                atol=1e-5,
                rtol=1e-5,
            ),
            "Final score equals last refinement score",
        )

    ###########################################################################
    # Final summary
    ###########################################################################

    print(
        "\n" + "=" * 79
    )

    print(
        "REFINEMENT PHASE-5 VERIFICATION PASSED"
    )

    print(
        "=" * 79
    )

    print(
        f"[INFO] Z_STM          = "
        f"{tuple(z_stm.shape)}"
    )

    print(
        f"[INFO] coarse         = "
        f"{tuple(prediction.trajectories.shape)}"
    )

    print(
        f"[INFO] refined        = "
        f"{tuple(refined_prediction.trajectories.shape)}"
    )

    print(
        f"[INFO] probabilities  = "
        f"{tuple(refined_prediction.probabilities.shape)}"
    )

    print(
        f"[INFO] scores         = "
        f"{tuple(refined_prediction.refinement_scores.shape)}"
    )

    print(
        f"[INFO] offsets        = "
        f"{tuple(refined_prediction.offsets.shape)}"
    )


if __name__ == "__main__":
    main()
