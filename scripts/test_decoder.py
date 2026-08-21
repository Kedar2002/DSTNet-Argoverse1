"""
scripts/test_decoder.py

Phase-5 targeted verification for the DSTNet trajectory decoder.

Tests
-----
1. Decoder construction
2. Input/output tensor shapes
3. Numerical stability
4. Prediction dataclass integration
5. Historical dimension preservation
6. Mode dimension preservation
7. Batch size > 1
8. Zero-input behaviour
9. Gradient propagation
10. Training-mode forward pass
11. Evaluation-mode forward pass
12. Invalid input validation

Current DSTNet contract
-----------------------

Z_STM:

    (B,N,H,K,D)

Decoder:

    two-layer MLP

Coarse trajectory:

    Y^(0):

    (B,N,H,K,T,2)

Default test dimensions
-----------------------

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
# Make repository root importable
###############################################################################

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from models.decoder.decoder import Decoder
from models.model_types import Prediction


###############################################################################
# Test utilities
###############################################################################


def check(
    condition: bool,
    message: str,
) -> None:
    """
    Raise an AssertionError when a test condition fails.
    """

    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def check_shape(
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    name: str,
) -> None:
    """
    Verify tensor shape.
    """

    actual = tuple(tensor.shape)

    check(
        actual == expected,
        f"{name}: shape={actual}",
    )

    check(
        actual == expected,
        f"{name} shape matches {expected}.",
    )


def check_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Verify that all tensor values are finite.
    """

    finite = bool(
        torch.isfinite(tensor).all().item()
    )

    check(
        finite,
        f"{name}: all values finite",
    )


def check_nonzero(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Verify that a tensor contains at least one non-zero value.
    """

    nonzero = bool(
        torch.any(
            tensor != 0,
        ).item()
    )

    check(
        nonzero,
        f"{name}: contains non-zero values",
    )


###############################################################################
# Argument parser
###############################################################################


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Targeted Phase-5 verification for the "
            "DSTNet trajectory decoder."
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
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


###############################################################################
# Main
###############################################################################


def main() -> None:
    """
    Execute the decoder verification suite.
    """

    args = parse_args()

    torch.manual_seed(
        args.seed,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    B = args.batch_size
    N = args.agents
    H = args.history
    K = args.modes
    D = args.hidden_dim
    T = args.prediction_steps

    print()
    print("=" * 79)
    print("DSTNet DECODER PHASE-5 TARGETED VERIFICATION")
    print("=" * 79)

    print(
        f"[INFO] Device = {device}"
    )

    print(
        f"[INFO] PyTorch = {torch.__version__}"
    )

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

    ###########################################################################
    # 1. Instantiate decoder
    ###########################################################################

    print()
    print("=" * 79)
    print("1. DECODER CONSTRUCTION")
    print("=" * 79)

    decoder = Decoder(
        hidden_dim=D,
        prediction_steps=T,
        dropout=0.0,
    ).to(device)

    check(
        isinstance(
            decoder,
            torch.nn.Module,
        ),
        "Decoder instantiated.",
    )

    print(
        f"[INFO] Decoder = {decoder}"
    )

    ###########################################################################
    # 2. Synthetic Z_STM
    ###########################################################################

    print()
    print("=" * 79)
    print("2. SYNTHETIC Z_STM")
    print("=" * 79)

    z_stm = torch.randn(
        B,
        N,
        H,
        K,
        D,
        device=device,
        dtype=torch.float32,
    )

    check_shape(
        z_stm,
        (B, N, H, K, D),
        "Z_STM",
    )

    check_finite(
        z_stm,
        "Z_STM",
    )

    ###########################################################################
    # 3. Evaluation forward pass
    ###########################################################################

    print()
    print("=" * 79)
    print("3. EVALUATION FORWARD PASS")
    print("=" * 79)

    decoder.eval()

    with torch.no_grad():

        trajectories = decoder(
            z_stm,
        )

    expected_shape = (
        B,
        N,
        H,
        K,
        T,
        2,
    )

    check_shape(
        trajectories,
        expected_shape,
        "Y^(0)",
    )

    check_finite(
        trajectories,
        "Y^(0)",
    )

    check_nonzero(
        trajectories,
        "Y^(0)",
    )

    ###########################################################################
    # 4. Prediction dataclass
    ###########################################################################

    print()
    print("=" * 79)
    print("4. PREDICTION DATACLASS")
    print("=" * 79)

    prediction = Prediction(
        trajectories=trajectories,
    )

    check(
        isinstance(
            prediction,
            Prediction,
        ),
        "Prediction object created.",
    )

    check(
        prediction.trajectories is trajectories,
        "Prediction stores decoder trajectories.",
    )

    check_shape(
        prediction.trajectories,
        expected_shape,
        "Prediction.trajectories",
    )

    ###########################################################################
    # 5. Historical dimension preservation
    ###########################################################################

    print()
    print("=" * 79)
    print("5. HISTORICAL DIMENSION PRESERVATION")
    print("=" * 79)

    check(
        trajectories.shape[2] == H,
        "Decoder preserves historical dimension H.",
    )

    ###########################################################################
    # 6. Mode dimension preservation
    ###########################################################################

    print()
    print("=" * 79)
    print("6. MODE DIMENSION PRESERVATION")
    print("=" * 79)

    check(
        trajectories.shape[3] == K,
        "Decoder preserves prediction-mode dimension K.",
    )

    ###########################################################################
    # 7. Future trajectory dimension
    ###########################################################################

    print()
    print("=" * 79)
    print("7. FUTURE TRAJECTORY DIMENSION")
    print("=" * 79)

    check(
        trajectories.shape[4] == T,
        "Decoder produces the configured number of future steps.",
    )

    check(
        trajectories.shape[5] == 2,
        "Decoder produces 2-D trajectory coordinates.",
    )

    ###########################################################################
    # 8. Batch size > 1
    ###########################################################################

    print()
    print("=" * 79)
    print("8. BATCH SIZE > 1")
    print("=" * 79)

    check(
        B > 1,
        "Test uses batch size greater than one.",
    )

    check(
        trajectories.shape[0] == B,
        "Decoder preserves batch dimension.",
    )

    ###########################################################################
    # 9. Mode-specific representations
    ###########################################################################

    print()
    print("=" * 79)
    print("9. MODE-WISE DECODING")
    print("=" * 79)

    mode_0 = z_stm[:, :, :, 0, :]

    mode_1 = z_stm[:, :, :, 1, :]

    with torch.no_grad():

        mode_0_output = decoder(
            mode_0.unsqueeze(3),
        )

        mode_1_output = decoder(
            mode_1.unsqueeze(3),
        )

    check_shape(
        mode_0_output,
        (
            B,
            N,
            H,
            1,
            T,
            2,
        ),
        "Mode-0 trajectory",
    )

    check_shape(
        mode_1_output,
        (
            B,
            N,
            H,
            1,
            T,
            2,
        ),
        "Mode-1 trajectory",
    )

    different_modes = not bool(
        torch.allclose(
            mode_0_output,
            mode_1_output,
        )
    )

    check(
        different_modes,
        "Different mode embeddings can produce different trajectories.",
    )

    ###########################################################################
    # 10. Zero-input behaviour
    ###########################################################################

    print()
    print("=" * 79)
    print("10. ZERO-INPUT BEHAVIOUR")
    print("=" * 79)

    zero_input = torch.zeros(
        B,
        N,
        H,
        K,
        D,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():

        zero_output = decoder(
            zero_input,
        )

    check_shape(
        zero_output,
        expected_shape,
        "Zero-input output",
    )

    check_finite(
        zero_output,
        "Zero-input output",
    )

    ###########################################################################
    # 11. Training-mode forward pass
    ###########################################################################

    print()
    print("=" * 79)
    print("11. TRAINING-MODE FORWARD PASS")
    print("=" * 79)

    decoder.train()

    z_train = torch.randn(
        B,
        N,
        H,
        K,
        D,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    train_output = decoder(
        z_train,
    )

    check_shape(
        train_output,
        expected_shape,
        "Training output",
    )

    check_finite(
        train_output,
        "Training output",
    )

    ###########################################################################
    # 12. Gradient propagation
    ###########################################################################

    print()
    print("=" * 79)
    print("12. GRADIENT PROPAGATION")
    print("=" * 79)

    loss = (
        train_output.square().mean()
    )

    check(
        bool(
            torch.isfinite(loss).item()
        ),
        "Decoder test loss is finite.",
    )

    loss.backward()

    check(
        z_train.grad is not None,
        "Gradient reaches Z_STM.",
    )

    if z_train.grad is None:
        raise RuntimeError(
            "Z_STM gradient unexpectedly None."
        )

    check_finite(
        z_train.grad,
        "Z_STM gradient",
    )

    check_nonzero(
        z_train.grad,
        "Z_STM gradient",
    )

    ###########################################################################
    # 13. Decoder parameters receive gradients
    ###########################################################################

    print()
    print("=" * 79)
    print("13. DECODER PARAMETER GRADIENTS")
    print("=" * 79)

    parameters_with_grad = 0

    for name, parameter in decoder.named_parameters():

        if parameter.requires_grad:

            check(
                parameter.grad is not None,
                f"Gradient reaches decoder parameter: {name}",
            )

            if parameter.grad is not None:

                check_finite(
                    parameter.grad,
                    f"Gradient: {name}",
                )

                parameters_with_grad += 1

    check(
        parameters_with_grad > 0,
        "At least one decoder parameter received a gradient.",
    )

    ###########################################################################
    # 14. Invalid dimensionality
    ###########################################################################

    print()
    print("=" * 79)
    print("14. INPUT VALIDATION")
    print("=" * 79)

    invalid_input = torch.randn(
        B,
        N,
        H,
        D,
        device=device,
    )

    validation_triggered = False

    try:

        decoder(
            invalid_input,
        )

    except ValueError:

        validation_triggered = True

    check(
        validation_triggered,
        "Decoder rejects input without mode dimension.",
    )

    ###########################################################################
    # 15. Hidden dimension validation
    ###########################################################################

    print()
    print("=" * 79)
    print("15. HIDDEN DIMENSION VALIDATION")
    print("=" * 79)

    wrong_dim = D + 1

    wrong_input = torch.randn(
        B,
        N,
        H,
        K,
        wrong_dim,
        device=device,
    )

    validation_triggered = False

    try:

        decoder(
            wrong_input,
        )

    except ValueError:

        validation_triggered = True

    check(
        validation_triggered,
        "Decoder rejects incorrect hidden dimension.",
    )

    ###########################################################################
    # Final summary
    ###########################################################################

    print()
    print("=" * 79)
    print("DECODER PHASE-5 VERIFICATION PASSED")
    print("=" * 79)

    print()
    print(f"[PASS] Z_STM     = {tuple(z_stm.shape)}")
    print(f"[PASS] Y^(0)     = {tuple(trajectories.shape)}")
    print(f"[PASS] Prediction = {tuple(prediction.trajectories.shape)}")

    print()
    print("[PASS] Decoder construction")
    print("[PASS] Two-layer MLP forward pass")
    print("[PASS] Historical dimension H preservation")
    print("[PASS] Prediction-mode K preservation")
    print("[PASS] Future horizon T preservation")
    print("[PASS] 2-D trajectory coordinates")
    print("[PASS] Batch size > 1")
    print("[PASS] Mode-wise decoding")
    print("[PASS] Numerical stability")
    print("[PASS] Prediction dataclass integration")
    print("[PASS] Training-mode forward pass")
    print("[PASS] Gradient propagation")
    print("[PASS] Decoder parameter gradients")
    print("[PASS] Input validation")

    print()
    print(
        "Decoder is structurally and numerically consistent "
        "with the current DSTNet decoder contract."
    )


###############################################################################
# Entry point
###############################################################################


if __name__ == "__main__":
    main()
