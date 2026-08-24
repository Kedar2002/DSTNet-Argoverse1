"""
tests.test_smoke_checkpoint

Post-smoke diagnostic test for the current DSTNet implementation.

Purpose
-------
Verify the checkpoint and current production training interfaces after
a successful CPU smoke-training run.

Checks
------
1. Smoke checkpoint exists and loads.
2. Current dataset configuration resolves correctly.
3. A small real Argoverse-1 subset can be built.
4. Current DataLoader interface works.
5. Current DSTNet model can be constructed.
6. Checkpoint weights can be restored.
7. One inference forward pass works.
8. Model parameters are finite.
9. Current optimizer / scheduler / loss construction works.

Reference
---------
DSTNet base paper:
Dynamic Trajectory Prediction for Autonomous Vehicles via
Spatio-Temporal Attention.

The paper defines the architectural/training methodology.
The current repository implementation is treated as the authoritative
source for the actual Python interfaces and tensor contracts.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset


###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


###############################################################################
# Current Production Training Interfaces
###############################################################################

from scripts.train import (
    CFG,
    build_dataset,
    build_dataloader,
    build_dataset_roots,
    build_model,
    build_criterion,
    build_training_optimizer,
    build_training_scheduler,
)


###############################################################################
# Configuration
###############################################################################

DEVICE = torch.device(
    "cpu"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "smoke"
    / "smoke_epoch_1.pth"
)

TEST_SCENES = 2

BATCH_SIZE = 2

NUM_WORKERS = 0

PIN_MEMORY = False


###############################################################################
# Output Helpers
###############################################################################

def print_section(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def passed(
    message: str,
) -> None:

    print(
        f"[PASS] {message}"
    )


def failed(
    message: str,
) -> None:

    print(
        f"[FAIL] {message}"
    )


###############################################################################
# Tensor Inspection
###############################################################################

def inspect_tensor(
    name: str,
    value: Any,
) -> None:

    if not isinstance(
        value,
        torch.Tensor,
    ):

        print(
            f"{name}: "
            f"{type(value).__name__}"
        )

        return

    finite = bool(
        torch.isfinite(
            value
        ).all()
    )

    print(
        f"{name}: "
        f"shape={tuple(value.shape)}, "
        f"dtype={value.dtype}, "
        f"device={value.device}, "
        f"finite={finite}"
    )


###############################################################################
# Recursive Batch Inspection
###############################################################################

def inspect_structure(
    value: Any,
    name: str = "batch",
    depth: int = 0,
    max_depth: int = 4,
) -> None:

    if depth > max_depth:
        return

    if isinstance(
        value,
        torch.Tensor,
    ):

        inspect_tensor(
            name,
            value,
        )

        return

    if isinstance(
        value,
        dict,
    ):

        for key, item in value.items():

            inspect_structure(
                item,
                name=f"{name}.{key}",
                depth=depth + 1,
                max_depth=max_depth,
            )

        return

    if isinstance(
        value,
        (list, tuple),
    ):

        for index, item in enumerate(
            value
        ):

            inspect_structure(
                item,
                name=f"{name}[{index}]",
                depth=depth + 1,
                max_depth=max_depth,
            )

        return


###############################################################################
# Recursive Device Transfer
###############################################################################

def move_to_device(
    value: Any,
    device: torch.device,
) -> Any:

    if isinstance(
        value,
        torch.Tensor,
    ):

        return value.to(
            device
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            key: move_to_device(
                item,
                device,
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            move_to_device(
                item,
                device,
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):

        return tuple(
            move_to_device(
                item,
                device,
            )
            for item in value
        )

    return value


###############################################################################
# 1. Checkpoint Integrity
###############################################################################

def test_checkpoint() -> dict:
    """
    Verify that the smoke checkpoint exists and can be loaded.
    """

    print_section(
        "1. Checkpoint Integrity"
    )

    print(
        f"Path : {CHECKPOINT_PATH}"
    )

    if not CHECKPOINT_PATH.exists():

        failed(
            "Checkpoint does not exist."
        )

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    size_mb = (
        CHECKPOINT_PATH.stat().st_size
        / 1024.0
        / 1024.0
    )

    print(
        f"Size : {size_mb:.2f} MB"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    print(
        "Checkpoint keys:"
    )

    for key in checkpoint:

        print(
            f"  - {key}"
        )

    passed(
        "Checkpoint exists and loads successfully."
    )

    return checkpoint


###############################################################################
# 2. Configuration / Dataset Roots
###############################################################################

def test_dataset_roots() -> tuple[Path, Path]:
    """
    Use the current configuration-based dataset-root interface.
    """

    print_section(
        "2. Dataset Configuration"
    )

    print(
        "Current dataset configuration:"
    )

    if hasattr(
        CFG,
        "dataset",
    ):

        dataset_cfg = CFG.dataset

        for name in (
            "train_dir",
            "val_dir",
            "cache_dir",
            "map_dir",
        ):

            if hasattr(
                dataset_cfg,
                name,
            ):

                print(
                    f"  {name:<16}: "
                    f"{getattr(dataset_cfg, name)}"
                )

    train_root, val_root = (
        build_dataset_roots()
    )

    print()
    print(
        f"Resolved training root   : "
        f"{train_root}"
    )

    print(
        f"Resolved validation root : "
        f"{val_root}"
    )

    if not train_root.exists():

        raise FileNotFoundError(
            "Training directory does not exist:\n"
            f"{train_root}"
        )

    if not val_root.exists():

        raise FileNotFoundError(
            "Validation directory does not exist:\n"
            f"{val_root}"
        )

    passed(
        "Training and validation roots resolve and exist."
    )

    return (
        train_root,
        val_root,
    )


###############################################################################
# 3. Dataset / DataLoader
###############################################################################

def test_dataloader(
    train_root: Path,
):
    """
    Build a tiny subset using the current production dataset pipeline.
    """

    print_section(
        "3. Dataset / DataLoader"
    )

    dataset = build_dataset(
        train_root,
        train=True,
    )

    print(
        f"Full dataset : "
        f"{len(dataset):,} scenes"
    )

    subset_size = min(
        TEST_SCENES,
        len(dataset),
    )

    dataset = Subset(
        dataset,
        range(subset_size),
    )

    print(
        f"Test subset  : "
        f"{len(dataset):,} scenes"
    )

    loader = build_dataloader(
        dataset,
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(
        f"Batch size   : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Workers      : "
        f"{NUM_WORKERS}"
    )

    print(
        f"Batches      : "
        f"{len(loader)}"
    )

    batch = next(
        iter(loader)
    )

    print()
    print(
        "Batch structure:"
    )

    inspect_structure(
        batch
    )

    passed(
        "Dataset and DataLoader produced a valid batch."
    )

    return batch


###############################################################################
# 4. Model Construction
###############################################################################

def test_model(
):
    """
    Construct the current production DSTNet.
    """

    print_section(
        "4. DSTNet Construction"
    )

    model = build_model()

    model.to(
        DEVICE
    )

    model.eval()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    passed(
        "Current DSTNet model constructed successfully."
    )

    return model


###############################################################################
# 5. Restore Checkpoint
###############################################################################

def extract_model_state(
    checkpoint: dict,
):
    """
    Locate the model state in the current checkpoint format.

    Supports the known naming variants without changing the
    production checkpoint implementation.
    """

    candidates = (
        "model_state_dict",
        "model",
        "state_dict",
    )

    for key in candidates:

        if key in checkpoint:

            return (
                key,
                checkpoint[key],
            )

    raise KeyError(
        "No model state found in checkpoint. "
        f"Checked: {candidates}"
    )


def test_checkpoint_restore(
    model: torch.nn.Module,
    checkpoint: dict,
) -> None:

    print_section(
        "5. Checkpoint Restore"
    )

    state_key, model_state = (
        extract_model_state(
            checkpoint
        )
    )

    print(
        f"Model state key : "
        f"{state_key}"
    )

    result = model.load_state_dict(
        model_state,
        strict=False,
    )

    missing = result.missing_keys
    unexpected = result.unexpected_keys

    print(
        f"Missing keys    : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected keys : "
        f"{len(unexpected)}"
    )

    if missing:

        print()
        print(
            "Missing keys:"
        )

        for key in missing[:20]:

            print(
                f"  {key}"
            )

    if unexpected:

        print()
        print(
            "Unexpected keys:"
        )

        for key in unexpected[:20]:

            print(
                f"  {key}"
            )

    if missing or unexpected:

        raise RuntimeError(
            "Checkpoint state_dict does not exactly "
            "match the current DSTNet model."
        )

    passed(
        "Smoke checkpoint restores exactly into current DSTNet."
    )


###############################################################################
# 6. Forward Pass
###############################################################################

def test_forward(
    model: torch.nn.Module,
    batch: Any,
):
    """
    Run one inference pass.

    The current train.py / model interface is inspected rather than
    assuming an obsolete positional interface.
    """

    print_section(
        "6. Forward Pass"
    )

    batch = move_to_device(
        batch,
        DEVICE,
    )

    signature = inspect.signature(
        model.forward
    )

    print(
        "Current DSTNet.forward:"
    )

    print(
        signature
    )

    ###########################################################################
    # The current collate output is expected to be a dictionary-like batch.
    ###########################################################################

    if not isinstance(
        batch,
        dict,
    ):

        raise TypeError(
            "Expected DataLoader batch to be a dictionary, "
            f"got {type(batch).__name__}."
        )

    print()
    print(
        "Batch keys:"
    )

    for key in batch:

        print(
            f"  - {key}"
        )

    ###########################################################################
    # Match named forward parameters.
    ###########################################################################

    kwargs = {}

    for name in signature.parameters:

        if name == "self":
            continue

        if name in batch:

            kwargs[name] = batch[name]

    print()
    print(
        "Matched forward inputs:"
    )

    for name, value in kwargs.items():

        if isinstance(
            value,
            torch.Tensor,
        ):

            print(
                f"  {name:<24} "
                f"{tuple(value.shape)}"
            )

        else:

            print(
                f"  {name:<24} "
                f"{type(value).__name__}"
            )

    if not kwargs:

        raise RuntimeError(
            "Could not match any model.forward parameters "
            "against the current batch."
        )

    ###########################################################################
    # Forward
    ###########################################################################

    with torch.no_grad():

        output = model(
            **kwargs
        )

    print()
    print(
        f"Output type : "
        f"{type(output).__name__}"
    )

    if isinstance(
        output,
        tuple,
    ):

        for index, item in enumerate(
            output
        ):

            inspect_tensor(
                f"output[{index}]",
                item,
            )

    elif isinstance(
        output,
        dict,
    ):

        for key, value in output.items():

            inspect_tensor(
                f"output.{key}",
                value,
            )

    else:

        inspect_tensor(
            "output",
            output,
        )

    passed(
        "DSTNet forward pass completed."
    )

    return output


###############################################################################
# 7. Parameter Numerical Sanity
###############################################################################

def test_parameter_sanity(
    model: torch.nn.Module,
) -> None:

    print_section(
        "7. Parameter Numerical Sanity"
    )

    invalid = []

    total = 0

    for name, parameter in model.named_parameters():

        total += parameter.numel()

        if not torch.isfinite(
            parameter
        ).all():

            invalid.append(
                name
            )

    print(
        f"Parameters checked : "
        f"{total:,}"
    )

    if invalid:

        print(
            "Invalid parameters:"
        )

        for name in invalid[:20]:

            print(
                f"  {name}"
            )

        raise RuntimeError(
            "NaN/Inf model parameters detected."
        )

    passed(
        "All model parameters are finite."
    )


###############################################################################
# 8. Training Components
###############################################################################

def test_training_components(
    model: torch.nn.Module,
) -> None:

    print_section(
        "8. Training Components"
    )

    criterion = build_criterion()

    optimizer = build_training_optimizer(
        model
    )

    scheduler = build_training_scheduler(
        optimizer,
        total_steps=1,
    )

    print(
        f"Criterion : "
        f"{criterion}"
    )

    print(
        f"Optimizer : "
        f"{optimizer.__class__.__name__}"
    )

    print(
        f"Scheduler : "
        f"{scheduler.__class__.__name__}"
    )

    print(
        f"Initial LR : "
        f"{optimizer.param_groups[0]['lr']}"
    )

    passed(
        "Current criterion, optimizer and scheduler "
        "interfaces are functional."
    )


###############################################################################
# Main
###############################################################################

def main() -> None:

    print()
    print("=" * 80)
    print(
        "DSTNet Smoke Checkpoint Diagnostic"
    )
    print("=" * 80)

    print(
        f"Project Root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Device       : "
        f"{DEVICE}"
    )

    ###########################################################################
    # 1
    ###########################################################################

    checkpoint = (
        test_checkpoint()
    )

    ###########################################################################
    # 2
    ###########################################################################

    train_root, _ = (
        test_dataset_roots()
    )

    ###########################################################################
    # 3
    ###########################################################################

    batch = test_dataloader(
        train_root
    )

    ###########################################################################
    # 4
    ###########################################################################

    model = test_model()

    ###########################################################################
    # 5
    ###########################################################################

    test_checkpoint_restore(
        model,
        checkpoint,
    )

    ###########################################################################
    # 6
    ###########################################################################

    test_forward(
        model,
        batch,
    )

    ###########################################################################
    # 7
    ###########################################################################

    test_parameter_sanity(
        model
    )

    ###########################################################################
    # 8
    ###########################################################################

    test_training_components(
        model
    )

    ###########################################################################
    # Complete
    ###########################################################################

    print()
    print("=" * 80)
    print(
        "DSTNet Diagnostic Complete"
    )
    print("=" * 80)

    print()
    print(
        "[PASS] All diagnostic stages completed."
    )

    print()


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":
    main()
