"""
tests.test_gsta_numerical_stability

Numerical-stability diagnostic for the CURRENT DSTNet GSTA implementation.

IMPORTANT
---------
This diagnostic is synchronized with the current repository APIs.

Current data contract
---------------------
scripts.train.build_dataset(...)
scripts.train.build_dataloader(...)

The current collate_fn returns a DICTIONARY:

    {
        "agent_trajectories": Tensor,
        "future_trajectories": Tensor,
        "map_centerlines": Tensor,
        "positions": Tensor,
        "headings": Tensor,
        "graph": list[SceneGraph],
        "agent_mask": Tensor,
        "map_mask": Tensor,
        "metadata": dict,
    }

Current Encoder -> GSTA contract
---------------------------------
Encoder constructs the relative spatio-temporal embeddings itself:

    relative = Encoder._build_relative_embeddings(...)

and calls:

    GSTA(
        Ea=agent_features,
        Em=lane_features,
        Er=relative,
        scene_graph=graph,
        agent_mask=agent_mask,
        map_mask=lane_mask,
    )

GSTA output:

    (B, N, H, K, D)

Paper reference
---------------
DSTNet:
"Dynamic Trajectory Prediction for Autonomous Vehicles via
Spatio-Temporal Attention"

GSTA corresponds to the paper's Section III-C flow:

    Eq. (3) Temporal Self-Attention
        ->
    Eq. (4) Spatial Self-Attention
        ->
    Eq. (5) Temporal -> Spatial
        ->
    Eq. (6) Spatial -> Temporal
        ->
    Eq. (7) Temporal Learnable Queries
        ->
    Eq. (8) Spatial Learnable Queries
        ->
    Eq. (9) Scene Representation

Scientific note
---------------
The paper does not fully specify every low-level operation needed to
aggregate edge-indexed relative embeddings into node-level features.

The current repository explicitly uses mean aggregation over incident
edges. This diagnostic verifies the CURRENT implementation and does not
claim that every such implementation detail is explicitly specified by
the paper.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor


###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


###############################################################################
# Project Imports
###############################################################################

from scripts import train as train_module


###############################################################################
# Configuration
###############################################################################

DEVICE = torch.device("cpu")

SEED = 42

TEST_SCENES = 2

BATCH_SIZE = 2

NUM_WORKERS = 0


###############################################################################
# Formatting Helpers
###############################################################################


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def subsection(title: str) -> None:
    print()
    print(title)
    print("-" * 80)


def pass_message(message: str) -> None:
    print(f"[PASS] {message}")


def info_message(message: str) -> None:
    print(f"[INFO] {message}")


###############################################################################
# Generic Validation Helpers
###############################################################################


def assert_finite(
    name: str,
    value: Tensor,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if not torch.isfinite(value).all():
        bad = ~torch.isfinite(value)

        bad_count = int(
            bad.sum().item()
        )

        raise FloatingPointError(
            f"{name} contains "
            f"{bad_count} NaN/Inf values."
        )


def tensor_statistics(
    name: str,
    value: Tensor,
) -> None:
    if not isinstance(value, Tensor):
        print(
            f"{name}: "
            f"type={type(value).__name__}"
        )
        return

    finite = bool(
        torch.isfinite(value).all().item()
    )

    print(
        f"{name}: "
        f"shape={tuple(value.shape)}, "
        f"dtype={value.dtype}, "
        f"device={value.device}, "
        f"finite={finite}"
    )

    if value.numel() > 0 and finite:
        print(
            f"  min={value.min().item():.6e}, "
            f"max={value.max().item():.6e}, "
            f"mean={value.mean().item():.6e}, "
            f"abs_max={value.abs().max().item():.6e}"
        )


###############################################################################
# Batch Access
###############################################################################


def get_batch_field(
    batch: dict[str, Any],
    name: str,
) -> Any:
    """
    Retrieve a field from the CURRENT dictionary-based batch contract.
    """

    if not isinstance(
        batch,
        dict,
    ):
        raise TypeError(
            "Current collate_fn contract requires "
            "batch to be a dictionary. "
            f"Got {type(batch).__name__}."
        )

    if name not in batch:
        raise KeyError(
            f"Batch is missing required key '{name}'. "
            f"Available keys: {list(batch.keys())}"
        )

    return batch[name]


###############################################################################
# Dataset Construction
###############################################################################


def build_test_batch() -> dict[str, Any]:
    """
    Construct a real Argoverse batch using the CURRENT train.py API.
    """

    train_root, _ = (
        train_module.build_dataset_roots()
    )

    if not train_root.exists():
        raise FileNotFoundError(
            "Training root does not exist:\n"
            f"{train_root}"
        )

    subsection(
        "Building Diagnostic Dataset"
    )

    print(
        f"Training root : {train_root}"
    )

    dataset = train_module.build_dataset(
        train_root,
        train=True,
    )

    print(
        f"Dataset size  : {len(dataset):,} scenes"
    )

    if len(dataset) < TEST_SCENES:
        raise RuntimeError(
            f"Dataset contains only "
            f"{len(dataset)} scenes."
        )

    from torch.utils.data import Subset

    subset = Subset(
        dataset,
        list(
            range(
                TEST_SCENES
            )
        ),
    )

    print(
        f"Diagnostic subset : "
        f"{len(subset)} scenes"
    )

    loader = train_module.build_dataloader(
        subset,
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    batch = next(
        iter(loader)
    )

    return batch


###############################################################################
# Batch Diagnostics
###############################################################################


def inspect_batch(
    batch: dict[str, Any],
) -> None:
    """
    Verify the current dictionary-based batch contract.
    """

    subsection(
        "Current Batch Contract"
    )

    print(
        f"Batch type : {type(batch).__name__}"
    )

    print(
        "Batch keys :"
    )

    for key in batch.keys():
        print(
            f"  - {key}"
        )

    required = (
        "agent_trajectories",
        "map_centerlines",
        "positions",
        "headings",
        "graph",
        "agent_mask",
        "map_mask",
    )

    for name in required:

        if name not in batch:
            raise KeyError(
                f"Batch is missing required key "
                f"'{name}'."
            )

    tensor_fields = (
        "agent_trajectories",
        "map_centerlines",
        "positions",
        "headings",
        "agent_mask",
        "map_mask",
    )

    for name in tensor_fields:

        value = batch[name]

        tensor_statistics(
            f"batch.{name}",
            value,
        )

        if not isinstance(
            value,
            Tensor,
        ):
            raise TypeError(
                f"batch['{name}'] must be a Tensor."
            )

        assert_finite(
            f"batch.{name}",
            value,
        )

    graph = batch["graph"]

    print(
        f"graph type : {type(graph).__name__}"
    )

    if not isinstance(
        graph,
        Sequence,
    ):
        raise TypeError(
            "Current collate_fn is expected to return "
            "graph as a sequence of SceneGraph objects."
        )

    print(
        f"graph count : {len(graph)}"
    )

    if len(graph) != BATCH_SIZE:
        raise RuntimeError(
            "Graph count does not match batch size. "
            f"Expected {BATCH_SIZE}, got {len(graph)}."
        )

    for index, scene_graph in enumerate(graph):

        print(
            f"  graph[{index}] : "
            f"{type(scene_graph).__name__}"
        )

    pass_message(
        "Real Argoverse batch matches the current dictionary contract."
    )


###############################################################################
# Model Construction
###############################################################################


def build_model() -> torch.nn.Module:
    """
    Build the current DSTNet through scripts.train.build_model().
    """

    subsection(
        "Building Current DSTNet"
    )

    model = train_module.build_model()

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

    pass_message(
        "Current DSTNet constructed successfully."
    )

    return model


###############################################################################
# Encoder Input Construction
###############################################################################


def build_gsta_inputs(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> tuple[
    Tensor,
    Tensor,
    Any,
]:
    """
    Construct EXACTLY the inputs that the current Encoder gives to GSTA.

    This intentionally uses:

        model.agent_encoder
        model.lane_encoder
        model.encoder.relative_embedding

    and does NOT invent a separate relative-embedding API.
    """

    subsection(
        "Building Exact Encoder -> GSTA Inputs"
    )

    agent_trajectories = (
        batch["agent_trajectories"]
        .to(DEVICE)
    )

    map_centerlines = (
        batch["map_centerlines"]
        .to(DEVICE)
    )

    graph = batch["graph"]

    agent_features = (
        model.agent_encoder(
            agent_trajectories
        )
    )

    map_features = (
        model.lane_encoder(
            map_centerlines
        )
    )

    ###########################################################################
    # The CURRENT Encoder constructs Er from SceneGraph.
    ###########################################################################

    relative = (
        model.encoder._build_relative_embeddings(
            graph=graph,
            batch_size=agent_features.shape[0],
        )
    )

    tensor_statistics(
        "Agent embeddings Ea",
        agent_features,
    )

    tensor_statistics(
        "Map embeddings Em",
        map_features,
    )

    assert_finite(
        "Agent embeddings Ea",
        agent_features,
    )

    assert_finite(
        "Map embeddings Em",
        map_features,
    )

    ###########################################################################
    # Relative embedding validation
    ###########################################################################

    if isinstance(
        relative,
        Sequence,
    ):
        print(
            f"Relative embeddings : "
            f"sequence length={len(relative)}"
        )

        if len(relative) != agent_features.shape[0]:
            raise RuntimeError(
                "Relative embedding count does not match "
                "batch size."
            )

        for index, embedding in enumerate(relative):

            inspect_relative_embedding(
                embedding,
                index,
            )

    else:

        inspect_relative_embedding(
            relative,
            0,
        )

    pass_message(
        "Exact current Encoder -> GSTA inputs are finite and constructed."
    )

    return (
        agent_features,
        map_features,
        relative,
    )


###############################################################################
# Relative Embedding Diagnostics
###############################################################################


def inspect_relative_embedding(
    relative: Any,
    index: int,
) -> None:
    """
    Inspect the current RelativeSpatioTemporalEmbedding object without
    assuming undocumented attribute names.
    """

    print()
    print(
        f"Relative embedding [{index}] : "
        f"{type(relative).__name__}"
    )

    candidate_attributes = (
        "edge_embeddings",
        "agent_embeddings",
        "map_embeddings",
        "relative_embeddings",
        "features",
        "edge_index",
        "values",
    )

    found_tensor = False

    for attribute in candidate_attributes:

        if not hasattr(
            relative,
            attribute,
        ):
            continue

        try:
            value = getattr(
                relative,
                attribute,
            )
        except Exception:
            continue

        if isinstance(
            value,
            Tensor,
        ):

            found_tensor = True

            tensor_statistics(
                f"relative.{attribute}",
                value,
            )

            assert_finite(
                f"relative.{attribute}",
                value,
            )

    if not found_tensor:

        print(
            "  No public tensor attribute matched the "
            "diagnostic aliases; GSTA will validate the object "
            "through its actual interface."
        )


###############################################################################
# GSTA Parameter Diagnostics
###############################################################################


def check_gsta_parameters(
    gsta: torch.nn.Module,
) -> None:
    subsection(
        "GSTA Parameter Numerical Sanity"
    )

    total = 0

    for name, parameter in gsta.named_parameters():

        total += parameter.numel()

        if not torch.isfinite(
            parameter
        ).all():

            raise FloatingPointError(
                "GSTA parameter contains NaN/Inf: "
                f"{name}"
            )

    print(
        f"GSTA parameters checked : "
        f"{total:,}"
    )

    pass_message(
        "All GSTA parameters are finite."
    )


###############################################################################
# Direct GSTA Forward
###############################################################################


def run_gsta(
    model: torch.nn.Module,
    batch: dict[str, Any],
    agent_features: Tensor,
    map_features: Tensor,
    relative: Any,
) -> Tensor:
    subsection(
        "Direct GSTA Forward"
    )

    gsta = model.encoder.gsta

    gsta.eval()

    graph = batch["graph"]

    agent_mask = (
        batch["agent_mask"]
        .to(DEVICE)
        .bool()
    )

    map_mask = (
        batch["map_mask"]
        .to(DEVICE)
        .bool()
    )

    with torch.no_grad():

        output = gsta(
            Ea=agent_features,
            Em=map_features,
            Er=relative,
            scene_graph=graph,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )

    expected_shape = (
        agent_features.shape[0],
        agent_features.shape[1],
        agent_features.shape[2],
        gsta.num_modes,
        gsta.hidden_dim,
    )

    print(
        f"Output shape   : "
        f"{tuple(output.shape)}"
    )

    print(
        f"Expected shape : "
        f"{expected_shape}"
    )

    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            "GSTA output shape mismatch."
        )

    assert_finite(
        "GSTA output",
        output,
    )

    print(
        f"Output abs max : "
        f"{output.abs().max().item():.6e}"
    )

    pass_message(
        "Direct GSTA forward is finite and has the expected shape."
    )

    return output


###############################################################################
# Mask Integrity
###############################################################################


def test_mask_integrity(
    model: torch.nn.Module,
    batch: dict[str, Any],
    agent_features: Tensor,
    map_features: Tensor,
    relative: Any,
) -> None:
    """
    Verify that explicitly padded agents/maps do not create NaN/Inf.

    The test does not require every padded output to be exactly zero because
    the current GSTA contract is responsible for numerical safety and
    masking, not for an externally imposed representation convention.
    """

    subsection(
        "Mask / Padding Numerical Stability"
    )

    gsta = model.encoder.gsta

    gsta.eval()

    graphs = batch["graph"]

    agent_mask = (
        batch["agent_mask"]
        .to(DEVICE)
        .bool()
        .clone()
    )

    map_mask = (
        batch["map_mask"]
        .to(DEVICE)
        .bool()
        .clone()
    )

    ###########################################################################
    # Force at least one padded agent/map position if possible.
    #
    # We modify ONLY the diagnostic masks, not the real tensors or graph.
    ###########################################################################

    if agent_mask.shape[1] > 1:

        agent_mask[:, -1] = False

    if map_mask.shape[1] > 1:

        map_mask[:, -1] = False

    with torch.no_grad():

        output = gsta(
            Ea=agent_features,
            Em=map_features,
            Er=relative,
            scene_graph=graphs,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )

    assert_finite(
        "GSTA masked output",
        output,
    )

    print(
        f"Masked output abs max : "
        f"{output.abs().max().item():.6e}"
    )

    ###########################################################################
    # Verify invalid target agents are zero when we explicitly mask them.
    ###########################################################################

    if agent_mask.shape[1] > 1:

        invalid_target = ~agent_mask

        masked_target_values = output[
            invalid_target
        ]

        if masked_target_values.numel() > 0:

            max_invalid = (
                masked_target_values.abs().max().item()
            )

            print(
                f"Masked-agent output abs max : "
                f"{max_invalid:.6e}"
            )

            if max_invalid > 1e-6:

                raise AssertionError(
                    "Current GSTA did not zero the "
                    "explicitly invalid target agents."
                )

    pass_message(
        "GSTA masking remains finite and padded target agents are suppressed."
    )


###############################################################################
# Backward Stability
###############################################################################


def test_backward_stability(
    model: torch.nn.Module,
    batch: dict[str, Any],
    agent_features: Tensor,
    map_features: Tensor,
    relative: Any,
) -> None:
    """
    Run a direct GSTA backward pass and verify finite gradients.
    """

    subsection(
        "GSTA Backward / Gradient Stability"
    )

    gsta = model.encoder.gsta

    gsta.train()

    gsta.zero_grad(
        set_to_none=True
    )

    agent_input = (
        agent_features
        .detach()
        .clone()
        .requires_grad_(True)
    )

    map_input = (
        map_features
        .detach()
        .clone()
        .requires_grad_(True)
    )

    agent_mask = (
        batch["agent_mask"]
        .to(DEVICE)
        .bool()
    )

    map_mask = (
        batch["map_mask"]
        .to(DEVICE)
        .bool()
    )

    output = gsta(
        Ea=agent_input,
        Em=map_input,
        Er=relative,
        scene_graph=batch["graph"],
        agent_mask=agent_mask,
        map_mask=map_mask,
    )

    assert_finite(
        "GSTA training output",
        output,
    )

    ###########################################################################
    # A simple scalar objective is sufficient for numerical gradient testing.
    ###########################################################################

    loss = output.square().mean()

    assert_finite(
        "Diagnostic GSTA loss",
        loss,
    )

    print(
        f"Diagnostic loss : "
        f"{loss.item():.6e}"
    )

    loss.backward()

    ###########################################################################
    # Input gradients
    ###########################################################################

    if agent_input.grad is None:
        raise RuntimeError(
            "No gradient reached GSTA agent input."
        )

    if map_input.grad is None:
        raise RuntimeError(
            "No gradient reached GSTA map input."
        )

    assert_finite(
        "Agent input gradient",
        agent_input.grad,
    )

    assert_finite(
        "Map input gradient",
        map_input.grad,
    )

    ###########################################################################
    # Parameter gradients
    ###########################################################################

    gradient_count = 0

    total_gradient_norm_squared = 0.0

    for name, parameter in gsta.named_parameters():

        if parameter.grad is None:
            continue

        gradient_count += 1

        assert_finite(
            f"GSTA gradient: {name}",
            parameter.grad,
        )

        gradient_norm = (
            parameter.grad.detach()
            .norm()
            .item()
        )

        total_gradient_norm_squared += (
            gradient_norm ** 2
        )

    total_gradient_norm = math.sqrt(
        total_gradient_norm_squared
    )

    print(
        f"Parameters with gradients : "
        f"{gradient_count}"
    )

    print(
        f"Total gradient norm        : "
        f"{total_gradient_norm:.6e}"
    )

    if gradient_count == 0:
        raise RuntimeError(
            "No GSTA parameters received gradients."
        )

    if not math.isfinite(
        total_gradient_norm
    ):
        raise FloatingPointError(
            "GSTA total gradient norm is NaN/Inf."
        )

    pass_message(
        "GSTA backward pass produced finite gradients."
    )


###############################################################################
# Extreme Input Stability
###############################################################################


def test_extreme_inputs(
    model: torch.nn.Module,
    batch: dict[str, Any],
    agent_features: Tensor,
    map_features: Tensor,
    relative: Any,
) -> None:
    """
    Test GSTA against increasingly large but still finite feature values.

    This is deliberately NOT an overflow test with Inf/NaN inputs.
    """

    subsection(
        "Extreme-but-Finite Input Stability"
    )

    gsta = model.encoder.gsta

    gsta.eval()

    agent_mask = (
        batch["agent_mask"]
        .to(DEVICE)
        .bool()
    )

    map_mask = (
        batch["map_mask"]
        .to(DEVICE)
        .bool()
    )

    scales = (
        1.0,
        10.0,
        100.0,
        1000.0,
    )

    for scale in scales:

        scaled_agent = (
            agent_features
            .detach()
            * scale
        )

        scaled_map = (
            map_features
            .detach()
            * scale
        )

        assert_finite(
            f"scaled agent input {scale}",
            scaled_agent,
        )

        assert_finite(
            f"scaled map input {scale}",
            scaled_map,
        )

        with torch.no_grad():

            output = gsta(
                Ea=scaled_agent,
                Em=scaled_map,
                Er=relative,
                scene_graph=batch["graph"],
                agent_mask=agent_mask,
                map_mask=map_mask,
            )

        assert_finite(
            f"GSTA output scale={scale}",
            output,
        )

        print(
            f"Scale {scale:8.1f} | "
            f"Output abs max "
            f"{output.abs().max().item():.6e}"
        )

    pass_message(
        "GSTA remained finite for all tested finite input scales."
    )


###############################################################################
# Repeated Forward Stability
###############################################################################


def test_repeated_forward(
    model: torch.nn.Module,
    batch: dict[str, Any],
    agent_features: Tensor,
    map_features: Tensor,
    relative: Any,
) -> None:
    """
    Verify deterministic evaluation behaviour.
    """

    subsection(
        "Repeated Forward Stability"
    )

    gsta = model.encoder.gsta

    gsta.eval()

    agent_mask = (
        batch["agent_mask"]
        .to(DEVICE)
        .bool()
    )

    map_mask = (
        batch["map_mask"]
        .to(DEVICE)
        .bool()
    )

    outputs: list[Tensor] = []

    with torch.no_grad():

        for iteration in range(3):

            output = gsta(
                Ea=agent_features,
                Em=map_features,
                Er=relative,
                scene_graph=batch["graph"],
                agent_mask=agent_mask,
                map_mask=map_mask,
            )

            assert_finite(
                f"repeated output {iteration + 1}",
                output,
            )

            outputs.append(
                output.detach().clone()
            )

            print(
                f"Iteration {iteration + 1}: "
                f"abs max = "
                f"{output.abs().max().item():.6e}"
            )

    first = outputs[0]

    for index in range(
        1,
        len(outputs),
    ):

        difference = (
            outputs[index]
            - first
        ).abs().max().item()

        print(
            f"Difference run 1 -> run "
            f"{index + 1}: "
            f"{difference:.6e}"
        )

        if difference > 1e-6:

            raise AssertionError(
                "GSTA eval-mode repeated forward "
                "passes are not deterministic."
            )

    pass_message(
        "Repeated GSTA evaluation passes are stable and deterministic."
    )


###############################################################################
# Encoder -> GSTA Boundary
###############################################################################


def test_encoder_gsta_boundary(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> None:
    """
    Verify that the CURRENT Encoder reaches GSTA successfully.

    This deliberately uses Encoder.forward() instead of duplicating the
    relative-embedding construction.
    """

    subsection(
        "Encoder -> GSTA Boundary"
    )

    model.eval()

    agent_trajectories = (
        batch["agent_trajectories"]
        .to(DEVICE)
    )

    map_centerlines = (
        batch["map_centerlines"]
        .to(DEVICE)
    )

    positions = (
        batch["positions"]
        .to(DEVICE)
    )

    agent_mask = (
        batch["agent_mask"]
        .to(DEVICE)
        .bool()
    )

    map_mask = (
        batch["map_mask"]
        .to(DEVICE)
        .bool()
    )

    graph = batch["graph"]

    with torch.no_grad():

        agent_features = (
            model.agent_encoder(
                agent_trajectories
            )
        )

        map_features = (
            model.lane_encoder(
                map_centerlines
            )
        )

        encoder_output = (
            model.encoder(
                agent_features,
                map_features,
                positions,
                graph,
                agent_mask=agent_mask,
                lane_mask=map_mask,
            )
        )

    assert_finite(
        "Encoder output after GSTA/Tri-ATM",
        encoder_output,
    )

    print(
        f"Encoder output shape : "
        f"{tuple(encoder_output.shape)}"
    )

    expected_shape = (
        agent_features.shape[0],
        agent_features.shape[1],
        agent_features.shape[2],
        model.encoder.num_modes,
        model.encoder.hidden_dim,
    )

    print(
        f"Expected shape       : "
        f"{expected_shape}"
    )

    if tuple(encoder_output.shape) != expected_shape:

        raise RuntimeError(
            "Encoder output shape mismatch."
        )

    pass_message(
        "Current Encoder reaches and returns successfully after GSTA."
    )


###############################################################################
# Main
###############################################################################


def main() -> None:

    section(
        "DSTNet GSTA Numerical Stability Diagnostic"
    )

    print(
        f"Project Root : {PROJECT_ROOT}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Seed         : {SEED}"
    )

    ###########################################################################
    # Reproducibility
    ###########################################################################

    train_module.set_random_seed(
        seed=SEED,
        deterministic=True,
    )

    ###########################################################################
    # Build batch
    ###########################################################################

    batch = build_test_batch()

    inspect_batch(
        batch
    )

    ###########################################################################
    # Build model
    ###########################################################################

    model = build_model()

    ###########################################################################
    # Build exact GSTA inputs
    ###########################################################################

    (
        agent_features,
        map_features,
        relative,
    ) = build_gsta_inputs(
        model,
        batch,
    )

    ###########################################################################
    # Parameter sanity
    ###########################################################################

    check_gsta_parameters(
        model.encoder.gsta
    )

    ###########################################################################
    # Direct GSTA
    ###########################################################################

    run_gsta(
        model,
        batch,
        agent_features,
        map_features,
        relative,
    )

    ###########################################################################
    # Mask integrity
    ###########################################################################

    test_mask_integrity(
        model,
        batch,
        agent_features,
        map_features,
        relative,
    )

    ###########################################################################
    # Backward stability
    ###########################################################################

    test_backward_stability(
        model,
        batch,
        agent_features,
        map_features,
        relative,
    )

    ###########################################################################
    # Extreme inputs
    ###########################################################################

    test_extreme_inputs(
        model,
        batch,
        agent_features,
        map_features,
        relative,
    )

    ###########################################################################
    # Repeated inference
    ###########################################################################

    test_repeated_forward(
        model,
        batch,
        agent_features,
        map_features,
        relative,
    )

    ###########################################################################
    # Encoder boundary
    ###########################################################################

    test_encoder_gsta_boundary(
        model,
        batch,
    )

    ###########################################################################
    # Complete
    ###########################################################################

    section(
        "GSTA Diagnostic Complete"
    )

    pass_message(
        "All GSTA numerical-stability diagnostics passed."
    )


###############################################################################
# Entry Point
###############################################################################


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 80)
        print("GSTA DIAGNOSTIC FAILED")
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        traceback.print_exc()

        raise
