"""
scripts.train

Production training entry point for DSTNet.

Responsibilities
----------------
- Configuration loading
- Dataset construction
- DataLoader construction
- Model construction
- Loss construction
- Optimizer construction
- Scheduler construction
- Training through engine.Trainer
- Validation
- Checkpointing
- Epoch logging

Configuration
-------------
All experiment/runtime parameters are obtained from the repository
configuration system.

Base configuration:
    configs/dataset.yaml
    configs/model.yaml
    configs/training.yaml
    configs/runtime.yaml

Optional overrides:
    configs/paper.yaml
    configs/debug.yaml
    configs/local_cpu.yaml

Current DSTNet model contract
-----------------------------
DSTNet.forward(
    *,
    agent_trajectories,
    map_centerlines,
    positions,
    graph,
    agent_mask=None,
    map_mask=None,
)

``headings`` remains part of the dataset/SceneGraph state contract,
but is not passed as a separate argument at the DSTNet boundary.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


###############################################################################
# Project Imports
###############################################################################

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.cache_manager import CacheManager
from datasets.collate import collate_fn
from datasets.map_loader import MapLoader
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser
from datasets.transforms import (
    build_eval_transform,
    build_train_transform,
)

from engine.optimizer import build_optimizer
from engine.scheduler import build_scheduler
from engine.trainer import Trainer

from losses.total_loss import TotalLoss

from models.dstnet import DSTNet

from utils.config import load_config


###############################################################################
# Configuration Helpers
###############################################################################


def _require_attribute(
    obj: Any,
    path: str,
) -> Any:
    """
    Retrieve a nested configuration attribute.

    Parameters
    ----------
    obj
        Configuration object.

    path
        Dot-separated path, e.g.
        ``"dataset.train_dir"``.

    Returns
    -------
    Any
        Configuration value.

    Raises
    ------
    AttributeError
        If the requested configuration entry does not exist.
    """

    current = obj

    for part in path.split("."):

        if not hasattr(
            current,
            part,
        ):

            raise AttributeError(
                "Required configuration entry "
                f"'{path}' was not found."
            )

        current = getattr(
            current,
            part,
        )

    return current


def _optional_attribute(
    obj: Any,
    paths: tuple[str, ...],
    default: Any,
) -> Any:
    """
    Retrieve the first available configuration entry.

    This is used only where the repository configuration may legitimately
    use one of several established names.

    Parameters
    ----------
    obj
        Configuration object.

    paths
        Candidate dot-separated paths.

    default
        Value returned if none of the paths exists.
    """

    for path in paths:

        current = obj

        found = True

        for part in path.split("."):

            if not hasattr(
                current,
                part,
            ):

                found = False

                break

            current = getattr(
                current,
                part,
            )

        if found:

            return current

    return default


###############################################################################
# Configuration
###############################################################################

CFG = load_config()


###############################################################################
# Reproducibility
###############################################################################


def set_random_seed(
    seed: int,
    deterministic: bool,
) -> None:
    """
    Configure random seeds.

    Parameters
    ----------
    seed
        Global random seed.

    deterministic
        Whether deterministic PyTorch behaviour should be requested.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    if deterministic:

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

    else:

        torch.backends.cudnn.deterministic = False


###############################################################################
# Device
###############################################################################


def resolve_device(
    requested: str,
) -> torch.device:
    """
    Resolve the configured device.

    ``auto`` selects CUDA when available, otherwise CPU.
    """

    requested = str(
        requested
    ).lower()

    if requested == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if requested == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "Configuration requested CUDA, "
                "but CUDA is not available."
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        requested
    )


###############################################################################
# Paths
###############################################################################


def resolve_path(
    path_value: str | Path,
) -> Path:
    """
    Resolve a repository-relative configuration path.

    Absolute paths are returned unchanged.
    """

    path = Path(
        path_value
    )

    if path.is_absolute():

        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


###############################################################################
# Dataset / Preprocessing
###############################################################################


def build_preprocessor() -> ScenePreprocessor:
    """
    Build ScenePreprocessor entirely from dataset configuration.
    """

    observation_steps = int(
        _require_attribute(
            CFG,
            "dataset.observation_steps",
        )
    )

    prediction_steps = int(
        _require_attribute(
            CFG,
            "dataset.prediction_steps",
        )
    )

    map_sample_points = int(
        _require_attribute(
            CFG,
            "dataset.map_sample_points",
        )
    )

    spatial_radius = float(
        _require_attribute(
            CFG,
            "dataset.spatial_radius",
        )
    )

    map_radius = float(
        _require_attribute(
            CFG,
            "dataset.map_radius",
        )
    )

    return ScenePreprocessor(
        observation_steps=observation_steps,
        prediction_steps=prediction_steps,
        map_sample_points=map_sample_points,
        spatial_radius=spatial_radius,
        map_radius=map_radius,
    )


###############################################################################
# Dataset Builder
###############################################################################


def build_dataset(
    root: Path,
    *,
    train: bool,
) -> ArgoverseDataset:
    """
    Construct one Argoverse dataset split.

    All dataset-related paths and preprocessing parameters originate
    from configuration.
    """

    map_root = resolve_path(
        _require_attribute(
            CFG,
            "dataset.root",
        )
    )

    #
    # The dataset root is the repository's Argoverse-1 directory.
    #
    # HD maps are stored below:
    #
    #     data/argoverse1/hd_maps/map_files
    #
    map_root = (
        map_root
        / "hd_maps"
        / "map_files"
    )

    map_loader = MapLoader(
        map_root=map_root,
    )

    parser = SceneParser(
        map_loader,
    )

    preprocessor = build_preprocessor()

    cache_enabled = bool(
        _optional_attribute(
            CFG,
            (
                "cache.enabled",
            ),
            True,
        )
    )

    cache_rebuild = bool(
        _optional_attribute(
            CFG,
            (
                "cache.rebuild",
            ),
            False,
        )
    )

    cache_root = resolve_path(
        _optional_attribute(
            CFG,
            (
                "dataset.cache_dir",
            ),
            "data/argoverse1/cache",
        )
    )

    cache = None

    if cache_enabled:

        cache = CacheManager(
            cache_root,
        )

        #
        # CacheManager currently controls its own cache behaviour.
        # ``cache_rebuild`` is therefore retained here as configuration
        # metadata rather than passed as an unsupported constructor
        # argument.
        #
        _ = cache_rebuild

    transform = (

        build_train_transform()

        if train

        else build_eval_transform()
    )

    return ArgoverseDataset(
        root=root,
        parser=parser,
        preprocessor=preprocessor,
        transform=transform,
        cache=cache,
    )


###############################################################################
# Dataset Split Roots
###############################################################################


def build_dataset_roots() -> tuple[Path, Path]:
    """
    Resolve the configured training and validation dataset roots.
    """

    train_root = resolve_path(
        _require_attribute(
            CFG,
            "dataset.train_dir",
        )
    )

    val_root = resolve_path(
        _require_attribute(
            CFG,
            "dataset.val_dir",
        )
    )

    return (
        train_root,
        val_root,
    )


###############################################################################
# DataLoader
###############################################################################


def build_dataloader(
    dataset: Dataset,
    *,
    train: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Construct a DataLoader from runtime/training configuration.

    ``Dataset`` is used here rather than ``ArgoverseDataset`` because
    framework-validation runs may legitimately pass a ``Subset``.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )


###############################################################################
# Model
###############################################################################


def build_model() -> DSTNet:
    """
    Build DSTNet from model configuration.

    The constructor arguments are explicitly mapped to the current
    DSTNet implementation.
    """

    observation_steps = int(
        _require_attribute(
            CFG,
            "dataset.observation_steps",
        )
    )

    prediction_steps = int(
        _require_attribute(
            CFG,
            "dataset.prediction_steps",
        )
    )

    map_points = int(
        _require_attribute(
            CFG,
            "dataset.map_sample_points",
        )
    )

    hidden_dim = int(
        _require_attribute(
            CFG,
            "model.hidden_dim",
        )
    )

    num_heads = int(
        _require_attribute(
            CFG,
            "model.num_heads",
        )
    )

    num_encoder_layers = int(
        _require_attribute(
            CFG,
            "model.num_encoder_layers",
        )
    )

    num_modes = int(
        _require_attribute(
            CFG,
            "model.num_modes",
        )
    )

    refinement_iterations = int(
        _require_attribute(
            CFG,
            "model.refinement_iterations",
        )
    )

    r_min = float(
        _optional_attribute(
            CFG,
            (
                "model.arp_mspa.r_min",
            ),
            2.0,
        )
    )

    r_max = float(
        _optional_attribute(
            CFG,
            (
                "model.arp_mspa.r_max",
            ),
            30.0,
        )
    )

    if r_max is not None:
        r_max = float(r_max)

    radius_hidden_dim = int(
        _optional_attribute(
            CFG,
            (
                "model.arp_mspa.radius_hidden_dim",
            ),
            256,
        )
    )

    if radius_hidden_dim is not None:
        radius_hidden_dim = int(radius_hidden_dim)

    dropout = float(
        _optional_attribute(
            CFG,
            (
                "model.dropout",
            ),
            0.1,
        )
    )

    model = DSTNet(
        observation_steps=observation_steps,
        prediction_steps=prediction_steps,
        map_points=map_points,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_modes=num_modes,
        refinement_iterations=refinement_iterations,
        r_min=r_min,
        r_max=r_max,
        radius_hidden_dim=radius_hidden_dim,
        dropout=dropout,
    )

    return model


###############################################################################
# Parameter Count
###############################################################################


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total,
        trainable,
    )


###############################################################################
# Loss
###############################################################################


def build_criterion() -> TotalLoss:
    """
    Build TotalLoss from configuration.

    The default values exactly match the current TotalLoss contract:

        proposal       = 1.0
        classification = 1.0
        score          = 0.1
        refinement     = 1.0
    """

    proposal_weight = float(
        _optional_attribute(
            CFG,
            (
                "training.proposal_weight",
                "loss.proposal_weight",
            ),
            1.0,
        )
    )

    classification_weight = float(
        _optional_attribute(
            CFG,
            (
                "training.classification_weight",
                "loss.classification_weight",
            ),
            1.0,
        )
    )

    score_weight = float(
        _optional_attribute(
            CFG,
            (
                "training.score_weight",
                "loss.score_weight",
            ),
            0.1,
        )
    )

    refinement_weight = float(
        _optional_attribute(
            CFG,
            (
                "training.refinement_weight",
                "loss.refinement_weight",
            ),
            1.0,
        )
    )

    return TotalLoss(
        proposal_weight=proposal_weight,
        classification_weight=classification_weight,
        score_weight=score_weight,
        refinement_weight=refinement_weight,
    )


###############################################################################
# Optimizer
###############################################################################


def build_training_optimizer(
    model: DSTNet,
) -> torch.optim.Optimizer:
    """
    Build optimizer from training configuration.
    """

    optimizer_name = str(
        _optional_attribute(
            CFG,
            (
                "training.optimizer",
            ),
            "adamw",
        )
    )

    learning_rate = float(
        _optional_attribute(
            CFG,
            (
                "training.learning_rate",
                "training.lr",
            ),
            1e-4,
        )
    )

    weight_decay = float(
        _optional_attribute(
            CFG,
            (
                "training.weight_decay",
            ),
            1e-2,
        )
    )

    return build_optimizer(
        model=model,
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


###############################################################################
# Scheduler
###############################################################################


def build_training_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
) -> Any:
    """
    Build scheduler from training configuration.

    The scheduler factory receives the total number of optimizer updates
    for the complete training run.
    """

    scheduler_name = str(
        _optional_attribute(
            CFG,
            (
                "training.scheduler",
            ),
            "cosine",
        )
    )

    return build_scheduler(
        optimizer,
        scheduler=scheduler_name,
        total_steps=total_steps,
    )


###############################################################################
# Checkpoint Directory
###############################################################################


def checkpoint_directory() -> Path:
    """
    Resolve configured checkpoint directory.
    """

    configured = _optional_attribute(
        CFG,
        (
            "runtime.checkpoint_dir",
        ),
        "checkpoints",
    )

    return resolve_path(
        configured
    )


###############################################################################
# Main
###############################################################################


def main() -> None:
    """
    Main production training entry point.
    """

    print()
    print("=" * 80)
    print("DSTNet PRODUCTION TRAINING")
    print("=" * 80)

    ###########################################################################
    # Runtime Configuration
    ###########################################################################

    requested_device = str(
        _optional_attribute(
            CFG,
            (
                "runtime.device",
            ),
            "auto",
        )
    )

    device = resolve_device(
        requested_device
    )

    num_workers = int(
        _optional_attribute(
            CFG,
            (
                "runtime.num_workers",
            ),
            0,
        )
    )

    pin_memory = bool(
        _optional_attribute(
            CFG,
            (
                "runtime.pin_memory",
            ),
            device.type == "cuda",
        )
    )

    seed = int(
        _optional_attribute(
            CFG,
            (
                "runtime.seed",
            ),
            42,
        )
    )

    deterministic = bool(
        _optional_attribute(
            CFG,
            (
                "runtime.deterministic",
            ),
            True,
        )
    )

    set_random_seed(
        seed=seed,
        deterministic=deterministic,
    )

    ###########################################################################
    # Training Configuration
    ###########################################################################

    batch_size = int(
        _optional_attribute(
            CFG,
            (
                "training.batch_size",
            ),
            2,
        )
    )

    epochs = int(
        _optional_attribute(
            CFG,
            (
                "training.epochs",
            ),
            30,
        )
    )

    gradient_clip = _optional_attribute(
        CFG,
        (
            "training.gradient_clip",
            "training.gradient_clip_norm",
        ),
        1.0,
    )

    if gradient_clip is not None:

        gradient_clip = float(
            gradient_clip
        )

    ###########################################################################
    # Print Configuration Summary
    ###########################################################################

    print()
    print("Runtime")
    print("-" * 80)

    print(
        f"Device       : {device}"
    )

    print(
        f"Seed         : {seed}"
    )

    print(
        f"Num workers  : {num_workers}"
    )

    print(
        f"Pin memory   : {pin_memory}"
    )

    print()
    print("Training")
    print("-" * 80)

    print(
        f"Batch size   : {batch_size}"
    )

    print(
        f"Epochs       : {epochs}"
    )

    print(
        f"Grad clip    : {gradient_clip}"
    )

    ###########################################################################
    # Dataset Paths
    ###########################################################################

    train_root, val_root = build_dataset_roots()

    if not train_root.exists():

        raise FileNotFoundError(
            f"Training directory does not exist: "
            f"{train_root}"
        )

    if not val_root.exists():

        raise FileNotFoundError(
            f"Validation directory does not exist: "
            f"{val_root}"
        )

    ###########################################################################
    # Dataset Construction
    ###########################################################################

    print()
    print("=" * 80)
    print("BUILDING DATASETS")
    print("=" * 80)

    train_dataset = build_dataset(
        train_root,
        train=True,
    )

    val_dataset = build_dataset(
        val_root,
        train=False,
    )

    print(
        f"Training scenes   : {len(train_dataset):,}"
    )

    print(
        f"Validation scenes : {len(val_dataset):,}"
    )

    ###########################################################################
    # DataLoader Construction
    ###########################################################################

    print()
    print("=" * 80)
    print("BUILDING DATALOADERS")
    print("=" * 80)

    train_loader = build_dataloader(
        train_dataset,
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = build_dataloader(
        val_dataset,
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"Training batches   : {len(train_loader):,}"
    )

    print(
        f"Validation batches : {len(val_loader):,}"
    )

    ###########################################################################
    # Model
    ###########################################################################

    print()
    print("=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)

    model = build_model()

    model.to(
        device
    )

    total_parameters, trainable_parameters = (
        count_parameters(
            model
        )
    )

    print(
        f"Device               : {device}"
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    ###########################################################################
    # Loss
    ###########################################################################

    criterion = build_criterion()

    print()
    print(
        f"Loss : {criterion}"
    )

    ###########################################################################
    # Optimizer
    ###########################################################################

    optimizer = build_training_optimizer(
        model
    )

    print()
    print(
        f"Optimizer : "
        f"{optimizer.__class__.__name__}"
    )

    print(
        f"Learning rate : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    print(
        f"Weight decay  : "
        f"{optimizer.param_groups[0]['weight_decay']:.8f}"
    )

    ###########################################################################
    # Scheduler
    ###########################################################################

    total_training_steps = (
        len(train_loader)
        * epochs
    )

    scheduler = build_training_scheduler(
        optimizer,
        total_steps=total_training_steps,
    )

    print(
        f"Scheduler : "
        f"{scheduler.__class__.__name__}"
    )

    print(
        f"Total optimizer steps : "
        f"{total_training_steps:,}"
    )

    ###########################################################################
    # Checkpoint Directory
    ###########################################################################

    checkpoint_dir = checkpoint_directory()

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Checkpoint directory : "
        f"{checkpoint_dir}"
    )

    ###########################################################################
    # Trainer
    ###########################################################################

    print()
    print("=" * 80)
    print("BUILDING TRAINER")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=str(
            checkpoint_dir
        ),
        gradient_clip=gradient_clip,
    )

    print(
        "Trainer initialized successfully."
    )

    ###########################################################################
    # Training
    ###########################################################################

    print()
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    trainer.fit(
        epochs=epochs
    )

    ###########################################################################
    # Completion
    ###########################################################################

    print()
    print("=" * 80)
    print("DSTNet TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Checkpoint directory : "
        f"{checkpoint_dir}"
    )


###############################################################################
# Entry Point
###############################################################################


if __name__ == "__main__":

    main()
