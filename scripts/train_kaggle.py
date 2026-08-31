"""
scripts.train_kaggle

Kaggle training entry point for the current DSTNet implementation.

Purpose
-------
Runs production-style training on the Kaggle Argoverse-1 dataset.

The training dataset is already preprocessed into SceneData cache files.
The Kaggle cache is composed from:

    1. The original uploaded cache dataset containing the existing
       preprocessed scenes.

    2. The newly generated cache archive containing the remaining
       preprocessed scenes.

The two cache sources are NOT copied into one giant directory.

Instead:

    - the new cache archive is extracted into /kaggle/working
    - the original uploaded cache remains read-only under /kaggle/input
    - CombinedCacheManager searches both locations

This avoids unnecessarily copying ~129,000 existing cache files.

Current data terminology
------------------------
The current collate pipeline provides:

    agent_trajectories
    future_trajectories
    map_centerlines
    positions
    headings
    agent_mask
    map_mask
    graph

Checkpointing
-------------
A complete local checkpoint is written every epoch.

An optional external backup path is provided below.

Cache
-----
The cache contains processed SceneData objects saved by CacheManager
as pickle files.

The original cache is read directly from Kaggle input.

The additional cache is extracted from its archive into Kaggle working
storage.

DSTNet Reference
----------------
This runner follows the current project implementation of DSTNet:

    Raw Argoverse scene
        -> SceneParser
        -> Transform
        -> ScenePreprocessor
        -> SceneData
        -> CacheManager
        -> ArgoverseDataset
        -> collate_fn
        -> DSTNet

The architecture follows the DSTNet paper:

    "DSTNet: Dynamic Trajectory Prediction for Autonomous Vehicles
     via Spatio-Temporal Attention"

The paper describes the dynamic spatio-temporal attention backbone,
multi-scale spatial/historical attention, multimodal interaction,
and anchor-based trajectory refinement.
"""

from __future__ import annotations

import csv
import os
import pickle
import shutil
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader


###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


###############################################################################
# Current Repository Imports
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

from engine.evaluator import Evaluator
from engine.optimizer import build_optimizer
from engine.scheduler import build_scheduler
from engine.utils import move_to_device

from losses.total_loss import TotalLoss

from models.dstnet import DSTNet


###############################################################################
# Kaggle Dataset Paths
###############################################################################

TRAIN_ROOT = Path(
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_train_v1.1/train/data"
)

VAL_ROOT = Path(
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_val_v1.1/val/data"
)

TEST_ROOT = Path(
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_test_v1.1/test_obs/data"
)

MAP_ROOT = Path(
    "/kaggle/input/datasets/kedaradhikari/"
    "argoverse1-hd-mapss/"
    "hd_maps/map_files"
)


###############################################################################
# Published Cache Dataset
###############################################################################

#
# This is the existing Kaggle dataset containing the original cache.
#
# It currently contains approximately 129,707 preprocessed scenes.
#
###############################################################################

PUBLISHED_CACHE_ROOT = Path(
    "/kaggle/input/datasets/kedaradhikari/"
    "dstnet-training-cache-checkpoints/"
    "cache"
)


###############################################################################
# Working Cache Locations
###############################################################################

#
# New cache archive(s) are extracted here.
#
# This directory is writable.
#
###############################################################################

NEW_CACHE_EXTRACT_ROOT = Path(
    "/kaggle/working/new_cache_extracted"
)


###############################################################################
# Writable Cache
###############################################################################

#
# This is used for:
#
#     - extracted additional cache
#     - any scenes that need to be generated during training
#     - VERSION file
#
###############################################################################

CACHE_ROOT = Path(
    "/kaggle/working/cache"
)

VAL_CACHE_ROOT = Path(
    "/kaggle/working/val_cache"
)


###############################################################################
# Kaggle Working Directories
###############################################################################

CHECKPOINT_ROOT = Path(
    "/kaggle/working/checkpoints"
)

LOG_ROOT = Path(
    "/kaggle/working/logs"
)

RESULTS_ROOT = Path(
    "/kaggle/working/results"
)


###############################################################################
# External Checkpoint Backup
###############################################################################

EXTERNAL_CHECKPOINT_ROOT = Path(
    "/YOUR/EXTERNAL/MOUNT/DSTNet/checkpoints"
)

EXTERNAL_CHECKPOINT_ENABLED = False

EXTERNAL_SAVE_EVERY = 5


###############################################################################
# Directories
###############################################################################

CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

VAL_CACHE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


###############################################################################
# Training Configuration
###############################################################################

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if torch.cuda.is_available():

    torch.backends.cudnn.benchmark = True


BATCH_SIZE = 3

NUM_WORKERS = 2

EPOCHS = 30

LEARNING_RATE = 1e-6

WEIGHT_DECAY = 1e-2

SAVE_EVERY = 1

GRADIENT_CLIP = 0.5

VALIDATE_EVERY = 5

REFINEMENT_ENABLED = False


###############################################################################
# Mixed Precision
###############################################################################

USE_AMP = False

AMP_DTYPE = torch.float32

LOG_EVERY = 1000


###############################################################################
# Early Stopping
###############################################################################

EARLY_STOPPING = False

PATIENCE = 3


###############################################################################
# Checkpoint Paths
###############################################################################

LATEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "latest.pth"
)

BEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "best_model.pth"
)

CSV_LOG = (
    LOG_ROOT
    / "training_log.csv"
)


###############################################################################
# Printing Utilities
###############################################################################


def print_header(
    title: str,
) -> None:

    print()

    print("=" * 80)

    print(title)

    print("=" * 80)


def print_section(
    title: str,
) -> None:

    print()

    print(title)

    print("-" * 80)


###############################################################################
# Cache Utilities
###############################################################################


def is_cache_file(
    path: Path,
) -> bool:

    return (
        path.is_file()
        and path.suffix.lower() == ".pkl"
    )


def find_tar_archives(
    root: Path,
) -> list[Path]:
    """
    Find cache archives recursively.

    Supported:

        .tar
        .tar.gz
        .tgz
        .tar.bz2
        .tbz2
        .tar.xz
        .txz
    """

    if not root.exists():

        return []

    suffixes = (
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tbz2",
        ".tar.xz",
        ".txz",
    )

    archives = []

    for path in root.rglob("*"):

        if not path.is_file():

            continue

        name = path.name.lower()

        if name.endswith(suffixes):

            archives.append(path)

    return sorted(
        archives
    )


def validate_archive_member(
    member: tarfile.TarInfo,
    destination: Path,
) -> Path:
    """
    Prevent path traversal while extracting tar archives.
    """

    target = (
        destination
        / member.name
    ).resolve()

    destination_resolved = (
        destination.resolve()
    )

    try:

        target.relative_to(
            destination_resolved
        )

    except ValueError:

        raise RuntimeError(
            "Unsafe tar archive member detected: "
            f"{member.name}"
        )

    return target


def extract_cache_archive(
    archive: Path,
    destination: Path,
) -> None:
    """
    Extract one cache archive safely.
    """

    print()

    print(
        f"Extracting cache archive:"
    )

    print(
        f"    {archive}"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tarfile.open(
        archive,
        mode="r:*",
    ) as tar:

        members = tar.getmembers()

        print(
            f"Archive members : "
            f"{len(members):,}"
        )

        for member in members:

            validate_archive_member(
                member,
                destination,
            )

        tar.extractall(
            destination,
        )

    print(
        "Archive extraction complete."
    )


def find_cache_directories(
    root: Path,
) -> list[Path]:
    """
    Find directories containing CacheManager-style .pkl files.

    A cache directory is expected to contain files such as:

        123.pkl
        456.pkl
        789.pkl
    """

    if not root.exists():

        return []

    candidate_counts: dict[Path, int] = {}

    for path in root.rglob("*.pkl"):

        if not path.is_file():

            continue

        parent = path.parent

        candidate_counts[parent] = (
            candidate_counts.get(
                parent,
                0,
            )
            + 1
        )

    return [
        directory
        for directory, count
        in sorted(
            candidate_counts.items(),
            key=lambda item: (
                -item[1],
                str(item[0]),
            ),
        )
        if count > 0
    ]


def prepare_additional_cache() -> list[Path]:
    """
    Locate and extract additional cache archives.

    Returns
    -------
    list[Path]
        Directories containing extracted .pkl cache files.
    """

    print_section(
        "Preparing Additional Cache"
    )

    if not PUBLISHED_CACHE_ROOT.exists():

        raise FileNotFoundError(
            "Published cache dataset does not exist:\n"
            f"{PUBLISHED_CACHE_ROOT}"
        )

    print(
        f"Published cache : "
        f"{PUBLISHED_CACHE_ROOT}"
    )

    archives = find_tar_archives(
        PUBLISHED_CACHE_ROOT
    )

    if not archives:

        print(
            "No cache archive found."
        )

        print(
            "The published dataset will be used "
            "as the cache source."
        )

        return []

    print(
        f"Cache archives found : "
        f"{len(archives):,}"
    )

    NEW_CACHE_EXTRACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for archive in archives:

        archive_marker = (
            NEW_CACHE_EXTRACT_ROOT
            / (
                archive.name
                + ".extracted"
            )
        )

        if archive_marker.exists():

            print()

            print(
                f"Already extracted : "
                f"{archive.name}"
            )

            continue

        extract_cache_archive(
            archive,
            NEW_CACHE_EXTRACT_ROOT,
        )

        archive_marker.write_text(
            "extracted",
            encoding="utf-8",
        )

    cache_directories = (
        find_cache_directories(
            NEW_CACHE_EXTRACT_ROOT
        )
    )

    if not cache_directories:

        raise RuntimeError(
            "Cache archive(s) were found and extracted, "
            "but no .pkl cache files could be located."
        )

    print()

    print(
        "Extracted cache directories:"
    )

    for directory in cache_directories:

        count = len(
            list(
                directory.glob("*.pkl")
            )
        )

        print(
            f"    {directory} "
            f"({count:,} files)"
        )

    return cache_directories


###############################################################################
# Combined Cache Manager
###############################################################################


class CombinedCacheManager:
    """
    Cache manager that combines multiple cache roots.

    Search order
    ------------
    1. Writable primary cache roots
    2. Additional extracted cache roots
    3. Original read-only Kaggle cache

    The first matching sequence ID is loaded.

    New cache files are saved only into the first writable root.
    """

    def __init__(
        self,
        *,
        writable_root: Path,
        additional_roots: list[Path],
        fallback_root: Path,
    ) -> None:

        self.writable_root = Path(
            writable_root
        )

        self.additional_roots = [
            Path(root)
            for root in additional_roots
        ]

        self.fallback_root = Path(
            fallback_root
        )

        self.writable_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.version_file = (
            self.writable_root
            / "VERSION"
        )

        if not self.version_file.exists():

            self.version_file.write_text(
                "1.0",
                encoding="utf-8",
            )

    ###########################################################################
    # Cache path
    ###########################################################################

    def cache_path(
        self,
        sequence_id: str,
    ) -> Path:

        return (
            self.writable_root
            / f"{sequence_id}.pkl"
        )

    ###########################################################################
    # Candidate paths
    ###########################################################################

    def candidate_paths(
        self,
        sequence_id: str,
    ) -> list[Path]:

        filename = (
            f"{sequence_id}.pkl"
        )

        paths = [
            self.writable_root
            / filename
        ]

        paths.extend(
            root / filename
            for root
            in self.additional_roots
        )

        paths.append(
            self.fallback_root
            / filename
        )

        return paths

    ###########################################################################
    # Exists
    ###########################################################################

    def exists(
        self,
        sequence_id: str,
    ) -> bool:

        for path in self.candidate_paths(
            sequence_id
        ):

            if path.is_file():

                return True

        return False

    ###########################################################################
    # Load
    ###########################################################################

    def load(
        self,
        sequence_id: str,
    ):
        """
        Load a cached SceneData object.

        The search order is:

            writable cache
            additional extracted cache
            original cache
        """

        candidate_paths = (
            self.candidate_paths(
                sequence_id
            )
        )

        for path in candidate_paths:

            if not path.is_file():

                continue

            try:

                with open(
                    path,
                    "rb",
                ) as file:

                    scene = pickle.load(
                        file
                    )

            except (
                EOFError,
                pickle.UnpicklingError,
            ):

                print()

                print(
                    f"[Cache] Corrupted cache detected: "
                    f"{path}"
                )

                if path.parent == self.writable_root:

                    path.unlink(
                        missing_ok=True
                    )

                raise RuntimeError(
                    "Corrupted cache detected for "
                    f"{sequence_id}."
                )

            return scene

        raise FileNotFoundError(
            "Cached scene not found: "
            f"{sequence_id}"
        )

    ###########################################################################
    # Save
    ###########################################################################

    def save(
        self,
        scene,
    ) -> None:

        path = (
            self.writable_root
            / f"{scene.sequence_id}.pkl"
        )

        temp_path = (
            self.writable_root
            / f"{scene.sequence_id}.tmp"
        )

        with open(
            temp_path,
            "wb",
        ) as file:

            pickle.dump(
                scene,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

            file.flush()

            os.fsync(
                file.fileno()
            )

        temp_path.replace(
            path
        )

    ###########################################################################
    # Count
    ###########################################################################

    def num_cached(
        self,
    ) -> int:

        sequence_ids = set()

        for root in [
            self.writable_root,
            *self.additional_roots,
            self.fallback_root,
        ]:

            if not root.exists():

                continue

            for path in root.glob(
                "*.pkl"
            ):

                sequence_ids.add(
                    path.stem
                )

        return len(
            sequence_ids
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "CombinedCacheManager("
            f"writable_root='{self.writable_root}', "
            f"additional_roots={self.additional_roots}, "
            f"fallback_root='{self.fallback_root}')"
        )


###############################################################################
# Parameter Counter
###############################################################################


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return (
        total,
        trainable,
    )


###############################################################################
# Scene Preprocessor
###############################################################################


def build_preprocessor() -> ScenePreprocessor:
    """
    Current ScenePreprocessor configuration.

    These values match the preprocessing used to generate the cache:

        observation_steps = 20
        prediction_steps = 30
        map_sample_points = 20
        spatial_radius = 30m
        map_radius = 30m
    """

    return ScenePreprocessor(

        observation_steps=20,

        prediction_steps=30,

        map_sample_points=20,

        spatial_radius=30.0,

        map_radius=30.0,
    )


###############################################################################
# Dataset
###############################################################################


def build_dataset(
    root: Path,
    *,
    train: bool,
    cache_manager,
) -> ArgoverseDataset:
    """
    Build one Argoverse-1 split.

    Cached SceneData is loaded before raw parsing/preprocessing.
    """

    if not root.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{root}"
        )

    if not MAP_ROOT.exists():

        raise FileNotFoundError(
            f"HD map directory does not exist: "
            f"{MAP_ROOT}"
        )

    map_loader = MapLoader(
        map_root=MAP_ROOT,
    )

    parser = SceneParser(
        map_loader,
    )

    preprocessor = build_preprocessor()

    transform = (
        build_train_transform()
        if train
        else build_eval_transform()
    )

    dataset = ArgoverseDataset(

        root=root,

        parser=parser,

        preprocessor=preprocessor,

        transform=transform,

        cache=cache_manager,
    )

    return dataset


###############################################################################
# DataLoader
###############################################################################


def build_dataloader(
    dataset: ArgoverseDataset,
    *,
    train: bool,
) -> DataLoader:

    kwargs = {

        "dataset":
            dataset,

        "batch_size":
            BATCH_SIZE,

        "shuffle":
            train,

        "num_workers":
            NUM_WORKERS,

        "collate_fn":
            collate_fn,

        "pin_memory":
            DEVICE.type == "cuda",

        "drop_last":
            False,
    }

    if NUM_WORKERS > 0:

        kwargs[
            "persistent_workers"
        ] = True

        kwargs[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        **kwargs,
    )


###############################################################################
# Model
###############################################################################


def build_model() -> DSTNet:

    model = DSTNet(
        refinement_enabled=REFINEMENT_ENABLED,
    )

    model.to(
        DEVICE
    )

    total, trainable = (
        count_parameters(
            model
        )
    )

    print_section(
        "Model"
    )

    print(
        f"Device               : "
        f"{DEVICE}"
    )

    print(
        f"Total Parameters     : "
        f"{total:,}"
    )

    print(
        f"Trainable Parameters : "
        f"{trainable:,}"
    )

    return model


###############################################################################
# Training Components
###############################################################################


def build_training_components(
    model: DSTNet,
    total_steps: int,
) -> tuple:

    optimizer = build_optimizer(

        model=model,

        optimizer="adamw",

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(

        optimizer,

        scheduler="cosine",

        total_steps=(
            total_steps
            * EPOCHS
        ),
    )

    ###########################################################################
    # IMPORTANT
    #
    # Current TotalLoss does NOT take refinement_enabled.
    #
    # Refinement is controlled by the DSTNet configuration.
    #
    ###########################################################################

    criterion = TotalLoss()

    return (
        optimizer,
        scheduler,
        criterion,
    )


###############################################################################
# Directory Creation
###############################################################################


def create_directories() -> None:

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    VAL_CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


###############################################################################
# Checkpoint State
###############################################################################


def build_checkpoint_state(
    *,
    epoch: int,
    model: DSTNet,
    optimizer,
    scheduler,
    train_loss: float,
    val_metrics: dict[str, float],
) -> dict:

    return {

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "val_metrics":
            val_metrics,

        "best_metric":
            val_metrics.get(
                "minADE",
                float("inf"),
            ),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

        "torch_rng_state":
            torch.get_rng_state(),

        "cuda_rng_state_all":
            (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
    }


###############################################################################
# Save Local Checkpoint
###############################################################################


def save_checkpoint(
    *,
    epoch: int,
    model: DSTNet,
    optimizer,
    scheduler,
    train_loss: float,
    val_metrics: dict[str, float],
    best: bool = False,
) -> Path:

    checkpoint = build_checkpoint_state(

        epoch=epoch,

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        train_loss=train_loss,

        val_metrics=val_metrics,
    )

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT,
    )

    if best:

        torch.save(
            checkpoint,
            BEST_CHECKPOINT,
        )

    print()

    print(
        f"Checkpoint saved : "
        f"{LATEST_CHECKPOINT}"
    )

    if best:

        print(
            f"Best model saved : "
            f"{BEST_CHECKPOINT}"
        )

    return LATEST_CHECKPOINT


###############################################################################
# External Checkpoint Backup
###############################################################################


def backup_checkpoint_externally(
    *,
    epoch: int,
    best: bool,
) -> None:

    if not EXTERNAL_CHECKPOINT_ENABLED:

        return

    if (
        epoch
        % EXTERNAL_SAVE_EVERY
        != 0
    ):

        return

    if str(
        EXTERNAL_CHECKPOINT_ROOT
    ).startswith(
        "/YOUR/EXTERNAL/"
    ):

        raise RuntimeError(

            "External checkpointing is enabled, "
            "but EXTERNAL_CHECKPOINT_ROOT still "
            "contains the placeholder path."
        )

    EXTERNAL_CHECKPOINT_ROOT.mkdir(

        parents=True,

        exist_ok=True,
    )

    external_epoch_path = (

        EXTERNAL_CHECKPOINT_ROOT

        / f"epoch_{epoch:04d}.pth"
    )

    shutil.copy2(

        LATEST_CHECKPOINT,

        external_epoch_path,
    )

    external_latest_path = (

        EXTERNAL_CHECKPOINT_ROOT

        / "latest_external.pth"
    )

    shutil.copy2(

        LATEST_CHECKPOINT,

        external_latest_path,
    )

    if (
        best
        and BEST_CHECKPOINT.exists()
    ):

        external_best_path = (

            EXTERNAL_CHECKPOINT_ROOT

            / "best_model.pth"
        )

        shutil.copy2(

            BEST_CHECKPOINT,

            external_best_path,
        )

    print()

    print(
        f"External checkpoint backup : "
        f"{external_epoch_path}"
    )


###############################################################################
# Resume
###############################################################################


def load_checkpoint(
    model: DSTNet,
    optimizer,
    scheduler,
) -> tuple[int, float]:

    if not LATEST_CHECKPOINT.exists():

        print_section(
            "Checkpoint"
        )

        print(
            "No checkpoint found."
        )

        return (
            0,
            float("inf"),
        )

    print_section(
        "Resuming Training"
    )

    checkpoint = torch.load(

        LATEST_CHECKPOINT,

        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler_state = checkpoint.get(
        "scheduler_state_dict",
        None,
    )

    if (
        scheduler is not None
        and scheduler_state is not None
    ):

        scheduler.load_state_dict(
            scheduler_state
        )

    ###########################################################################
    # CPU RNG
    ###########################################################################

    try:

        torch_rng_state = checkpoint.get(
            "torch_rng_state",
            checkpoint.get(
                "rng_state"
            ),
        )

        if isinstance(
            torch_rng_state,
            torch.Tensor,
        ):

            if (
                torch_rng_state.dtype
                == torch.uint8
            ):

                torch.set_rng_state(
                    torch_rng_state.cpu()
                )

    except Exception as exc:

        print(
            f"Warning: CPU RNG state could not be restored: "
            f"{exc}"
        )

    ###########################################################################
    # CUDA RNG
    ###########################################################################

    if torch.cuda.is_available():

        try:

            cuda_rng_state = checkpoint.get(
                "cuda_rng_state_all",
                checkpoint.get(
                    "cuda_rng_states"
                ),
            )

            if cuda_rng_state is not None:

                valid_cuda_states = []

                for state in cuda_rng_state:

                    if isinstance(
                        state,
                        torch.Tensor,
                    ):

                        state = state.cpu()

                        if (
                            state.dtype
                            == torch.uint8
                        ):

                            valid_cuda_states.append(
                                state
                            )

                if (
                    len(valid_cuda_states)
                    == torch.cuda.device_count()
                ):

                    torch.cuda.set_rng_state_all(
                        valid_cuda_states
                    )

                else:

                    print(
                        "Warning: CUDA RNG state is "
                        "incompatible with the current "
                        "CUDA configuration. Skipping "
                        "CUDA RNG restoration."
                    )

        except Exception as exc:

            print(
                f"Warning: CUDA RNG state could not be restored: "
                f"{exc}"
            )

    epoch = int(
        checkpoint.get(
            "epoch",
            0,
        )
    )

    val_metrics = checkpoint.get(
        "val_metrics",
        {},
    )

    best_metric = float(
        checkpoint.get(
            "best_metric",
            val_metrics.get(
                "minADE",
                float("inf"),
            ),
        )
    )

    print(
        f"Resumed from epoch "
        f"{epoch}"
    )

    print(
        f"Best minADE : "
        f"{best_metric:.6f}"
    )

    return (
        epoch,
        best_metric,
    )


###############################################################################
# CSV Logging
###############################################################################

CSV_HEADER = [

    "epoch",

    "train_loss",

    "minADE",

    "minFDE",

    "MissRate",

    "learning_rate",

    "epoch_time",
]


def initialize_csv() -> None:

    if CSV_LOG.exists():

        return

    with open(

        CSV_LOG,

        "w",

        newline="",

        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            CSV_HEADER
        )


def append_csv(
    *,
    epoch: int,
    train_loss: float,
    metrics: dict[str, float],
    learning_rate: float,
    epoch_time: float,
) -> None:

    with open(

        CSV_LOG,

        "a",

        newline="",

        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(

            [

                epoch,

                train_loss,

                metrics.get(
                    "minADE",
                    0.0,
                ),

                metrics.get(
                    "minFDE",
                    0.0,
                ),

                metrics.get(
                    "MissRate",
                    0.0,
                ),

                learning_rate,

                epoch_time,
            ]
        )


###############################################################################
# Training
###############################################################################


def train_one_epoch(
    *,
    epoch: int,
    model: DSTNet,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    criterion: TotalLoss,
    scaler: GradScaler,
) -> float:

    model.train()

    running_loss = 0.0

    num_batches = len(
        dataloader
    )

    if num_batches == 0:

        raise RuntimeError(
            "Training DataLoader contains zero batches."
        )

    epoch_start = (
        time.perf_counter()
    )

    skipped_batches = 0

    for batch_index, batch in enumerate(

        dataloader,

        start=1,
    ):

        batch_start = (
            time.perf_counter()
        )

        #######################################################################
        # Move batch
        #######################################################################

        batch = move_to_device(
            batch,
            DEVICE,
        )

        #######################################################################
        # Clear gradients
        #######################################################################

        optimizer.zero_grad(
            set_to_none=True,
        )

        #######################################################################
        # Forward + loss
        #######################################################################

        if USE_AMP:

            with autocast(

                device_type=DEVICE.type,

                dtype=AMP_DTYPE,

                enabled=True,
            ):

                (
                    coarse_prediction,
                    refined_prediction,
                ) = model(

                    agent_trajectories=(
                        batch[
                            "agent_trajectories"
                        ]
                    ),

                    map_centerlines=(
                        batch[
                            "map_centerlines"
                        ]
                    ),

                    positions=(
                        batch[
                            "positions"
                        ]
                    ),

                    headings=(
                        batch[
                            "headings"
                        ]
                    ),

                    graph=(
                        batch[
                            "graph"
                        ]
                    ),

                    agent_mask=batch.get(
                        "agent_mask"
                    ),

                    map_mask=batch.get(
                        "map_mask"
                    ),
                )

                losses = criterion(

                    prediction=(
                        coarse_prediction
                    ),

                    refined_prediction=(
                        refined_prediction
                    ),

                    ground_truth=(
                        batch[
                            "future_trajectories"
                        ]
                    ),
                )

                loss = losses[
                    "loss"
                ]

        else:

            (
                coarse_prediction,
                refined_prediction,
            ) = model(

                agent_trajectories=(
                    batch[
                        "agent_trajectories"
                    ]
                ),

                map_centerlines=(
                    batch[
                        "map_centerlines"
                    ]
                ),

                positions=(
                    batch[
                        "positions"
                    ]
                ),

                headings=(
                    batch[
                        "headings"
                    ]
                ),

                graph=(
                    batch[
                        "graph"
                    ]
                ),

                agent_mask=batch.get(
                    "agent_mask"
                ),

                map_mask=batch.get(
                    "map_mask"
                ),
            )

            losses = criterion(

                prediction=(
                    coarse_prediction
                ),

                refined_prediction=(
                    refined_prediction
                ),

                ground_truth=(
                    batch[
                        "future_trajectories"
                    ]
                ),
            )

            loss = losses[
                "loss"
            ]

        #######################################################################
        # Loss sanity
        #######################################################################

        if not torch.isfinite(
            loss
        ).all():

            print()

            print(
                "=" * 80
            )

            print(
                "NON-FINITE LOSS DETECTED"
            )

            print(
                "=" * 80
            )

            print(
                f"Epoch : {epoch}"
            )

            print(
                f"Batch : {batch_index}"
            )

            print(
                f"Loss  : "
                f"{loss.detach().item()}"
            )

            raise FloatingPointError(
                "Non-finite loss detected."
            )

        #######################################################################
        # Backward
        #######################################################################

        if USE_AMP:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

        else:

            loss.backward()

        #######################################################################
        # Gradient diagnostics
        #######################################################################

        nonfinite_gradients = []

        max_gradient_value = 0.0

        for name, parameter in (
            model.named_parameters()
        ):

            if parameter.grad is None:

                continue

            gradient = parameter.grad

            if not torch.isfinite(
                gradient
            ).all():

                nonfinite_gradients.append(
                    name
                )

                continue

            current_max = (
                gradient.detach()
                .abs()
                .max()
                .item()
            )

            max_gradient_value = max(
                max_gradient_value,
                current_max,
            )

        #######################################################################
        # Invalid gradient handling
        #######################################################################

        if nonfinite_gradients:

            print()

            print(
                "=" * 80
            )

            print(
                "NON-FINITE GRADIENTS — "
                "SKIPPING BATCH"
            )

            print(
                "=" * 80
            )

            print(
                f"Epoch : {epoch}"
            )

            print(
                f"Batch : {batch_index}"
            )

            print(
                f"Loss  : "
                f"{loss.detach().item():.8f}"
            )

            print(
                f"Affected parameters : "
                f"{len(nonfinite_gradients)}"
            )

            for name in (
                nonfinite_gradients[:10]
            ):

                print(
                    f"  {name}"
                )

            if (
                len(nonfinite_gradients)
                > 10
            ):

                print(
                    f"  ... and "
                    f"{len(nonfinite_gradients) - 10} more"
                )

            if USE_AMP:

                scaler.update()

            optimizer.zero_grad(
                set_to_none=True,
            )

            skipped_batches += 1

            continue

        #######################################################################
        # Gradient clipping
        #######################################################################

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP,
                error_if_nonfinite=True,
            )
        )

        #######################################################################
        # Optimizer update
        #######################################################################

        if USE_AMP:

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            optimizer.step()

        #######################################################################
        # Scheduler
        #######################################################################

        if scheduler is not None:

            scheduler.step()

        #######################################################################
        # Statistics
        #######################################################################

        loss_value = (
            loss.detach().item()
        )

        running_loss += (
            loss_value
        )

        batch_time = (
            time.perf_counter()
            - batch_start
        )

        #######################################################################
        # Logging
        #######################################################################

        if (

            batch_index == 1

            or batch_index % LOG_EVERY == 0

            or batch_index == num_batches
        ):

            print(

                f"Epoch {epoch:03d} "

                f"[{batch_index:06d}/"
                f"{num_batches:06d}] "

                f"loss={loss_value:.6f} "

                f"grad={float(gradient_norm):.4f} "

                f"maxgrad={max_gradient_value:.4e} "

                f"lr="
                f"{optimizer.param_groups[0]['lr']:.8e} "

                f"time={batch_time:.2f}s",

                flush=True,
            )

    ###########################################################################
    # Epoch statistics
    ###########################################################################

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    average_loss = (
        running_loss
        / num_batches
    )

    print()

    print(
        f"Training Epoch {epoch} "
        f"Loss : {average_loss:.6f}",
        flush=True,
    )

    print(
        f"Training Epoch Time : "
        f"{epoch_time:.2f} s",
        flush=True,
    )

    print(
        f"Skipped Non-Finite Batches : "
        f"{skipped_batches}",
        flush=True,
    )

    return average_loss


###############################################################################
# Validation
###############################################################################


def validate_one_epoch(
    *,
    model: DSTNet,
    dataloader: DataLoader,
) -> dict[str, float]:

    print()

    print("=" * 80)

    print(
        "Validation"
    )

    print("=" * 80)

    evaluator = Evaluator(

        model=model,

        dataloader=dataloader,

        device=DEVICE,
    )

    start_time = (
        time.perf_counter()
    )

    metrics = evaluator.evaluate()

    validation_time = (
        time.perf_counter()
        - start_time
    )

    print()

    print(
        "-" * 80
    )

    print(
        "Validation Metrics"
    )

    print(
        "-" * 80
    )

    for key in sorted(
        metrics.keys()
    ):

        print(

            f"{key:<15}"

            f"{metrics[key]:.6f}"
        )

    print()

    print(
        f"Validation Time : "
        f"{validation_time:.2f} s"
    )

    required_metrics = (

        "minADE",

        "minFDE",

        "MissRate",
    )

    for metric_name in (
        required_metrics
    ):

        if metric_name not in metrics:

            raise RuntimeError(

                f"Missing validation metric "
                f"'{metric_name}'."
            )

        value = metrics[
            metric_name
        ]

        if not torch.isfinite(

            torch.tensor(

                value,

                dtype=torch.float64,
            )
        ):

            raise RuntimeError(

                f"Metric '{metric_name}' "
                "is NaN or Inf."
            )

    return metrics


###############################################################################
# Epoch Summary
###############################################################################


def print_epoch_summary(
    *,
    epoch: int,
    train_loss: float,
    metrics: dict[str, float],
    learning_rate: float,
    epoch_time: float,
) -> None:

    print()

    print("=" * 80)

    print(
        f"Epoch {epoch} Summary"
    )

    print("=" * 80)

    print(
        f"Train Loss : "
        f"{train_loss:.6f}"
    )

    print(
        f"minADE     : "
        f"{metrics.get('minADE', float('nan')):.6f}"
    )

    print(
        f"minFDE     : "
        f"{metrics.get('minFDE', float('nan')):.6f}"
    )

    print(
        f"MissRate   : "
        f"{metrics.get('MissRate', float('nan')):.6f}"
    )

    print(
        f"Learning Rate : "
        f"{learning_rate:.8e}"
    )

    print(
        f"Epoch Time   : "
        f"{epoch_time:.2f} s"
    )


###############################################################################
# Cache Diagnostics
###############################################################################


def print_cache_summary(
    cache_manager: CombinedCacheManager,
    additional_roots: list[Path],
) -> None:

    print_section(
        "Cache Summary"
    )

    original_count = 0

    if PUBLISHED_CACHE_ROOT.exists():

        original_count = len(
            list(
                PUBLISHED_CACHE_ROOT.glob(
                    "*.pkl"
                )
            )
        )

    additional_count = 0

    for root in additional_roots:

        additional_count += len(
            list(
                root.glob("*.pkl")
            )
        )

    working_count = len(
        list(
            CACHE_ROOT.glob("*.pkl")
        )
    )

    combined_count = (
        cache_manager.num_cached()
    )

    print(
        f"Original cache files : "
        f"{original_count:,}"
    )

    print(
        f"Additional cache files : "
        f"{additional_count:,}"
    )

    print(
        f"Working cache files : "
        f"{working_count:,}"
    )

    print(
        f"Unique combined scenes : "
        f"{combined_count:,}"
    )

    print()

    print(
        f"Original cache root : "
        f"{PUBLISHED_CACHE_ROOT}"
    )

    for index, root in enumerate(
        additional_roots,
        start=1,
    ):

        print(
            f"Additional cache {index} : "
            f"{root}"
        )

    print(
        f"Writable cache root : "
        f"{CACHE_ROOT}"
    )


###############################################################################
# Main Training Pipeline
###############################################################################


def run_training() -> None:

    create_directories()

    initialize_csv()

    ###########################################################################
    # Prepare cache
    ###########################################################################

    additional_cache_roots = (
        prepare_additional_cache()
    )

    train_cache = CombinedCacheManager(

        writable_root=CACHE_ROOT,

        additional_roots=additional_cache_roots,

        fallback_root=PUBLISHED_CACHE_ROOT,
    )

    print_cache_summary(

        train_cache,

        additional_cache_roots,
    )

    ###########################################################################
    # Datasets
    ###########################################################################

    print_section(
        "Building Datasets"
    )

    print(
        "IMPORTANT:"
    )

    print(
        "Training scenes are loaded from the "
        "preprocessed cache whenever available."
    )

    train_dataset = build_dataset(

        TRAIN_ROOT,

        train=True,

        cache_manager=train_cache,
    )

    ###########################################################################
    # Validation gets its own writable cache.
    ###########################################################################

    val_cache = CacheManager(
        VAL_CACHE_ROOT
    )

    val_dataset = build_dataset(

        VAL_ROOT,

        train=False,

        cache_manager=val_cache,
    )

    print()

    print(
        f"Training Scenes   : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation Scenes : "
        f"{len(val_dataset):,}"
    )

    ###########################################################################
    # Cache coverage
    ###########################################################################

    cached_training_scenes = 0

    for sequence_id in (
        train_dataset.sequence_ids
    ):

        if train_cache.exists(
            sequence_id
        ):

            cached_training_scenes += 1

    print()

    print(
        f"Training scenes already cached : "
        f"{cached_training_scenes:,}"
    )

    print(
        f"Training scenes requiring preprocessing : "
        f"{len(train_dataset) - cached_training_scenes:,}"
    )

    ###########################################################################
    # DataLoaders
    ###########################################################################

    print_section(
        "Building DataLoaders"
    )

    train_loader = build_dataloader(

        train_dataset,

        train=True,
    )

    val_loader = build_dataloader(

        val_dataset,

        train=False,
    )

    print(
        f"Training Batches   : "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation Batches : "
        f"{len(val_loader):,}"
    )

    ###########################################################################
    # Model
    ###########################################################################

    model = build_model()

    scaler = GradScaler(

        device="cuda",

        enabled=USE_AMP,
    )

    ###########################################################################
    # Training Components
    ###########################################################################

    (
        optimizer,
        scheduler,
        criterion,
    ) = build_training_components(

        model,

        total_steps=len(
            train_loader
        ),
    )

    ###########################################################################
    # Resume
    ###########################################################################

    (
        start_epoch,
        best_metric,
    ) = load_checkpoint(

        model,

        optimizer,

        scheduler,
    )

    epochs_without_improvement = 0

    training_start = (
        time.perf_counter()
    )

    ###########################################################################
    # Epoch Loop
    ###########################################################################

    try:

        for epoch in range(

            start_epoch + 1,

            EPOCHS + 1,
        ):

            print_header(

                f"Epoch "
                f"{epoch}/{EPOCHS}"
            )

            epoch_start = (
                time.perf_counter()
            )

            train_loss = train_one_epoch(

                epoch=epoch,

                model=model,

                dataloader=train_loader,

                optimizer=optimizer,

                scheduler=scheduler,

                criterion=criterion,

                scaler=scaler,
            )

            ###################################################################
            # Validation
            ###################################################################

            if (

                epoch % VALIDATE_EVERY == 0

                or epoch == EPOCHS
            ):

                val_metrics = (
                    validate_one_epoch(

                        model=model,

                        dataloader=val_loader,
                    )
                )

            else:

                val_metrics = {

                    "minADE":
                        best_metric,

                    "minFDE":
                        float("nan"),

                    "MissRate":
                        float("nan"),
                }

            ###################################################################
            # Epoch timing
            ###################################################################

            epoch_time = (

                time.perf_counter()

                - epoch_start
            )

            learning_rate = (

                optimizer.param_groups[0][
                    "lr"
                ]
            )

            append_csv(

                epoch=epoch,

                train_loss=train_loss,

                metrics=val_metrics,

                learning_rate=learning_rate,

                epoch_time=epoch_time,
            )

            ###################################################################
            # Best model
            ###################################################################

            current_metric = (
                val_metrics.get(

                    "minADE",

                    float("inf"),
                )
            )

            is_best = (

                current_metric

                < best_metric
            )

            if is_best:

                best_metric = (
                    current_metric
                )

                epochs_without_improvement = 0

                print()

                print(

                    f"✓ New Best Model "

                    f"(minADE="
                    f"{best_metric:.6f})"
                )

            else:

                epochs_without_improvement += 1

                print(

                    f"No improvement "

                    f"({epochs_without_improvement}/"
                    f"{PATIENCE})"
                )

            ###################################################################
            # Checkpoint
            ###################################################################

            if (

                epoch
                % SAVE_EVERY
                == 0
            ):

                save_checkpoint(

                    epoch=epoch,

                    model=model,

                    optimizer=optimizer,

                    scheduler=scheduler,

                    train_loss=train_loss,

                    val_metrics=val_metrics,

                    best=is_best,
                )

                backup_checkpoint_externally(

                    epoch=epoch,

                    best=is_best,
                )

            ###################################################################
            # Summary
            ###################################################################

            print_epoch_summary(

                epoch=epoch,

                train_loss=train_loss,

                metrics=val_metrics,

                learning_rate=learning_rate,

                epoch_time=epoch_time,
            )

            ###################################################################
            # Early stopping
            ###################################################################

            if (

                EARLY_STOPPING

                and

                epochs_without_improvement
                >= PATIENCE
            ):

                print()

                print("=" * 80)

                print(
                    "Early stopping triggered."
                )

                print(

                    f"No improvement for "
                    f"{PATIENCE} epochs."
                )

                print("=" * 80)

                break

    except KeyboardInterrupt:

        print()

        print("=" * 80)

        print(
            "Training Interrupted"
        )

        print("=" * 80)

        if "epoch" in locals():

            save_checkpoint(

                epoch=epoch,

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

                train_loss=(

                    train_loss

                    if "train_loss" in locals()

                    else float("inf")
                ),

                val_metrics=(

                    val_metrics

                    if "val_metrics" in locals()

                    else {}
                ),

                best=False,
            )

            backup_checkpoint_externally(

                epoch=epoch,

                best=False,
            )

        raise

    ###########################################################################
    # Complete
    ###########################################################################

    total_training_time = (

        time.perf_counter()

        - training_start
    )

    print_header(
        "Training Complete"
    )

    print(
        f"Best minADE : "
        f"{best_metric:.6f}"
    )

    print(
        f"Total Time  : "
        f"{total_training_time / 3600:.2f} hours"
    )

    print(
        f"Checkpoints : "
        f"{CHECKPOINT_ROOT}"
    )

    print(
        f"Logs        : "
        f"{CSV_LOG}"
    )

    print(
        f"Original Cache : "
        f"{PUBLISHED_CACHE_ROOT}"
    )

    print(
        f"Additional Cache : "
        f"{NEW_CACHE_EXTRACT_ROOT}"
    )

    if EXTERNAL_CHECKPOINT_ENABLED:

        print(
            f"External backups : "
            f"{EXTERNAL_CHECKPOINT_ROOT}"
        )

    else:

        print(

            "External backups : DISABLED "
            "(configure "
            "EXTERNAL_CHECKPOINT_ROOT "
            "and enable "
            "EXTERNAL_CHECKPOINT_ENABLED)"
        )


###############################################################################
# Entry Point
###############################################################################


def main() -> None:

    print_header(
        "DSTNet Kaggle Training"
    )

    print_section(
        "Kaggle Configuration"
    )

    print(
        f"Train root       : "
        f"{TRAIN_ROOT}"
    )

    print(
        f"Val root         : "
        f"{VAL_ROOT}"
    )

    print(
        f"Map root         : "
        f"{MAP_ROOT}"
    )

    print(
        f"Published cache  : "
        f"{PUBLISHED_CACHE_ROOT}"
    )

    print(
        f"Additional cache : "
        f"{NEW_CACHE_EXTRACT_ROOT}"
    )

    print(
        f"Working cache    : "
        f"{CACHE_ROOT}"
    )

    print(
        f"Device           : "
        f"{DEVICE}"
    )

    print(
        f"Batch size       : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs           : "
        f"{EPOCHS}"
    )

    print(
        f"External checkpointing : "
        f"{EXTERNAL_CHECKPOINT_ENABLED}"
    )

    run_training()


###############################################################################
# Entry Point
###############################################################################


if __name__ == "__main__":

    main()
