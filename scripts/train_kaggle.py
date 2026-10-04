"""
scripts.train_kaggle

Kaggle training entry point for the current DSTNet implementation.

Purpose
-------
Runs production-style training on the Kaggle Argoverse-1 dataset while
using versioned preprocessed SceneData caches.

Training cache sources
----------------------
Training scenes are loaded from two persistent Kaggle datasets:

    1. av1-train-p1-cache
    2. av1-train-p2-cache

The first cache is checked first. If a sequence is not present there,
the additional training cache is checked.

Validation scenes are loaded from:

    av1-val-cache

All required scene IDs are checked against the mounted cache shards before
training begins. Cache generation is handled by
``scripts/prepare_kaggle_cache.py``.

Current model/data terminology
------------------------------
The training batch uses:

    agent_trajectories
    future_trajectories
    map_centerlines
    positions
    agent_mask
    map_mask
    graph

``headings`` remains available in the dataset/scene representation but is
not passed as a separate argument to the current DSTNet forward interface.

Checkpointing
-------------
A full local checkpoint is written every epoch.

An optional external backup path is provided below. Once that path is
changed to a persistent mounted location, the script copies a complete
checkpoint there every ``EXTERNAL_SAVE_EVERY`` epochs.

Cache
-----
The persistent Kaggle caches are read-only and must use the current cache
format. Startup verifies cache versions, scene coverage, and shard overlap.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
import pickle
from contextlib import nullcontext

import warnings

import numpy as np

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True",
)
os.environ.setdefault(
    "DSTNET_VALIDATE_FINITE",
    "0",
)

import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Dataset


###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# Kaggle commonly has Hugging Face's package named ``datasets`` installed.
# A notebook may have imported it before this script runs, in which case
# Python would otherwise resolve it instead of this repository's package.
_loaded_datasets = sys.modules.get("datasets")
if _loaded_datasets is not None:
    _loaded_datasets_path = getattr(_loaded_datasets, "__file__", None)
    _is_project_datasets = False
    if _loaded_datasets_path is not None:
        try:
            Path(_loaded_datasets_path).resolve().relative_to(PROJECT_ROOT)
            _is_project_datasets = True
        except ValueError:
            pass

    if not _is_project_datasets:
        for _module_name in tuple(sys.modules):
            if _module_name == "datasets" or _module_name.startswith("datasets."):
                del sys.modules[_module_name]


###############################################################################
# Current Repository Imports
###############################################################################

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.augmentation import RandomReflectionDataset
from datasets.cache_config import (
    CACHE_VERSION,
    KAGGLE_PREPROCESSING_CONFIG,
)
from datasets.collate import collate_fn
from datasets.transforms import (
    build_eval_transform,
)

from engine.evaluator import Evaluator
from engine.optimizer import build_optimizer
from engine.scheduler import build_scheduler
from engine.utils import (
    move_to_device,
    select_supervised_agents,
)
from utils.numerics import FINITE_CHECKS_ENABLED

from losses.total_loss import TotalLoss

from models.dstnet import DSTNet


###############################################################################
# Kaggle Dataset Paths
###############################################################################

def _resolve_kaggle_input(
    environment_name: str,
    candidates: tuple[Path, ...],
) -> Path:
    """Use an explicit path override, or the first mounted Kaggle path."""

    override = os.environ.get(environment_name, "").strip()
    if override:
        return Path(override).expanduser()

    return next(
        (candidate for candidate in candidates if candidate.is_dir()),
        candidates[0],
    )


_ARGOVERSE_SLUG = "argoverse1-motion-dataset"
_ARGOVERSE_OWNER = os.environ.get(
    "DSTNET_ARGOVERSE_OWNER",
    "narendarmallireddy",
).strip()

TRAIN_ROOT = _resolve_kaggle_input(
    "DSTNET_TRAIN_ROOT",
    (
        Path("/kaggle/input") / _ARGOVERSE_SLUG
        / "forecasting_train_v1.1/train/data",
        Path("/kaggle/input/datasets") / _ARGOVERSE_OWNER
        / _ARGOVERSE_SLUG / "forecasting_train_v1.1/train/data",
        Path("/kaggle/input") / _ARGOVERSE_OWNER / _ARGOVERSE_SLUG
        / "forecasting_train_v1.1/train/data",
    ),
)

VAL_ROOT = _resolve_kaggle_input(
    "DSTNET_VAL_ROOT",
    (
        Path("/kaggle/input") / _ARGOVERSE_SLUG
        / "forecasting_val_v1.1/val/data",
        Path("/kaggle/input/datasets") / _ARGOVERSE_OWNER
        / _ARGOVERSE_SLUG / "forecasting_val_v1.1/val/data",
        Path("/kaggle/input") / _ARGOVERSE_OWNER / _ARGOVERSE_SLUG
        / "forecasting_val_v1.1/val/data",
    ),
)

TEST_ROOT = Path(
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/"
    "forecasting_test_v1.1/test_obs/data"
)

MAP_ROOT = Path(os.environ.get(
    "DSTNET_MAP_ROOT",
    "/kaggle/input/datasets/kedaradhikari/"
    "argoverse1-hd-mapss/"
    "hd_maps/map_files",
))


###############################################################################
# Persistent Preprocessed Cache Datasets
###############################################################################
#
# IMPORTANT:
#
# These are READ-ONLY Kaggle input datasets.
#
# Training uses both caches:
#
#     TRAIN_CACHE_ROOT
#     TRAIN_ADDITIONAL_CACHE_ROOT
#
# Validation uses:
#
#     VAL_CACHE_ROOT
#
###############################################################################

_CACHE_OWNER = os.environ.get(
    "DSTNET_KAGGLE_OWNER",
    "kedaradhikari",
).strip()


def _resolve_cache_root(
    environment_name: str,
    dataset_slug: str,
) -> Path:
    """Resolve a mounted cache dataset across Kaggle mount layouts."""

    override = os.environ.get(environment_name, "").strip()
    if override:
        return Path(override).expanduser()

    candidates = (
        Path("/kaggle/input") / dataset_slug / "cache",
        Path("/kaggle/input") / dataset_slug.replace("-", "_") / "cache",
        Path("/kaggle/input/datasets") / _CACHE_OWNER / dataset_slug / "cache",
        Path("/kaggle/input/datasets") / _CACHE_OWNER
        / dataset_slug.replace("-", "_") / "cache",
        Path("/kaggle/input") / _CACHE_OWNER / dataset_slug / "cache",
        Path("/kaggle/input") / _CACHE_OWNER
        / dataset_slug.replace("-", "_") / "cache",
    )
    return next(
        (candidate for candidate in candidates if candidate.is_dir()),
        candidates[0],
    )


TRAIN_CACHE_ROOT = _resolve_cache_root(
    "DSTNET_TRAIN_CACHE_A",
    "av1-train-p1-cache",
)

TRAIN_ADDITIONAL_CACHE_ROOT = _resolve_cache_root(
    "DSTNET_TRAIN_CACHE_B",
    "av1-train-p2-cache",
)

VAL_CACHE_ROOT = _resolve_cache_root(
    "DSTNET_VAL_CACHE",
    "av1-val-cache",
)


###############################################################################
# Kaggle Working Directories
###############################################################################

CHECKPOINT_ROOT = Path(os.environ.get(
    "DSTNET_CHECKPOINT_ROOT",
    "/kaggle/working/checkpoints",
))

LOG_ROOT = Path(os.environ.get(
    "DSTNET_LOG_ROOT",
    "/kaggle/working/logs",
))

RESULTS_ROOT = Path(os.environ.get(
    "DSTNET_RESULTS_ROOT",
    "/kaggle/working/results",
))

# ---------------------------------------------------------------------------
# Local fallback cache.
#
# This is NOT the persistent training/validation cache.
#
# If a CSV scene is not found in the persistent cache datasets, the normal
# ArgoverseDataset pipeline will preprocess it and write it here.
# ---------------------------------------------------------------------------

# CACHE_ROOT = (
#     Path(
#         "/kaggle/working/cache"
#     )
# )


###############################################################################
# External Checkpoint Backup
###############################################################################

EXTERNAL_CHECKPOINT_ROOT = Path(
    "/YOUR/EXTERNAL/MOUNT/DSTNet/checkpoints"
)

EXTERNAL_CHECKPOINT_ENABLED = False

EXTERNAL_SAVE_EVERY = 5


###############################################################################
# Training Configuration
###############################################################################

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

ALLOW_CPU = os.environ.get(
    "DSTNET_ALLOW_CPU",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}

if torch.cuda.is_available():

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(
        max(1, int(os.environ.get("DSTNET_TORCH_THREADS", "1")))
    )


SEED = 42

BATCH_SIZE = int(
    os.environ.get(
        "DSTNET_BATCH_SIZE",
        "8",
    )
)

NUM_WORKERS = int(
    os.environ.get(
        "DSTNET_NUM_WORKERS",
        str(min(4, os.cpu_count() or 2)),
    )
)

EPOCHS = int(os.environ.get("DSTNET_EPOCHS", "30"))

LEARNING_RATE = float(os.environ.get("DSTNET_LEARNING_RATE", "2e-5"))

WEIGHT_DECAY = float(os.environ.get("DSTNET_WEIGHT_DECAY", "1e-4"))

SAVE_EVERY = 1

GRADIENT_CLIP = 5.0

VALIDATE_EVERY = max(
    1,
    int(os.environ.get("DSTNET_VALIDATE_EVERY", "1")),
)

REFINEMENT_ENABLED = os.environ.get(
    "DSTNET_REFINEMENT_ENABLED",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}

INIT_CHECKPOINT = os.environ.get("DSTNET_INIT_CHECKPOINT", "").strip()


###############################################################################
# Mixed Precision
###############################################################################

USE_AMP = True

NATIVE_BF16_SUPPORTED = (
    DEVICE.type == "cuda"
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.get_device_capability(DEVICE)[0] >= 8
    and torch.cuda.is_bf16_supported()
)

AMP_DTYPE = (
    torch.bfloat16
    if NATIVE_BF16_SUPPORTED
    else torch.float16
)

AMP_ENABLED = (
    USE_AMP
    and DEVICE.type == "cuda"
)

SCALER_ENABLED = (
    AMP_ENABLED
    and AMP_DTYPE == torch.float16
)


###############################################################################
# Logging
###############################################################################

LOG_EVERY = max(1, int(os.environ.get("DSTNET_LOG_EVERY", "200")))


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
# Parameter Counter
###############################################################################


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:

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


def _prediction_to_float32(prediction):
    """Keep AMP for the model while evaluating losses in float32."""

    if prediction is None:
        return None

    updates = {
        field.name: value.float()
        for field in fields(prediction)
        if isinstance(
            (value := getattr(prediction, field.name)),
            torch.Tensor,
        )
        and value.is_floating_point()
    }
    return replace(prediction, **updates)


def seed_worker(_: int) -> None:
    """Seed Python and NumPy in each DataLoader worker."""

    worker_seed = torch.initial_seed() % (2**32)
    torch.set_num_threads(1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def set_random_seed() -> None:
    """Make shuffling and any worker-side randomness reproducible."""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def validate_runtime() -> None:
    """Fail early when this Kaggle training job has no GPU attached."""

    if DEVICE.type != "cuda":
        if ALLOW_CPU:
            print("CUDA is unavailable; CPU training was explicitly allowed.")
            return
        raise RuntimeError(
            "No CUDA GPU is available. In Kaggle, set Notebook Settings > "
            "Accelerator to GPU and restart the session. For local debugging "
            "only, set DSTNET_ALLOW_CPU=1."
        )

    properties = torch.cuda.get_device_properties(DEVICE)
    memory_gib = properties.total_memory / (1024**3)
    print(
        f"CUDA GPU               : {properties.name} "
        f"({memory_gib:.1f} GiB)"
    )


###############################################################################
# Scene Preprocessor
###############################################################################


###############################################################################
# Multi-Cache Manager
###############################################################################


class MultiCacheManager:
    """
    Read-only cache manager for multiple persistent Kaggle cache datasets.

    The caches are searched in the order supplied to the constructor.

    Example
    -------
    Training:

        MultiCacheManager(
            [
                TRAIN_CACHE_ROOT,
                TRAIN_ADDITIONAL_CACHE_ROOT,
            ]
        )

    Search order:

        1. Original training cache
        2. Additional training cache

    Validation:

        MultiCacheManager(
            [
                VAL_CACHE_ROOT,
            ]
        )

    This manager is intentionally READ-ONLY.

    It never:
        - creates directories
        - creates VERSION files
        - writes .tmp files
        - writes .pkl files
        - falls back to /kaggle/working
    """

    def __init__(
        self,
        cache_roots: list[Path],
    ) -> None:

        if not cache_roots:

            raise ValueError(
                "MultiCacheManager requires at least "
                "one cache root."
            )

        self.cache_roots = [
            Path(root)
            for root in cache_roots
        ]
        self._path_by_id: dict[str, Path] = {}
        self._duplicate_ids: set[str] = set()
        for root in self.cache_roots:
            for path in root.glob("*.pkl"):
                if path.stem in self._path_by_id:
                    self._duplicate_ids.add(path.stem)
                self._path_by_id.setdefault(path.stem, path)

    ###########################################################################
    # Exists
    ###########################################################################

    def exists(
        self,
        sequence_id: str,
    ) -> bool:

        return sequence_id in self._path_by_id

    def ids(self) -> set[str]:
        """Return all cached scene IDs across the mounted roots."""

        return set(self._path_by_id)

    def missing(self, sequence_ids: list[str]) -> list[str]:
        """Return requested IDs absent from every mounted cache root."""

        cached = self.ids()
        return [sequence_id for sequence_id in sequence_ids if sequence_id not in cached]

    def duplicate_ids(self) -> set[str]:
        """Find IDs copied into more than one shard."""

        return set(self._duplicate_ids)

    ###########################################################################
    # Load
    ###########################################################################

    def load(
        self,
        sequence_id: str,
    ):
        """
        Load a cached SceneData object.

        The cache is searched in the configured order.

        No CacheManager instance is created here because CacheManager's
        constructor is writable and would attempt to create/update files
        inside the read-only Kaggle input dataset.
        """

        path = self._path_by_id.get(sequence_id)
        if path is not None:
            try:
                with open(path, "rb") as file:

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"numpy\.core\.numeric is deprecated.*",
                            category=DeprecationWarning,
                        )
                        scene = pickle.load(file)

            except (
                EOFError,
                pickle.UnpicklingError,
            ) as exc:

                raise RuntimeError(
                    "Corrupted persistent cache detected: "
                    f"{path}"
                ) from exc

            ###################################################################
            # Validate cached object
            ###################################################################

            from datasets.scene_data import SceneData

            if not isinstance(
                scene,
                SceneData,
            ):

                raise TypeError(
                    "Invalid cached object in "
                    f"{path}. Expected SceneData, "
                    f"got {type(scene).__name__}."
                )

            return scene

        raise FileNotFoundError(
            "Cached scene was not found in any "
            f"persistent cache for sequence '{sequence_id}'. "
            f"Searched: {self.cache_roots}"
        )

    ###########################################################################
    # Save
    ###########################################################################

    def save(
        self,
        scene,
    ) -> None:

        raise RuntimeError(
            "MultiCacheManager is read-only. "
            "Training/validation must not create or modify "
            "persistent cache files."
        )


###############################################################################
# Cache Validation
###############################################################################


def validate_cache_roots() -> None:
    """
    Verify the persistent cache directories before training begins.

    The original CSV directories remain configured so that the dataset
    can obtain the sequence IDs and establish dataset length.

    They are NOT used for parsing or preprocessing in cache-only mode.

    Every scene required for training and validation must exist in the
    persistent cache datasets.
    """

    print_section(
        "Persistent Cache Configuration"
    )

    cache_roots = [

        (
            "Training cache",
            TRAIN_CACHE_ROOT,
            "train-1",
        ),

        (
            "Additional training cache",
            TRAIN_ADDITIONAL_CACHE_ROOT,
            "train-2",
        ),

        (
            "Validation cache",
            VAL_CACHE_ROOT,
            "val",
        ),
    ]

    for name, root, expected_split in cache_roots:

        if not root.exists():

            raise FileNotFoundError(
                f"{name} does not exist: {root}. Attach the Kaggle cache "
                "dataset to this notebook or set its DSTNET_*_CACHE path "
                "override."
            )

        version_file = root / "VERSION"
        if not version_file.is_file():
            raise FileNotFoundError(
                f"{name} has no VERSION file: {version_file}. "
                "Regenerate it with scripts/prepare_kaggle_cache.py."
            )

        cache_version = version_file.read_text(encoding="utf-8").strip()
        if cache_version != CACHE_VERSION:
            raise RuntimeError(
                f"{name} uses cache format {cache_version!r}; "
                f"expected {CACHE_VERSION!r}. Regenerate this cache."
            )

        manifest_path = root.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{name} has no manifest: {manifest_path}. Publish the "
                "complete output root created by prepare_kaggle_cache.py."
            )
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if (
            manifest.get("complete") is not True
            or manifest.get("cache_version") != CACHE_VERSION
            or manifest.get("split") != expected_split
            or manifest.get("preprocessing") != KAGGLE_PREPROCESSING_CONFIG
        ):
            raise RuntimeError(
                f"{name} manifest is incomplete or has the wrong split/version: "
                f"{manifest_path}"
            )

        pkl_count = sum(1 for _ in root.glob("*.pkl"))

        print(
            f"{name:<28}: "
            f"{root}"
        )

        print(
            f"{'':28}  "
            f"cache files = {pkl_count:,}"
        )

        if manifest.get("scene_count") != pkl_count:
            raise RuntimeError(
                f"{name} manifest lists {manifest.get('scene_count')} scenes, "
                f"but {pkl_count} pickle files are mounted."
            )

        if pkl_count == 0:

            raise RuntimeError(
                f"{name} exists but contains "
                f"no .pkl cache files: {root}"
            )


###############################################################################
# Dataset
###############################################################################


def build_dataset(
    root: Path,
    *,
    train: bool,
) -> ArgoverseDataset:
    """
    Build one Argoverse-1 split using persistent preprocessed caches.

    Training
    --------
    Uses the union of:

        TRAIN_CACHE_ROOT
        TRAIN_ADDITIONAL_CACHE_ROOT

    Validation
    ----------
    Uses:

        VAL_CACHE_ROOT

    The dataset operates in strict cache-only mode.

    The CSV root is retained for dataset indexing and sequence IDs, but
    missing cached scenes are NOT parsed or preprocessed.
    """

    if not root.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{root}"
        )

    ###########################################################################
    # Select persistent cache(s)
    ###########################################################################

    if train:

        persistent_cache_roots = [

            TRAIN_CACHE_ROOT,

            TRAIN_ADDITIONAL_CACHE_ROOT,
        ]

    else:

        persistent_cache_roots = [

            VAL_CACHE_ROOT,
        ]

    ###########################################################################
    # Read-through cache
    ###########################################################################

    cache = MultiCacheManager(

        cache_roots=persistent_cache_roots,

    )

    ###########################################################################
    # Transform
    ###########################################################################

    ###########################################################################
    # Dataset
    ###########################################################################

    dataset = ArgoverseDataset(

        root=root,

        parser=None,

        preprocessor=None,

        transform=build_eval_transform(),

        cache=cache,

        cache_only=True,
    )

    missing = cache.missing(dataset.sequence_ids)
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing):,} {('training' if train else 'validation')} "
            "scenes are missing from the mounted cache(s). First missing "
            f"IDs: {preview}"
        )

    unexpected = cache.ids().difference(dataset.sequence_ids)
    if unexpected:
        preview = ", ".join(sorted(unexpected)[:10])
        raise RuntimeError(
            f"Cache contains {len(unexpected):,} scene IDs that do not "
            f"belong to this split. First IDs: {preview}"
        )

    duplicates = cache.duplicate_ids()
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise RuntimeError(
            f"Cache shards overlap on {len(duplicates):,} scene IDs. "
            f"First IDs: {preview}"
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

    if train:
        dataset = RandomReflectionDataset(dataset)

    generator = torch.Generator()
    generator.manual_seed(
        SEED + (0 if train else 1)
    )

    kwargs = {

        "dataset": dataset,

        "batch_size": BATCH_SIZE,

        "shuffle": train,

        "num_workers": NUM_WORKERS,

        "collate_fn": collate_fn,

        "pin_memory": (
            DEVICE.type == "cuda"
        ),

        "drop_last": False,

        "worker_init_fn": seed_worker,

        "generator": generator,
    }

    if NUM_WORKERS > 0:

        kwargs["persistent_workers"] = True

        kwargs["prefetch_factor"] = 2

    return DataLoader(
        **kwargs,
    )


###############################################################################
# Model
###############################################################################


def build_model() -> DSTNet:
    """
    Build the current DSTNet.
    """

    model = DSTNet(
        refinement_enabled=REFINEMENT_ENABLED,
    )

    model.to(
        DEVICE,
    )

    total, trainable = count_parameters(
        model,
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

        foreach=(DEVICE.type == "cuda"),
    )

    total_updates = max(
        1,
        total_steps * EPOCHS,
    )

    warmup_steps = min(
        int(total_updates * 0.03),
        max(0, total_updates - 1),
    )

    scheduler = build_scheduler(

        optimizer,

        scheduler="warmup_cosine",

        total_steps=total_updates,

        warmup_steps=warmup_steps,
    )

    criterion = TotalLoss(
        refinement_enabled=REFINEMENT_ENABLED,
    )

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

    # CACHE_ROOT.mkdir(
    #     parents=True,
    #     exist_ok=True,
    # )


###############################################################################
# Checkpoint State
###############################################################################


def build_checkpoint_state(
    *,
    epoch: int,
    model: DSTNet,
    optimizer,
    scheduler,
    scaler: GradScaler | None,
    train_loss: float,
    val_metrics: dict[str, float],
    best_metric: float | None,
    steps_per_epoch: int,
) -> dict:

    return {

        "epoch": epoch,

        "model_stage": (
            "refinement" if REFINEMENT_ENABLED else "backbone"
        ),

        "refinement_enabled": REFINEMENT_ENABLED,

        "steps_per_epoch": steps_per_epoch,

        "train_loss": train_loss,

        "val_metrics": val_metrics,

        "best_metric": (
            float(best_metric)
            if best_metric is not None
            else float(
                val_metrics.get(
                    "minADE",
                    float("inf"),
                )
            )
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (

            scheduler.state_dict()

            if scheduler is not None

            else None
        ),

        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),

        "torch_rng_state": (
            torch.get_rng_state()
        ),

        "cuda_rng_state_all": (

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
    scaler: GradScaler | None = None,
    train_loss: float,
    val_metrics: dict[str, float],
    best_metric: float | None = None,
    steps_per_epoch: int = 0,
    best: bool = False,
) -> Path:

    checkpoint = build_checkpoint_state(

        epoch=epoch,

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        scaler=scaler,

        train_loss=train_loss,

        val_metrics=val_metrics,

        best_metric=best_metric,

        steps_per_epoch=steps_per_epoch,
    )

    latest_tmp = LATEST_CHECKPOINT.with_suffix(
        LATEST_CHECKPOINT.suffix + ".tmp"
    )

    torch.save(checkpoint, latest_tmp)
    os.replace(latest_tmp, LATEST_CHECKPOINT)

    if best:

        best_tmp = BEST_CHECKPOINT.with_suffix(
            BEST_CHECKPOINT.suffix + ".tmp"
        )

        torch.save(checkpoint, best_tmp)
        os.replace(best_tmp, BEST_CHECKPOINT)

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
        and
        BEST_CHECKPOINT.exists()
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
    scaler: GradScaler | None = None,
    steps_per_epoch: int = 1,
) -> tuple[int, float]:

    if INIT_CHECKPOINT:
        init_path = Path(INIT_CHECKPOINT)
        if not init_path.is_file():
            raise FileNotFoundError(
                f"DSTNET_INIT_CHECKPOINT does not exist: {init_path}"
            )

        print_section("Initializing Model Weights")
        checkpoint = torch.load(init_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        incompatible = model.load_state_dict(
            state_dict,
            strict=not REFINEMENT_ENABLED,
        )
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        allowed_missing = (
            {key for key in model.state_dict() if key.startswith("refinement.")}
            if REFINEMENT_ENABLED
            else set()
        )
        invalid_missing = set(missing) - allowed_missing
        if invalid_missing or unexpected:
            raise RuntimeError(
                "Initialization checkpoint is incompatible. "
                f"Missing keys: {sorted(invalid_missing)[:10]}; "
                f"unexpected keys: {unexpected[:10]}"
            )

        print(f"Initialized from : {init_path}")
        if missing:
            print(
                f"New refinement parameters: {len(missing):,}; "
                "optimizer/scheduler state starts fresh."
            )
        return 0, float("inf")

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
        # Keep RNG state tensors on CPU. In particular, CUDA RNG state APIs
        # require CPU ByteTensors; mapping the whole checkpoint to DEVICE
        # moves those tensors to CUDA and breaks resume on newer PyTorch.
        # load_state_dict below copies model and optimizer state to the
        # appropriate device.
        map_location="cpu",
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

    checkpoint_epoch = int(
        checkpoint.get("epoch", 0)
    )

    saved_steps_per_epoch = int(
        checkpoint.get("steps_per_epoch", steps_per_epoch)
    )

    if (
        scheduler is not None
        and hasattr(scheduler, "lr_lambdas")
        and (
            "steps_per_epoch" not in checkpoint
            or saved_steps_per_epoch != steps_per_epoch
        )
    ):
        # A changed batch size changes updates per epoch. Rebase the schedule
        # to the completed epoch count instead of carrying a stale step count.
        scheduler.last_epoch = checkpoint_epoch * steps_per_epoch
        scheduler._step_count = scheduler.last_epoch + 1

    # Optimizer state restores its saved hyperparameters too. Reapply the
    # current Kaggle run's LR/decay while retaining Adam's moment estimates.
    for group_index, group in enumerate(optimizer.param_groups):
        group["initial_lr"] = LEARNING_RATE
        group["weight_decay"] = (
            WEIGHT_DECAY
            if group_index == 0
            else 0.0
        )

    if scheduler is not None and hasattr(scheduler, "lr_lambdas"):
        scheduler.base_lrs = [
            LEARNING_RATE
            for _ in optimizer.param_groups
        ]

        for group, lr_lambda in zip(
            optimizer.param_groups,
            scheduler.lr_lambdas,
        ):
            group["lr"] = (
                LEARNING_RATE
                * lr_lambda(scheduler.last_epoch)
            )

        scheduler._last_lr = [
            group["lr"]
            for group in optimizer.param_groups
        ]

    scaler_state = checkpoint.get(
        "scaler_state_dict"
    )

    if scaler is not None and scaler_state:
        scaler.load_state_dict(scaler_state)

    ###########################################################################
    # CPU RNG
    ###########################################################################

    try:

        torch_rng_state = checkpoint.get(
            "torch_rng_state",
            checkpoint.get("rng_state"),
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
                checkpoint.get("cuda_rng_states"),
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
                        "Warning: CUDA RNG state in checkpoint "
                        "is incompatible with the current CUDA "
                        "device configuration. Skipping RNG "
                        "restoration."
                    )

        except Exception as exc:

            print(
                f"Warning: CUDA RNG state could not be restored: "
                f"{exc}"
            )

    epoch = checkpoint_epoch

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
    supervised_agent_count = 0
    successful_updates = 0
    skipped_amp_updates = 0
    consecutive_skipped_amp_updates = 0
    skipped_nonfinite_gradient_updates = 0
    consecutive_nonfinite_gradient_updates = 0
    max_consecutive_amp_skips = max(
        1,
        int(os.environ.get("DSTNET_MAX_CONSECUTIVE_AMP_SKIPS", "25")),
    )

    num_batches = len(dataloader)

    if num_batches == 0:

        raise RuntimeError(
            "Training DataLoader contains zero batches."
        )

    epoch_start = time.perf_counter()

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):

        batch_start = time.perf_counter()

        #######################################################################
        # Move batch
        #######################################################################

        cpu_supervision_mask = batch.get(
            "future_mask",
            batch.get("agent_mask"),
        )

        if (
            cpu_supervision_mask is not None
            and not bool(cpu_supervision_mask.any())
        ):
            raise RuntimeError(
                "This batch contains no agents with a complete future label."
            )

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
        # Forward
        #######################################################################

        amp_context = (
            autocast(
                device_type=DEVICE.type,
                dtype=AMP_DTYPE,
                enabled=True,
            )
            if AMP_ENABLED
            else nullcontext()
        )

        with amp_context:
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


        supervision_mask = batch.get(
            "future_mask",
            batch.get("agent_mask"),
        )
        if (
            supervision_mask is not None
            and "agent_mask" in batch
        ):
            supervision_mask = (
                supervision_mask
                & batch["agent_mask"].bool()
            )

        (
            coarse_prediction,
            refined_prediction,
            ground_truth,
        ) = select_supervised_agents(
            coarse_prediction,
            refined_prediction,
            batch["future_trajectories"],
            supervision_mask,
            validate_non_empty=False,
        )

        # AMP substantially speeds up the DSTNet forward/backward passes.
        # The trajectory losses use full precision to avoid overflow and
        # preserve useful gradients, especially on Kaggle's FP16 GPUs.
        coarse_prediction = _prediction_to_float32(coarse_prediction)
        refined_prediction = _prediction_to_float32(refined_prediction)
        ground_truth = ground_truth.float()

        losses = criterion(
            prediction=coarse_prediction,
            refined_prediction=refined_prediction,
            ground_truth=ground_truth,
        )

        loss = losses["loss"]

        #######################################################################
        # Loss sanity check
        #######################################################################

        loss_value = loss.detach().float().item()
        if not np.isfinite(loss_value):

            print()
            print("=" * 80)
            print("NON-FINITE LOSS DETECTED")
            print("=" * 80)

            print(
                f"Epoch : {epoch}"
            )

            print(
                f"Batch : {batch_index}"
            )

            print(
                f"Loss  : "
                f"{loss_value}"
            )

            raise FloatingPointError(
                "Non-finite loss detected."
            )

        #######################################################################
        # Backward
        #######################################################################

        if SCALER_ENABLED:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

        else:

            loss.backward()

        #######################################################################
        # Gradient clipping
        #######################################################################

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP,
                error_if_nonfinite=False,
            )
        )

        #######################################################################
        # Optimizer update
        #######################################################################

        did_update = bool(torch.isfinite(gradient_norm).item())

        if not did_update:
            # A non-finite global norm can arise before individual gradients
            # become non-finite. Never pass such a batch to AdamW. If AMP is
            # scaled, update the scaler so it can reduce its scale after an
            # overflow recorded by unscale_().
            optimizer.zero_grad(set_to_none=True)
            if SCALER_ENABLED:
                scaler.update()
            skipped_nonfinite_gradient_updates += 1
            consecutive_nonfinite_gradient_updates += 1
            if consecutive_nonfinite_gradient_updates >= max_consecutive_amp_skips:
                raise FloatingPointError(
                    "Gradient clipping found a non-finite total norm for "
                    f"{consecutive_nonfinite_gradient_updates} consecutive "
                    "batches. The optimizer was not updated for those batches. "
                    "Lower DSTNET_LEARNING_RATE or DSTNET_BATCH_SIZE, then "
                    "resume from the latest completed-epoch checkpoint."
                )

        elif SCALER_ENABLED:

            previous_scale = scaler.get_scale()

            scaler.step(
                optimizer
            )

            scaler.update()

            did_update = scaler.get_scale() >= previous_scale

            if did_update:
                successful_updates += 1
                consecutive_skipped_amp_updates = 0
                consecutive_nonfinite_gradient_updates = 0
            else:
                skipped_amp_updates += 1
                consecutive_skipped_amp_updates += 1
                if consecutive_skipped_amp_updates >= max_consecutive_amp_skips:
                    raise FloatingPointError(
                        "FP16 GradScaler skipped "
                        f"{consecutive_skipped_amp_updates} consecutive updates. "
                        "Try a smaller DSTNET_BATCH_SIZE or use a GPU with "
                        "BF16 support."
                    )

        else:

            optimizer.step()
            successful_updates += 1
            consecutive_nonfinite_gradient_updates = 0

        #######################################################################
        # Scheduler
        #######################################################################

        if scheduler is not None and did_update:

            scheduler.step()

        #######################################################################
        # Statistics
        #######################################################################

        current_agent_count = coarse_prediction.trajectories.shape[1]
        running_loss += (
            loss.detach().float()
            * current_agent_count
        )
        supervised_agent_count += current_agent_count

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

    average_loss = float(
        (running_loss / max(1, supervised_agent_count)).item()
    )

    if successful_updates == 0:
        raise FloatingPointError(
            "No optimizer updates succeeded during this epoch."
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

    if skipped_amp_updates:
        print(
            f"AMP updates skipped : {skipped_amp_updates:,}",
            flush=True,
        )

    if skipped_nonfinite_gradient_updates:
        print(
            "Non-finite gradient batches skipped : "
            f"{skipped_nonfinite_gradient_updates:,}",
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

        autocast_enabled=AMP_ENABLED,

        autocast_dtype=AMP_DTYPE,
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

    print("-" * 80)

    print(
        "Validation Metrics"
    )

    print("-" * 80)

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
# Main Training Pipeline
###############################################################################


def run_training() -> None:

    validate_runtime()

    set_random_seed()

    create_directories()

    initialize_csv()

    validate_cache_roots()

    ###########################################################################
    # Datasets
    ###########################################################################

    print_section(
        "Building Datasets"
    )

    train_dataset = build_dataset(

        TRAIN_ROOT,

        train=True,
    )

    val_dataset = build_dataset(

        VAL_ROOT,

        train=False,
    )

    print(
        f"Training Scenes   : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation Scenes : "
        f"{len(val_dataset):,}"
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
        enabled=SCALER_ENABLED,
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

        scaler,

        len(train_loader),
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

            ###################################################################
            # Training
            ###################################################################

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

            did_validate = (
                epoch % VALIDATE_EVERY == 0
                or epoch == EPOCHS
            )

            if did_validate:

                val_metrics = (
                    validate_one_epoch(

                        model=model,

                        dataloader=val_loader,
                    )
                )

            else:

                val_metrics = {

                    "minADE": best_metric,

                    "minFDE": float("nan"),

                    "MissRate": float("nan"),
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

            ###################################################################
            # CSV
            ###################################################################

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
                did_validate
                and current_metric < best_metric
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

            elif did_validate:

                epochs_without_improvement += 1

                print(
                    f"No improvement "
                    f"({epochs_without_improvement}/"
                    f"{PATIENCE})"
                )

            ###################################################################
            # Local checkpoint
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

                    scaler=scaler,

                    train_loss=train_loss,

                    val_metrics=val_metrics,

                    best_metric=best_metric,

                    steps_per_epoch=len(train_loader),

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

                scaler=scaler,

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

                best_metric=(
                    best_metric
                    if "best_metric" in locals()
                    else float("inf")
                ),

                steps_per_epoch=len(train_loader),

                best=False,
            )

            backup_checkpoint_externally(

                epoch=epoch,

                best=False,
            )

        raise

    ###########################################################################
    # Completion
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

    # print(
    #     f"Local fallback cache : "
    #     f"{CACHE_ROOT}"
    # )

    print()
    print(
        "Persistent training caches:"
    )

    print(
        f"  {TRAIN_CACHE_ROOT}"
    )

    print(
        f"  {TRAIN_ADDITIONAL_CACHE_ROOT}"
    )

    print(
        "Persistent validation cache:"
    )

    print(
        f"  {VAL_CACHE_ROOT}"
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
        f"Train root             : "
        f"{TRAIN_ROOT}"
    )

    print(
        f"Val root               : "
        f"{VAL_ROOT}"
    )

    print(
        f"Map root (cache build) : "
        f"{MAP_ROOT}"
    )

    print(
        f"Training cache         : "
        f"{TRAIN_CACHE_ROOT}"
    )

    print(
        f"Additional train cache : "
        f"{TRAIN_ADDITIONAL_CACHE_ROOT}"
    )

    print(
        f"Validation cache       : "
        f"{VAL_CACHE_ROOT}"
    )

    # print(
    #     f"Fallback cache         : "
    #     f"{CACHE_ROOT}"
    # )

    print(
        f"Device                 : "
        f"{DEVICE}"
    )

    print(
        f"Batch size             : "
        f"{BATCH_SIZE}"
    )

    print(
        f"DataLoader workers     : "
        f"{NUM_WORKERS}"
    )

    print(
        f"Mixed precision        : "
        f"{AMP_ENABLED} ({AMP_DTYPE})"
    )

    print(
        f"Native BF16 supported  : "
        f"{NATIVE_BF16_SUPPORTED}"
    )

    print(
        f"FP16 gradient scaler   : "
        f"{SCALER_ENABLED}"
    )

    print(
        f"Per-layer finite checks: "
        f"{FINITE_CHECKS_ENABLED}"
    )

    print(
        f"Learning rate          : "
        f"{LEARNING_RATE:.2e}"
    )

    print(
        f"Refinement enabled     : "
        f"{REFINEMENT_ENABLED}"
    )

    print(
        f"Weight initialization  : "
        f"{INIT_CHECKPOINT or 'resume latest / random initialization'}"
    )

    print(
        f"Epochs                 : "
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
