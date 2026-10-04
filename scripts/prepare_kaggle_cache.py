"""Build one versioned Kaggle cache shard for Argoverse 1.

Run once per split from a Kaggle notebook. Training is deterministically
divided into two approximately size-balanced shards so neither cache build
needs to keep the full training cache in ``/kaggle/working`` at once.

Examples::

    python scripts/prepare_kaggle_cache.py --split train-1
    python scripts/prepare_kaggle_cache.py --split train-2
    python scripts/prepare_kaggle_cache.py --split val

Each run writes ``<output-root>/cache/*.pkl`` and a small ``manifest.json``.
Publish each output root as a separate Kaggle dataset and attach the two
training datasets plus the validation dataset to the training notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TRAIN_ROOT = Path(
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/forecasting_train_v1.1/train/data"
)
DEFAULT_VAL_ROOT = Path(
    "/kaggle/input/datasets/narendarmallireddy/"
    "argoverse1-motion-dataset/forecasting_val_v1.1/val/data"
)
DEFAULT_MAP_ROOT = Path(
    "/kaggle/input/datasets/kedaradhikari/"
    "argoverse1-hd-mapss/hd_maps/map_files"
)


def _training_shards(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Assign scenes by deterministic greedy raw-file-size balancing."""

    bins: tuple[list[Path], list[Path]] = ([], [])
    estimated_bytes = [0, 0]
    for path in sorted(files, key=lambda item: (-item.stat().st_size, item.stem)):
        index = min(range(2), key=lambda candidate: (estimated_bytes[candidate], len(bins[candidate])))
        bins[index].append(path)
        estimated_bytes[index] += path.stat().st_size
    return (
        sorted(bins[0], key=lambda item: item.stem),
        sorted(bins[1], key=lambda item: item.stem),
    )


def _list_csv(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    files = sorted(root.glob("*.csv"), key=lambda item: item.stem)
    if not files:
        raise RuntimeError(f"No CSV scenes found in {root}")
    return files


def _output_root(split: str) -> Path:
    defaults = {
        "train-1": "/kaggle/working/dstnet-training-cache-part-1",
        "train-2": "/kaggle/working/dstnet-training-cache-part-2",
        "val": "/kaggle/working/dstnet-validation-cache",
    }
    return Path(os.environ.get("DSTNET_CACHE_OUTPUT", defaults[split]))


def build_cache(
    *,
    split: str,
    train_root: Path,
    val_root: Path,
    map_root: Path,
    output_root: Path,
    max_cache_gb: float,
) -> None:
    from datasets.cache_config import (
        CACHE_VERSION,
        KAGGLE_PREPROCESSING_CONFIG,
    )
    from datasets.cache_manager import CacheManager
    from datasets.map_loader import MapLoader
    from datasets.preprocess import ScenePreprocessor
    from datasets.scene_parser import SceneParser

    if max_cache_gb <= 0:
        raise ValueError("--max-cache-gb must be positive.")

    if split == "val":
        files = _list_csv(val_root)
    else:
        train_files = _list_csv(train_root)
        first, second = _training_shards(train_files)
        files = first if split == "train-1" else second

    if not map_root.is_dir():
        raise FileNotFoundError(f"HD map directory does not exist: {map_root}")

    cache_root = output_root / "cache"
    cache = CacheManager(cache_root)
    map_loader = MapLoader(map_root=map_root)
    parser = SceneParser(map_loader)
    preprocessor = ScenePreprocessor(**KAGGLE_PREPROCESSING_CONFIG)

    cache_ids = {path.stem for path in cache_root.glob("*.pkl")}
    expected_ids = {path.stem for path in files}
    unexpected = cache_ids - expected_ids
    if unexpected:
        raise RuntimeError(
            "Output cache already contains IDs outside this split: "
            + ", ".join(sorted(unexpected)[:10])
            + ". Choose a new --output-root."
        )

    cached_bytes = sum(path.stat().st_size for path in cache_root.glob("*.pkl"))
    max_bytes = int(max_cache_gb * 1_000_000_000)
    started = time.perf_counter()

    print(f"Split              : {split}")
    print(f"Scenes in split    : {len(files):,}")
    print(f"Already cached     : {len(cache_ids):,}")
    print(f"Output cache       : {cache_root}")
    print(f"Cache version      : {CACHE_VERSION}")
    print(f"Raw CSV size       : {sum(path.stat().st_size for path in files) / 1_000_000_000:.2f} GB")

    for index, csv_path in enumerate(files, start=1):
        if csv_path.stem not in cache_ids:
            raw_scene = parser.parse(csv_path)
            processed_scene = preprocessor.preprocess(raw_scene)
            if processed_scene.sequence_id != csv_path.stem:
                raise RuntimeError(
                    f"Processed ID mismatch for {csv_path.name}: "
                    f"{processed_scene.sequence_id!r}"
                )
            cache.save(processed_scene)
            cached_bytes += cache.cache_path(csv_path.stem).stat().st_size
            cache_ids.add(csv_path.stem)

            if cached_bytes > max_bytes:
                raise RuntimeError(
                    f"Cache exceeded --max-cache-gb={max_cache_gb:.2f} after "
                    f"{index:,} scenes ({cached_bytes / 1_000_000_000:.2f} GB). "
                    "This split is too large for one Kaggle output dataset; "
                    "resume into a larger-capacity environment or divide it "
                    "into additional shards. The completed scene files can "
                    "be retained by rerunning with the same output root."
                )

        if index == 1 or index % 250 == 0 or index == len(files):
            elapsed = time.perf_counter() - started
            print(
                f"[{index:>6,}/{len(files):,}] "
                f"cached={len(cache_ids):,} "
                f"size={cached_bytes / 1_000_000_000:.2f} GB "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    if cache_ids != expected_ids:
        missing = sorted(expected_ids - cache_ids)
        raise RuntimeError(
            f"Cache is incomplete ({len(missing):,} missing): "
            + ", ".join(missing[:10])
        )

    manifest = {
        "cache_version": CACHE_VERSION,
        "split": split,
        "scene_count": len(files),
        "cache_size_bytes": cached_bytes,
        "preprocessing": KAGGLE_PREPROCESSING_CONFIG,
        "sequence_ids": sorted(expected_ids),
        "complete": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)

    print(f"Complete cache     : {cache_root}")
    print(f"Manifest           : {manifest_path}")
    print(f"Total elapsed      : {(time.perf_counter() - started) / 3600:.2f} hours")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one complete, versioned Argoverse-1 cache dataset for Kaggle."
        ),
        epilog=(
            "Run separately with --split train-1, --split train-2, and "
            "--split val. Each output root is ready to publish as one Kaggle dataset."
        ),
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("train-1", "train-2", "val"),
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path(os.environ.get("DSTNET_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)),
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=Path(os.environ.get("DSTNET_VAL_ROOT", DEFAULT_VAL_ROOT)),
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=Path(os.environ.get("DSTNET_MAP_ROOT", DEFAULT_MAP_ROOT)),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Dataset root to write (contains cache/ and manifest.json).",
    )
    parser.add_argument(
        "--max-cache-gb",
        type=float,
        default=14.0,
        help="Stop after crossing this cache size; defaults below Kaggle's 15 GB working limit.",
    )
    args = parser.parse_args()

    build_cache(
        split=args.split,
        train_root=args.train_root,
        val_root=args.val_root,
        map_root=args.map_root,
        output_root=args.output_root or _output_root(args.split),
        max_cache_gb=args.max_cache_gb,
    )


if __name__ == "__main__":
    main()
