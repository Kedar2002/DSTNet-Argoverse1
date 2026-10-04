"""Preprocess local Argoverse CSV scenes into versioned SceneData caches."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def preprocess_splits(
    *,
    splits: list[str],
    cache_root: Path,
    dataset_config: dict,
) -> None:
    from datasets.cache_manager import CacheManager
    from datasets.map_loader import MapLoader
    from datasets.preprocess import ScenePreprocessor
    from datasets.scene_parser import SceneParser
    from datasets.transforms import Identity

    dataset_root = _resolve(dataset_config["root"])
    map_root = dataset_root / "hd_maps" / "map_files"
    if not map_root.is_dir():
        raise FileNotFoundError(f"HD map directory does not exist: {map_root}")

    map_loader = MapLoader(map_root)
    parser = SceneParser(map_loader)
    preprocessor = ScenePreprocessor(
        observation_steps=int(dataset_config["observation_steps"]),
        prediction_steps=int(dataset_config["prediction_steps"]),
        map_sample_points=int(dataset_config["map_sample_points"]),
        spatial_radius=float(dataset_config["spatial_radius"]),
        map_radius=float(dataset_config["map_radius"]),
        frame_rate=float(dataset_config.get("frame_rate", 10.0)),
    )
    cache = CacheManager(cache_root)
    transform = Identity()

    total = 0
    created = 0
    cache_bytes = cache.cache_size()
    started = time.perf_counter()
    for split in splits:
        split_root = _resolve(dataset_config[f"{split}_dir"])
        if not split_root.is_dir():
            raise FileNotFoundError(f"{split} directory does not exist: {split_root}")
        files = sorted(split_root.glob("*.csv"), key=lambda path: path.stem)
        if not files:
            raise RuntimeError(f"No CSV scenes found for {split} in {split_root}")

        print(f"{split}: {len(files):,} scenes from {split_root}")
        for index, csv_path in enumerate(files, start=1):
            total += 1
            if cache.exists(csv_path.stem):
                continue
            scene = transform(parser.parse(csv_path))
            processed = preprocessor.preprocess(scene)
            cache.save(processed)
            created += 1
            cache_bytes += cache.cache_path(csv_path.stem).stat().st_size

            if created == 1 or created % 250 == 0:
                print(
                    f"processed={created:,} total_seen={total:,} "
                    f"cache_size={cache_bytes / 1024**3:.2f} GiB "
                    f"elapsed={(time.perf_counter() - started) / 60:.1f} min",
                    flush=True,
                )

        print(f"{split}: finished {len(files):,} scenes")

    print(f"Created scenes : {created:,}")
    print(f"Cache entries  : {cache.num_cached():,}")
    print(f"Cache size     : {cache_bytes / 1024**3:.2f} GiB")
    print(f"Cache version  : {cache.summary()['version']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dataset.yaml",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the configured cache directory (recommended for rebuilds).",
    )
    args = parser.parse_args()

    import yaml

    from datasets.cache_config import CACHE_VERSION

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dataset_config = config["dataset"]
    cache_root = args.cache_dir or _resolve(dataset_config["cache_dir"])
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    print(f"Cache output : {cache_root}")
    print(f"Cache format : {CACHE_VERSION}")
    print("Existing entries are preserved and skipped; use a new --cache-dir to rebuild.")
    preprocess_splits(
        splits=splits,
        cache_root=cache_root,
        dataset_config=dataset_config,
    )


if __name__ == "__main__":
    main()
