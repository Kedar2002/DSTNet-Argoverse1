"""
datasets.cache_manager

Disk cache for processed SceneData objects.

The cache avoids repeatedly parsing and preprocessing the
Argoverse dataset during training.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

from datasets.scene_data import SceneData


CACHE_VERSION = "1.0"


class CacheManager:
    """
    Manage cached SceneData objects.
    """

    def __init__(
        self,
        cache_root: str | Path,
    ) -> None:

        self.cache_root = Path(cache_root)

        self.cache_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.version_file = (
            self.cache_root / "VERSION"
        )

        self._write_version()

    ###########################################################################
    # Version
    ###########################################################################

    def _write_version(self) -> None:

        if self.version_file.exists():
            return

        self.version_file.write_text(
            CACHE_VERSION,
            encoding="utf-8",
        )

        ###########################################################################
    # Cache Paths
    ###########################################################################

    def cache_path(
        self,
        sequence_id: str,
    ) -> Path:

        return (
            self.cache_root
            / f"{sequence_id}.pkl"
        )

    def exists(
        self,
        sequence_id: str,
    ) -> bool:

        return self.cache_path(
            sequence_id
        ).exists()

    ###########################################################################
    # Save
    ###########################################################################

    def save(
        self,
        scene: SceneData,
    ) -> None:

        path = self.cache_path(
            scene.sequence_id
        )

        with open(
            path,
            "wb",
        ) as f:

            pickle.dump(
                scene,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        ###########################################################################
    # Load
    ###########################################################################

    def load(
        self,
        sequence_id: str,
    ) -> SceneData:

        path = self.cache_path(
            sequence_id,
        )

        with open(
            path,
            "rb",
        ) as f:

            scene = pickle.load(f)

        if not isinstance(
            scene,
            SceneData,
        ):
            raise TypeError(
                "Invalid cached object."
            )

        return scene

    ###########################################################################
    # Remove
    ###########################################################################

    def remove(
        self,
        sequence_id: str,
    ) -> None:

        path = self.cache_path(
            sequence_id,
        )

        if path.exists():
            path.unlink()

        ###########################################################################
    # Utilities
    ###########################################################################

    def clear(self) -> None:

        if self.cache_root.exists():

            shutil.rmtree(
                self.cache_root,
            )

        self.cache_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._write_version()

    def num_cached(self) -> int:

        return len(
            list(
                self.cache_root.glob(
                    "*.pkl"
                )
            )
        )

    def cache_size(self) -> int:

        total = 0

        for file in self.cache_root.glob(
            "*.pkl"
        ):

            total += file.stat().st_size

        return total

    def summary(self) -> dict:

        return {
            "cache_root": str(
                self.cache_root
            ),
            "cached_files": self.num_cached(),
            "cache_size_bytes": self.cache_size(),
            "version": CACHE_VERSION,
        }


