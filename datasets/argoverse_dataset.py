"""
datasets.argoverse_dataset

PyTorch Dataset for Argoverse 1 Motion Forecasting.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from datasets.cache_manager import CacheManager
from datasets.preprocess import ScenePreprocessor
from datasets.scene_data import SceneData
from datasets.scene_parser import SceneParser
from datasets.transforms import (
    Identity,
    Transform,
)


class ArgoverseDataset(Dataset):
    """
    PyTorch Dataset for Argoverse 1.
    """

    def __init__(
        self,
        root: str | Path,
        parser: SceneParser,
        preprocessor: ScenePreprocessor,
        transform: Transform | None = None,
        cache: CacheManager | None = None,
    ) -> None:

        self.root = Path(root)

        self.parser = parser

        self.preprocessor = preprocessor

        self.transform = (
            transform
            if transform is not None
            else Identity()
        )

        self.cache = cache

        self.files = sorted(
            self.root.glob("*.csv")
        )

        if not self.files:

            raise RuntimeError(
                f"No CSV files found in {self.root}"
            )

        ###########################################################################
    # Dataset
    ###########################################################################

    def __len__(
        self,
    ) -> int:

        return len(self.files)

    def __getitem__(
        self,
        index: int,
    ) -> SceneData:

        csv_path = self.files[index]

        sequence_id = csv_path.stem

        #######################################################################
        # Cache
        #######################################################################

        if (
            self.cache is not None
            and self.cache.exists(sequence_id)
        ):

            return self.cache.load(
                sequence_id
            )

        #######################################################################
        # Parse
        #######################################################################

        scene = self.parser.parse(
            csv_path,
        )

        #######################################################################
        # Transform
        #######################################################################

        scene = self.transform(
            scene,
        )

        #######################################################################
        # Preprocess
        #######################################################################

        processed = self.preprocessor.preprocess(
            scene,
        )

        #######################################################################
        # Cache
        #######################################################################

        if self.cache is not None:

            self.cache.save(
                processed,
            )

        return processed

        ###########################################################################
    # Utilities
    ###########################################################################

    @property
    def num_scenes(
        self,
    ) -> int:

        return len(self)

    @property
    def sequence_ids(
        self,
    ) -> list[str]:

        return [
            file.stem
            for file in self.files
        ]

    def summary(
        self,
    ) -> dict:

        return {
            "root": str(self.root),
            "num_scenes": len(self),
            "cache_enabled": self.cache is not None,
        }

    def __repr__(
        self,
    ) -> str:

        return (
            "ArgoverseDataset("
            f"splits={len(self)}, "
            f"root='{self.root}')"
        )
