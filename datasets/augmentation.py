"""Augment already-preprocessed scenes without invalidating cache files."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from torch.utils.data import Dataset


class RandomReflectionDataset(Dataset):
    """Randomly reflect local scene geometry across the x axis.

    This augmentation operates after cache loading, so cached canonical
    scenes remain immutable on disk and the transform also works when a
    dataset is served entirely from preprocessed ``SceneData`` objects.
    """

    def __init__(
        self,
        dataset: Dataset,
        probability: float = 0.5,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between zero and one.")
        self.dataset = dataset
        self.probability = float(probability)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _reflect(scene: Any) -> Any:
        """Reflect all stored local geometry; graph connectivity is invariant."""

        for agent in scene.agents:
            for key in (
                "observed",
                "future",
                "velocity",
                "acceleration",
                "last_position",
            ):
                value = agent.get(key)
                if isinstance(value, np.ndarray) and value.ndim > 0:
                    value[..., 1] *= -1.0

            for key in ("headings", "last_heading"):
                value = agent.get(key)
                if isinstance(value, np.ndarray):
                    value *= -1.0
                elif value is not None:
                    agent[key] = -float(value)

        for map_element in scene.maps:
            for key in ("centerline", "centroid", "direction"):
                value = map_element.get(key)
                if isinstance(value, np.ndarray) and value.ndim > 0:
                    value[..., 1] *= -1.0

            if map_element.get("heading") is not None:
                map_element["heading"] = -float(map_element["heading"])

        graph = scene.scene_graph
        graph.state_positions[..., 1] *= -1.0
        graph.state_headings *= -1.0
        graph.map_positions[..., 1] *= -1.0
        graph.map_headings *= -1.0
        return scene

    def __getitem__(self, index: int) -> Any:
        scene = self.dataset[index]
        if random.random() < self.probability:
            scene = self._reflect(scene)
        return scene


__all__ = ["RandomReflectionDataset"]
