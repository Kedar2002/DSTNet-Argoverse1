"""
datasets.transforms

Data augmentation transforms for DSTNet.

Transforms operate directly on RawScene before preprocessing.

Training:
    RawScene
        ↓
    Transform
        ↓
    Preprocessor

Validation/Test:
    RawScene
        ↓
    Preprocessor
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Iterable

import numpy as np

from datasets.geometry import rotate_points
from datasets.raw_scene import RawScene


###############################################################################
# Base Transform
###############################################################################


class Transform(ABC):
    """
    Base class for all transforms.
    """

    @abstractmethod
    def __call__(
        self,
        scene: RawScene,
    ) -> RawScene:
        """
        Apply transform.
        """
        raise NotImplementedError


###############################################################################
# Compose
###############################################################################


class Compose(Transform):
    """
    Compose multiple transforms.
    """

    def __init__(
        self,
        transforms: Iterable[Transform],
    ) -> None:

        self.transforms = list(transforms)

    def __call__(
        self,
        scene: RawScene,
    ) -> RawScene:

        for transform in self.transforms:

            scene = transform(scene)

        return scene

    ###############################################################################
# Random Rotation
###############################################################################


class RandomRotation(Transform):
    """
    Randomly rotate the complete scene.

    Rotation is applied to every trajectory and lane.
    """

    def __init__(
        self,
        max_angle_deg: float = 30.0,
        probability: float = 0.5,
    ) -> None:

        self.max_angle = np.deg2rad(
            max_angle_deg,
        )

        self.probability = probability

    def __call__(
        self,
        scene: RawScene,
    ) -> RawScene:

        if np.random.rand() > self.probability:
            return scene

        scene = deepcopy(scene)

        theta = np.random.uniform(
            -self.max_angle,
            self.max_angle,
        )

        for track in scene.tracks.values():

            track.positions = rotate_points(
                track.positions,
                theta,
            )

        for lane in scene.lanes.values():

            lane.centerline = rotate_points(
                lane.centerline,
                theta,
            )

        return scene

    ###############################################################################
# Random Translation
###############################################################################


class RandomTranslation(Transform):
    """
    Apply random XY translation.
    """

    def __init__(
        self,
        max_translation: float = 2.0,
        probability: float = 0.5,
    ) -> None:

        self.max_translation = max_translation

        self.probability = probability

    def __call__(
        self,
        scene: RawScene,
    ) -> RawScene:

        if np.random.rand() > self.probability:
            return scene

        scene = deepcopy(scene)

        offset = np.random.uniform(
            -self.max_translation,
            self.max_translation,
            size=2,
        ).astype(np.float32)

        for track in scene.tracks.values():

            track.positions += offset

        for lane in scene.lanes.values():

            lane.centerline += offset

        return scene

    ###############################################################################
# Identity
###############################################################################


class Identity(Transform):
    """
    No-op transform.
    """

    def __call__(
        self,
        scene: RawScene,
    ) -> RawScene:

        return scene


###############################################################################
# Factory
###############################################################################


def build_train_transform() -> Compose:
    """
    Default augmentation pipeline.
    """

    return Compose(
        [
            RandomRotation(),
            RandomTranslation(),
        ]
    )


def build_eval_transform() -> Compose:
    """
    Validation/Test pipeline.
    """

    return Compose(
        [
            Identity(),
        ]
    )


