from __future__ import annotations

import numpy as np
import torch

from datasets.preprocess import ScenePreprocessor
from engine.utils import select_supervised_agents
from models.model_types import Prediction
from scripts.prepare_kaggle_cache import _training_shards


def test_history_positions_require_matching_focal_timestamps() -> None:
    timestamps = np.arange(50, dtype=np.float64) / 10.0
    requested = timestamps[:20]
    positions = np.stack((timestamps, -timestamps), axis=-1).astype(np.float32)

    aligned = ScenePreprocessor._align_positions(
        timestamps,
        positions,
        requested,
    )
    shifted = ScenePreprocessor._align_positions(
        timestamps[5:],
        positions[5:],
        requested,
    )

    assert aligned is not None
    np.testing.assert_array_equal(aligned, positions[:20])
    assert shifted is None


def test_future_alignment_stops_at_first_missing_timestamp() -> None:
    timestamps = np.concatenate(
        (np.arange(21, dtype=np.float64), np.arange(22, 50, dtype=np.float64))
    )
    positions = np.stack((timestamps, timestamps), axis=-1).astype(np.float32)

    future = ScenePreprocessor._align_future_prefix(
        timestamps,
        positions,
        np.arange(20, 25, dtype=np.float64),
    )

    np.testing.assert_array_equal(future[:, 0], np.array([20.0], dtype=np.float32))


def test_training_supervision_uses_only_final_history_and_valid_agents() -> None:
    trajectories = torch.zeros(2, 3, 4, 6, 30, 2)
    for history_index in range(4):
        trajectories[:, :, history_index] = float(history_index)
    probabilities = torch.full((2, 3, 4, 6), 1.0 / 6.0)
    prediction = Prediction(trajectories, probabilities)
    ground_truth = torch.zeros(2, 3, 30, 2)
    mask = torch.tensor([[True, False, True], [False, True, False]])

    selected, refined, selected_gt = select_supervised_agents(
        prediction,
        None,
        ground_truth,
        mask,
    )

    assert refined is None
    assert selected.trajectories.shape == (1, 3, 1, 6, 30, 2)
    assert selected.probabilities.shape == (1, 3, 1, 6)
    assert torch.all(selected.trajectories == 3.0)
    assert selected_gt.shape == (1, 3, 30, 2)


def test_training_shards_are_deterministic_balanced_and_disjoint(tmp_path) -> None:
    files = []
    for index, size in enumerate((9, 8, 7, 6, 5, 4, 3, 2)):
        path = tmp_path / f"{index:02d}.csv"
        path.write_bytes(b"x" * size)
        files.append(path)

    first, second = _training_shards(files)
    first_again, second_again = _training_shards(list(reversed(files)))

    assert [path.name for path in first] == [path.name for path in first_again]
    assert [path.name for path in second] == [path.name for path in second_again]
    assert {path.stem for path in first}.isdisjoint(path.stem for path in second)
    assert {path.stem for path in first + second} == {path.stem for path in files}
    size_a = sum(path.stat().st_size for path in first)
    size_b = sum(path.stat().st_size for path in second)
    assert abs(size_a - size_b) <= max(path.stat().st_size for path in files)
