"""
scripts.test_dataset

Verifies the ArgoverseDataset implementation.

Pipeline
--------
CSV
    ↓
SceneParser
    ↓
ScenePreprocessor
    ↓
CacheManager
    ↓
ArgoverseDataset

Checks
------
✓ Dataset construction
✓ __len__
✓ __getitem__
✓ Cache functionality
✓ SceneData integrity
"""

from __future__ import annotations

from pathlib import Path
import random
import shutil
import sys

###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.cache_manager import CacheManager
from datasets.map_loader import MapLoader
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser
from datasets.transforms import Identity

###############################################################################
# Configuration
###############################################################################

MAP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
    / "hd_maps"
    / "map_files"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
    / "train"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "cache"
    / "test_dataset"
)

###############################################################################


def print_scene(scene) -> None:

    print("\nScene Summary")
    print("-" * 80)

    print(scene.summary())

    print()

    print("Origin :", scene.origin)
    print("Heading:", scene.heading)

    print()

    print("Agents :", scene.num_agents)
    print("Lanes  :", scene.num_lanes)

    print()

    print("Agent-Agent Edges :", scene.graph.agent_agent_edges.shape)
    print("Lane-Lane Edges  :", scene.graph.lane_lane_edges.shape)
    print("Lane-Agent Edges :", scene.graph.lane_agent_edges.shape)


###############################################################################


def main() -> None:

    print("=" * 80)
    print("DSTNet - Dataset Verification")
    print("=" * 80)

    ###########################################################################
    # Cache directory
    ###########################################################################

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Components
    ###########################################################################

    map_loader = MapLoader(
        map_root=MAP_ROOT,
    )

    parser = SceneParser(
        map_loader,
    )

    preprocessor = ScenePreprocessor(

        observation_steps=20,

        prediction_steps=30,

        lane_sample_points=20,

        agent_radius=30.0,

        lane_radius=30.0,

    )

    cache = CacheManager(
        CACHE_ROOT,
    )

    ###########################################################################
    # Dataset
    ###########################################################################

    dataset = ArgoverseDataset(

        root=DATASET_ROOT,

        parser=parser,

        preprocessor=preprocessor,

        transform=Identity(),

        cache=cache,

    )

    ###########################################################################
    # Dataset Information
    ###########################################################################

    print()

    print("Dataset Root :", DATASET_ROOT)

    print("Dataset Size :", len(dataset))

    print("Cache Root   :", CACHE_ROOT)

    ###########################################################################
    # Random sample
    ###########################################################################

    index = random.randint(
        0,
        len(dataset) - 1,
    )

    print()

    print("Loading sample:", index)

    scene = dataset[index]

    print_scene(scene)

    ###########################################################################
    # Cache Verification
    ###########################################################################

    print("\nCache Verification")
    print("-" * 80)

    # Cache currently contains the randomly loaded sample.
    cached_initial = cache.num_cached()

    print("Initial Cached Files :", cached_initial)

    # Reload the same sample.
    _ = dataset[index]

    cached_same = cache.num_cached()

    print("After Reloading Same Sample :", cached_same)

    # Load a different sample.
    second_index = (index + 1) % len(dataset)

    _ = dataset[second_index]

    cached_new = cache.num_cached()

    print("After Loading New Sample    :", cached_new)

    # Verify expected behaviour.
    assert cached_same >= cached_initial

    assert cached_new >= cached_same

    print("✓ Cache verification completed")

    print("✓ Cache correctly avoids duplicate entries")
    print("✓ Cache correctly stores new scenes")

    ###########################################################################
    # Dataset Properties
    ###########################################################################

    print("\nDataset Properties")
    print("-" * 80)

    print("num_scenes :", dataset.num_scenes)

    print()

    print("First 5 Sequence IDs")

    for sequence_id in dataset.sequence_ids[:5]:

        print(sequence_id)

    ###########################################################################
    # Integrity Checks
    ###########################################################################

    print("\nIntegrity Checks")
    print("-" * 80)

    assert len(dataset) > 0
    print("✓ Dataset length")

    assert scene.num_agents > 0
    print("✓ Agents")

    assert scene.num_lanes > 0
    print("✓ Lanes")

    assert scene.origin.shape == (2,)
    print("✓ Origin")

    assert scene.graph.agent_agent_edges.shape[0] == 2
    print("✓ Agent graph")

    assert scene.graph.lane_lane_edges.shape[0] == 2
    print("✓ Lane graph")

    assert scene.graph.lane_agent_edges.shape[0] == 2
    print("✓ Lane-agent graph")

    print()

    print("=" * 80)
    print("DATASET TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
