"""
scripts.test_collate

Verifies the DSTNet batch collation pipeline.

Pipeline
--------
CSV
    ↓
SceneParser
    ↓
ScenePreprocessor
    ↓
ArgoverseDataset
    ↓
DataLoader
    ↓
collate_fn

Checks
------
✓ Dataset loading
✓ DataLoader
✓ Batch collation
✓ Tensor shapes
✓ Masks
✓ Graph batching
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.cache_manager import CacheManager
from datasets.collate import collate_fn
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
    / "test_collate"
)

BATCH_SIZE = 4

###############################################################################


def print_tensor(name: str, tensor: torch.Tensor) -> None:

    print(
        f"{name:<25}"
        f"{tuple(tensor.shape)}"
        f"    {tensor.dtype}"
    )


###############################################################################


def main() -> None:

    print("=" * 80)
    print("DSTNet - Collate Verification")
    print("=" * 80)

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

    dataset = ArgoverseDataset(

        root=DATASET_ROOT,

        parser=parser,

        preprocessor=preprocessor,

        transform=Identity(),

        cache=cache,

    )

    ###########################################################################
    # DataLoader
    ###########################################################################

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        collate_fn=collate_fn,

    )

    ###########################################################################
    # First Batch
    ###########################################################################

    batch = next(iter(loader))

    print("\nTensor Shapes")
    print("-" * 80)

    print_tensor(
        "agent_trajectories",
        batch["agent_trajectories"],
    )

    print_tensor(
        "future_trajectories",
        batch["future_trajectories"],
    )

    print_tensor(
        "lane_centerlines",
        batch["lane_centerlines"],
    )

    print_tensor(
        "positions",
        batch["positions"],
    )

    print_tensor(
        "headings",
        batch["headings"],
    )

    print_tensor(
        "agent_mask",
        batch["agent_mask"],
    )

    print_tensor(
        "lane_mask",
        batch["lane_mask"],
    )

    ###########################################################################
    # Graphs
    ###########################################################################

    print("\nGraphs")
    print("-" * 80)

    graphs = batch["graph"]

    print("Graphs in batch :", len(graphs))

    for i, graph in enumerate(graphs):

        print()

        print(f"Scene {i}")

        print(
            "Agent-Agent :",
            graph.agent_agent_edges.shape,
        )

        print(
            "Lane-Lane   :",
            graph.lane_lane_edges.shape,
        )

        print(
            "Lane-Agent  :",
            graph.lane_agent_edges.shape,
        )

    ###########################################################################
    # Metadata
    ###########################################################################

    print("\nMetadata")
    print("-" * 80)

    metadata = batch["metadata"]

    print("Sequence IDs :", metadata["sequence_ids"])
    print("Cities       :", metadata["cities"])

    print()

    print_tensor(
        "origins",
        metadata["origins"],
    )

    print_tensor(
        "scene_headings",
        metadata["scene_headings"],
    )

    ###########################################################################
    # Integrity Checks
    ###########################################################################

    print("\nIntegrity Checks")
    print("-" * 80)

    assert batch["agent_trajectories"].ndim == 4
    print("✓ agent_trajectories")

    assert batch["future_trajectories"].ndim == 4
    print("✓ future_trajectories")

    assert batch["lane_centerlines"].ndim == 4
    print("✓ lane_centerlines")

    assert batch["positions"].ndim == 3
    print("✓ positions")

    assert batch["headings"].ndim == 2
    print("✓ headings")

    assert batch["agent_mask"].dtype == torch.bool
    print("✓ agent_mask")

    assert batch["lane_mask"].dtype == torch.bool
    print("✓ lane_mask")

    assert len(batch["graph"]) == BATCH_SIZE
    print("✓ graph list")

    print()

    print("=" * 80)
    print("COLLATE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
