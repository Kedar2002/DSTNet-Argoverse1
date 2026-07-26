"""
scripts.test_scene_parser

Verifies the SceneParser using the actual Argoverse 1 dataset.

Pipeline
--------
CSV
    ↓
MapLoader
    ↓
SceneParser
    ↓
RawScene

Checks
------
✓ CSV loading
✓ Metadata parsing
✓ Track parsing
✓ Target agent detection
✓ AV detection
✓ Lane parsing
✓ Scene statistics
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import random
import sys

###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################

from datasets.map_loader import MapLoader
from datasets.scene_parser import SceneParser

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

###############################################################################
# Main
###############################################################################


def main() -> None:

    print("=" * 80)
    print("DSTNet - Scene Parser Verification")
    print("=" * 80)

    ###########################################################################
    # Dataset
    ###########################################################################

    csv_files = sorted(
        DATASET_ROOT.glob("*.csv")
    )

    if not csv_files:

        raise RuntimeError(
            f"No CSV files found in\n{DATASET_ROOT}"
        )

    print(f"\nDataset : {DATASET_ROOT}")
    print(f"Sequences : {len(csv_files)}")

    ###########################################################################
    # Pick one scene
    ###########################################################################

    csv_path = random.choice(csv_files)

    print(f"\nSelected Scene : {csv_path.name}")

    ###########################################################################
    # Load HD maps
    ###########################################################################

    print("\nLoading HD Maps...")

    map_loader = MapLoader(
        map_root=MAP_ROOT,
    )

    print("✓ HD maps loaded")

    ###########################################################################
    # Parser
    ###########################################################################

    parser = SceneParser(
        map_loader,
    )

    print("\nParsing scene...")

    scene = parser.parse(
        csv_path,
    )

    print("✓ Scene parsed")

    ###########################################################################
    # Metadata
    ###########################################################################

    print("\nMetadata")
    print("-" * 80)

    print("Sequence ID :", scene.metadata.sequence_id)
    print("City        :", scene.metadata.city)
    print("Target ID   :", scene.metadata.focal_track_id)

    ###########################################################################
    # Statistics
    ###########################################################################

    print("\nScene Statistics")
    print("-" * 80)

    print("Tracks :", scene.num_tracks)
    print("Lanes  :", scene.num_lanes)

    ###########################################################################
    # Target Agent
    ###########################################################################

    target = scene.target_track

    print("\nTarget Agent")
    print("-" * 80)

    print("Track ID :", target.track_id)
    print("Category :", target.category)
    print("Object   :", target.object_type)
    print("Length   :", target.length)

    ###########################################################################
    # AV
    ###########################################################################

    av = scene.av_track

    print("\nAutonomous Vehicle")
    print("-" * 80)

    if av is None:

        print("No AV track found.")

    else:

        print("Track ID :", av.track_id)
        print("Length   :", av.length)

    ###########################################################################
    # Sample Tracks
    ###########################################################################

    print("\nSample Tracks")
    print("-" * 80)

    for track in list(scene.tracks.values())[:5]:

        print(
            f"{track.track_id:<12}"
            f"{track.category:<10}"
            f"{track.length:>4}"
        )

    ###########################################################################
    # Sample Lanes
    ###########################################################################

    print("\nSample Lanes")
    print("-" * 80)

    for lane in list(scene.lanes.values())[:5]:

        print(f"Lane {lane.lane_id}")
        print(f"Points        : {lane.num_points}")
        print(f"Intersection  : {lane.is_intersection}")
        print(f"Turn          : {lane.turn_direction}")
        print(f"Traffic Ctrl  : {lane.has_traffic_control}")
        print()

    ###########################################################################
    # Summary
    ###########################################################################

    print("\nScene Summary")
    print("-" * 80)

    print(scene.summary())

    ###########################################################################
    # Final
    ###########################################################################

    print("\n" + "=" * 80)
    print("Scene Parser Verification Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
