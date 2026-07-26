"""
scripts.test_preprocessor

Verifies the preprocessing pipeline using the actual
Argoverse-1 Motion Forecasting dataset.

Pipeline
--------
CSV
    ↓
SceneParser
    ↓
RawScene
    ↓
ScenePreprocessor
    ↓
SceneData

Checks
------
✓ Map loading
✓ Scene parsing
✓ Reference frame
✓ Agent preprocessing
✓ Lane preprocessing
✓ Graph construction
✓ SceneData integrity
"""

from __future__ import annotations

from pathlib import Path
import random
import sys

###############################################################################
# Repository
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################

from datasets.map_loader import MapLoader
from datasets.scene_parser import SceneParser
from datasets.preprocess import ScenePreprocessor

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


def print_agent(agent: dict, index: int) -> None:

    print(f"\nAgent {index}")

    print("-" * 60)

    print("Track ID      :", agent["track_id"])
    print("Category      :", agent["category"])
    print("Observed      :", agent["observed"].shape)
    print("Future        :", agent["future"].shape)
    print("Velocity      :", agent["velocity"].shape)
    print("Speed         :", agent["speed"].shape)
    print("Acceleration  :", agent["acceleration"].shape)
    print("Heading       :", agent["heading"].shape)


def print_lane(lane: dict, index: int) -> None:

    print(f"\nLane {index}")

    print("-" * 60)

    print("Lane ID           :", lane["lane_id"])
    print("Centerline Shape  :", lane["centerline"].shape)
    print("Direction Shape   :", lane["direction"].shape)
    print("Turn              :", lane["turn_direction"])
    print("Intersection      :", lane["is_intersection"])
    print("Traffic Control   :", lane["has_traffic_control"])


###############################################################################


def main() -> None:

    print("=" * 80)
    print("DSTNet - Preprocessor Verification")
    print("=" * 80)

    ###########################################################################

    csv_files = sorted(DATASET_ROOT.glob("*.csv"))

    if not csv_files:

        raise RuntimeError(
            f"No csv files found in\n{DATASET_ROOT}"
        )

    csv_path = random.choice(csv_files)

    print("\nScene :", csv_path.name)

    ###########################################################################
    # Map Loader
    ###########################################################################

    loader = MapLoader(
        map_root=MAP_ROOT,
    )

    ###########################################################################
    # Parser
    ###########################################################################

    parser = SceneParser(
        loader,
    )

    scene = parser.parse(
        csv_path,
    )

    ###########################################################################
    # Preprocessor
    ###########################################################################

    preprocessor = ScenePreprocessor(

        observation_steps=20,

        prediction_steps=30,

        lane_sample_points=20,

        agent_radius=30.0,

        lane_radius=30.0,

    )

    processed = preprocessor.preprocess(
        scene,
    )

    ###########################################################################
    # Scene Information
    ###########################################################################

    print("\nProcessed Scene")
    print("-" * 80)

    print("Sequence ID :", processed.sequence_id)
    print("City        :", processed.city)

    print()

    print("Origin  :", processed.origin)

    print("Heading :", processed.heading)

    print()

    print("Agents :", processed.num_agents)

    print("Lanes  :", processed.num_lanes)

    ###########################################################################
    # Graph
    ###########################################################################

    print("\nGraph")

    print("-" * 80)

    print(
        "Agent-Agent Edges :",
        processed.graph.agent_agent_edges.shape,
    )

    print(
        "Lane-Lane Edges :",
        processed.graph.lane_lane_edges.shape,
    )

    print(
        "Lane-Agent Edges :",
        processed.graph.lane_agent_edges.shape,
    )

    ###########################################################################
    # Sample Agents
    ###########################################################################

    print("\nSample Agents")

    print("=" * 80)

    for i, agent in enumerate(processed.agents[:3]):

        print_agent(agent, i)

    ###########################################################################
    # Sample Lanes
    ###########################################################################

    print("\nSample Lanes")

    print("=" * 80)

    for i, lane in enumerate(processed.lanes[:3]):

        print_lane(lane, i)

    ###########################################################################
    # Scene Summary
    ###########################################################################

    print("\nScene Summary")

    print("-" * 80)

    print(processed.summary())

    ###########################################################################
    # Integrity Checks
    ###########################################################################

    print("\nIntegrity Checks")

    print("-" * 80)

    assert processed.origin.shape == (2,)
    print("✓ Origin")

    assert isinstance(processed.heading, float)
    print("✓ Heading")

    assert processed.num_agents > 0
    print("✓ Agents")

    assert processed.num_lanes > 0
    print("✓ Lanes")

    assert processed.graph.agent_agent_edges.shape[0] == 2
    print("✓ Agent Graph")

    assert processed.graph.lane_lane_edges.shape[0] == 2
    print("✓ Lane Graph")

    assert processed.graph.lane_agent_edges.shape[0] == 2
    print("✓ Lane-Agent Graph")

    print()

    print("=" * 80)

    print("PREPROCESSOR TEST PASSED")

    print("=" * 80)


if __name__ == "__main__":
    main()
