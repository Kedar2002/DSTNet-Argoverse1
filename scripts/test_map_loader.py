"""
scripts.test_map_loader

Verifies the custom Argoverse-1 HD map loader.

Checks
------
✓ Load all HD maps
✓ Print discovered cities
✓ Print lane statistics
✓ Validate topology
✓ KD-tree spatial queries
✓ Inspect one lane
"""

from __future__ import annotations

from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Allow running from repository root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------

from datasets.map_loader import MapLoader


###############################################################################
# Configuration
###############################################################################

# Change this only if your HD maps are stored elsewhere.
MAP_ROOT = PROJECT_ROOT / "data" / "argoverse1" / "hd_maps" / "map_files"


###############################################################################
# Main
###############################################################################


def main() -> None:

    print("=" * 80)
    print("DSTNet - Map Loader Verification")
    print("=" * 80)

    print(f"\nHD Map directory : {MAP_ROOT}")

    if not MAP_ROOT.exists():
        raise FileNotFoundError(
            f"HD map directory not found:\n{MAP_ROOT}"
        )

    ###################################################################
    # Load maps
    ###################################################################

    print("\nLoading maps...")

    loader = MapLoader(
        map_root=MAP_ROOT,
    )

    print("✓ Maps loaded successfully")

    ###################################################################
    # Summary
    ###################################################################

    print("\nLoader Summary")
    print("-" * 80)

    loader.print_summary()

    print()

    print(loader.statistics())

    ###################################################################
    # City information
    ###################################################################

    print("\nCities")
    print("-" * 80)

    for city in loader.cities:

        summary = loader.city_summary(city)

        print(summary)

    ###################################################################
    # Topology validation
    ###################################################################

    print("\nTopology Validation")
    print("-" * 80)

    for city in loader.cities:

        ok = loader.check_topology(city)

        print(f"{city:<8} : {'PASS' if ok else 'FAIL'}")

    ###################################################################
    # Lane inspection
    ###################################################################

    print("\nInspecting first lane from every city")
    print("-" * 80)

    for city in loader.cities:

        lane_ids = loader.list_lane_ids(city)

        if not lane_ids:

            print(f"{city}: No lanes")

            continue

        lane_id = lane_ids[0]

        print(f"\nCity : {city}")

        print(loader.inspect_lane(lane_id, city))

    ###################################################################
    # Spatial query
    ###################################################################

    print("\nSpatial Query Test")
    print("-" * 80)

    for city in loader.cities:

        lane_ids = loader.list_lane_ids(city)

        if not lane_ids:
            continue

        lane = loader.get_lane_segment(
            lane_ids[0],
            city,
        )

        if lane is None:
            print(f"{city}: Failed to retrieve lane.")
            continue

        centroid = lane.centroid

        nearby = loader.get_lane_ids_in_xy_bbox(
            x=float(centroid[0]),
            y=float(centroid[1]),
            city=city,
            query_search_range_manhattan=20.0,
        )

        print()

        print(f"City      : {city}")

        print(f"Lane      : {lane.lane_id}")

        print(f"Centroid  : {centroid}")

        print(f"Nearby    : {len(nearby)} lanes")

    ###################################################################
    # Nearest lane
    ###################################################################

    print("\nNearest Lane Test")
    print("-" * 80)

    for city in loader.cities:

        lane_ids = loader.list_lane_ids(city)

        if not lane_ids:
            continue

        lane = loader.get_lane_segment(
            lane_ids[0],
            city,
        )

        if lane is None:
            print(f"{city}: Failed to retrieve lane.")
            continue

        point = lane.centroid

        nearest = loader.get_nearest_lane(
            x=float(point[0]),
            y=float(point[1]),
            city=city,
        )

        print()

        print(f"City : {city}")

        print(
            "Expected :",
            lane.lane_id,
        )

        if nearest is None:

            print("Nearest  : None")

        else:

            print("Nearest  :", nearest.lane_id)

    ###################################################################
    # Final
    ###################################################################

    print("\n" + "=" * 80)
    print("Map Loader Verification Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
