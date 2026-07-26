"""
scripts.visualize_scene

Visualize Argoverse 1 scenes.

Supports

    • Raw scene visualization
    • Processed scene visualization
    • Side-by-side comparison

Usage
-----

Raw scene

python -m scripts.visualize_scene \
    --scene data/argoverse1/train/1.csv \
    --mode raw

Processed scene

python -m scripts.visualize_scene \
    --scene data/argoverse1/train/1.csv \
    --mode processed

Both

python -m scripts.visualize_scene \
    --scene data/argoverse1/train/1.csv \
    --mode both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from datasets.map_loader import MapLoader
from datasets.scene_parser import SceneParser
from datasets.preprocess import ScenePreprocessor

# These will be implemented next
from visualization.raw_scene_plotter import plot_raw_scene
from visualization.processed_scene_plotter import plot_processed_scene

###############################################################################
# Project Paths
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
)

MAP_ROOT = (
    DATA_ROOT
    / "hd_maps"
    / "map_files"
)

###############################################################################
# Defaults
###############################################################################

OBSERVATION_STEPS = 20
PREDICTION_STEPS = 30

LANE_SAMPLE_POINTS = 20

AGENT_RADIUS = 30.0
LANE_RADIUS = 40.0


###############################################################################
# Argument Parser
###############################################################################

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="DSTNet Scene Visualizer"
    )

    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Path to Argoverse CSV scene.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=[
            "raw",
            "processed",
            "both",
        ],
        help="Visualization mode.",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save figure path.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--show_ids",
        action="store_true",
    )

    return parser

###############################################################################
# Build Parser
###############################################################################

def build_parser() -> SceneParser:

    map_loader = MapLoader(

        map_root=MAP_ROOT,

    )

    parser = SceneParser(

        map_loader,

    )

    return parser

###############################################################################
# Load Scene
###############################################################################

def load_scene(
    scene_path: Path,
):

    print()

    print("=" * 80)
    print("Loading Scene")
    print("=" * 80)

    parser = build_parser()

    raw_scene = parser.parse(

        scene_path,

    )

    print(

        f"Sequence : "

        f"{raw_scene.metadata.sequence_id}"

    )

    print(

        f"City     : "

        f"{raw_scene.metadata.city}"

    )

    print(

        f"Tracks   : "

        f"{raw_scene.num_tracks}"

    )

    print(

        f"Lanes    : "

        f"{raw_scene.num_lanes}"

    )

    ###############################################################
    # Process Scene
    ###############################################################

    preprocessor = build_preprocessor()

    processed_scene = preprocessor.preprocess(

        raw_scene,

    )

    return (

        raw_scene,

        processed_scene,

    )


###############################################################################
# Build Preprocessor
###############################################################################

def build_preprocessor():

    return ScenePreprocessor(

        observation_steps=OBSERVATION_STEPS,

        prediction_steps=PREDICTION_STEPS,

        lane_sample_points=LANE_SAMPLE_POINTS,

        agent_radius=AGENT_RADIUS,

        lane_radius=LANE_RADIUS,

    )

###############################################################################
# Figure
###############################################################################

def build_figure(
    mode: str,
):

    if mode == "both":

        figure, axes = plt.subplots(

            1,

            2,

            figsize=(16, 8),

        )

        return figure, axes

    figure, axis = plt.subplots(

        figsize=(9, 9),

    )

    return figure, axis

###############################################################################
# Visualization
###############################################################################

def visualize(

    raw_scene,

    processed_scene,

    *,

    mode: str,

    show_ids: bool,

):

    figure, axes = build_figure(

        mode,

    )

    if mode == "raw":

        plot_raw_scene(

            raw_scene,

            ax=axes,

            show_ids=show_ids,

        )

    elif mode == "processed":

        plot_processed_scene(

            processed_scene,

            ax=axes,

            show_ids=show_ids,

        )

    else:

        plot_raw_scene(

            raw_scene,

            ax=axes[0],

            show_ids=show_ids,

        )

        plot_processed_scene(

            processed_scene,

            ax=axes[1],

            show_ids=show_ids,

        )

    return figure

###############################################################################
# Save Figure
###############################################################################

def save_figure(
    path: Path,
    dpi: int,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    print()
    print(f"Saved : {path}")

###############################################################################
# Main
###############################################################################

def main():

    args = build_argument_parser().parse_args()

    raw_scene, processed_scene = load_scene(

        Path(args.scene),

    )

    figure = visualize(

        raw_scene,

        processed_scene,

        mode=args.mode,

        show_ids=args.show_ids,

    )

    if args.save is not None:

        save_figure(

            Path(args.save),

            args.dpi,

        )

    plt.show()


if __name__ == "__main__":

    main()
