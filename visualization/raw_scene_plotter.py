"""
visualization.raw_scene_plotter

Publication-quality visualization of an Argoverse 1 RawScene.

Displays

- HD lane centerlines
- Target agent
- Autonomous Vehicle (AV)
- Other traffic participants
- Observed trajectories
- Future trajectories
- Direction arrows

Coordinates are shown in the original world frame.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from datasets.raw_scene import (
    RawScene,
    RawTrack,
)

###############################################################################
# Plot Style
###############################################################################

LANE_COLOR = "#D6D6D6"

TARGET_OBSERVED = "#0B7A75"      # teal
TARGET_FUTURE = "#C62828"        # red

AV_COLOR = "#F4A641"

VEHICLE_COLOR = "#707070"

PEDESTRIAN_COLOR = "#6AA84F"

CYCLIST_COLOR = "#D68910"

BUS_COLOR = "#7E57C2"

UNKNOWN_COLOR = "#8A8A8A"

###############################################################################
# Drawing Parameters
###############################################################################

OBSERVED_STEPS = 20

LANE_WIDTH = 0.8

TARGET_WIDTH = 2.8

AV_WIDTH = 2.2

OTHER_WIDTH = 1.2

ARROW_EVERY = 4

SCENE_MARGIN = 5.0

###############################################################################
# Utility
###############################################################################


def actor_colour(
    track: RawTrack,
) -> str:
    """
    Return colour for an actor.
    """

    if track.is_target:
        return TARGET_OBSERVED

    if track.is_av:
        return AV_COLOR

    mapping = {

        "VEHICLE": VEHICLE_COLOR,

        "PEDESTRIAN": PEDESTRIAN_COLOR,

        "CYCLIST": CYCLIST_COLOR,

        "BUS": BUS_COLOR,

    }

    return mapping.get(

        track.object_type.upper(),

        UNKNOWN_COLOR,

    )


def actor_width(
    track: RawTrack,
) -> float:

    if track.is_target:
        return TARGET_WIDTH

    if track.is_av:
        return AV_WIDTH

    return OTHER_WIDTH


###############################################################################
# Direction Arrows
###############################################################################


def draw_direction_arrows(
    ax,
    trajectory: np.ndarray,
    colour: str,
    *,
    every: int = ARROW_EVERY,
    scale: float = 12.0,
):
    """
    Draw arrows indicating travel direction.
    """

    if len(trajectory) < 2:
        return

    for i in range(

        every,

        len(trajectory),

        every,

    ):

        p0 = trajectory[i - 1]

        p1 = trajectory[i]

        ax.annotate(

            "",

            xy=p1,

            xytext=p0,

            arrowprops=dict(

                arrowstyle="-|>",

                color=colour,

                lw=1.0,

                shrinkA=0,

                shrinkB=0,

                mutation_scale=scale,

            ),

            zorder=25,

        )


###############################################################################
# Lane Plotting
###############################################################################


def plot_lanes(
    ax,
    scene: RawScene,
):
    """
    Draw HD lane centerlines.
    """

    for lane in scene.lanes.values():

        centerline = lane.centerline

        ax.plot(

            centerline[:, 0],

            centerline[:, 1],

            color=LANE_COLOR,

            linewidth=LANE_WIDTH,

            solid_capstyle="round",

            alpha=0.95,

            zorder=1,

        )


###############################################################################
# Single Track
###############################################################################


def plot_track(
    ax,
    track: RawTrack,
):
    """
    Draw one actor.
    """

    trajectory = track.positions

    observed = trajectory[:OBSERVED_STEPS]

    future = trajectory[OBSERVED_STEPS:]

    colour = actor_colour(track)

    width = actor_width(track)

    ###########################################################################
    # Observed
    ###########################################################################

    ax.plot(

        observed[:, 0],

        observed[:, 1],

        color=colour,

        linewidth=width,

        solid_capstyle="round",

        zorder=15,

    )

    ###########################################################################
    # Future
    ###########################################################################

    if len(future) > 1:

        future_colour = (

            TARGET_FUTURE

            if track.is_target

            else colour

        )

        ax.plot(

            future[:, 0],

            future[:, 1],

            "--",

            color=future_colour,

            linewidth=max(

                width - 0.4,

                1.0,

            ),

            dashes=(5, 3),

            zorder=16,

        )

    ###########################################################################
    # Direction
    ###########################################################################

    draw_direction_arrows(

        ax,

        observed,

        colour,

    )

###############################################################################
# Track Collection
###############################################################################

def plot_tracks(
    ax,
    scene: RawScene,
    *,
    show_ids: bool = False,
):
    """
    Plot every actor in the scene.
    """

    for track in scene.tracks.values():

        plot_track(

            ax,

            track,

        )

        #######################################################################
        # Optional Track ID
        #######################################################################

        if show_ids:

            position = track.positions[

                min(
                    OBSERVED_STEPS - 1,
                    len(track.positions) - 1,
                )

            ]

            ax.text(

                position[0],

                position[1],

                track.track_id,

                fontsize=7,

                color="black",

                ha="center",

                va="center",

                zorder=40,

            )


###############################################################################
# Scene Bounds
###############################################################################

def compute_scene_bounds(
    scene: RawScene,
) -> tuple[float, float, float, float]:
    """
    Compute axis limits automatically.
    """

    points = []

    ###########################################################################
    # Lanes
    ###########################################################################

    for lane in scene.lanes.values():

        points.append(

            lane.centerline,

        )

    ###########################################################################
    # Tracks
    ###########################################################################

    for track in scene.tracks.values():

        points.append(

            track.positions,

        )

    ###########################################################################
    # Safety
    ###########################################################################

    if len(points) == 0:

        return (

            -10,

            10,

            -10,

            10,

        )

    points = np.concatenate(

        points,

        axis=0,

    )

    xmin = points[:, 0].min() - SCENE_MARGIN

    xmax = points[:, 0].max() + SCENE_MARGIN

    ymin = points[:, 1].min() - SCENE_MARGIN

    ymax = points[:, 1].max() + SCENE_MARGIN

    return (

        xmin,

        xmax,

        ymin,

        ymax,

    )


###############################################################################
# Axis Formatting
###############################################################################

def style_axes(
    ax,
    scene: RawScene,
):
    """
    Apply publication-style formatting.
    """

    xmin, xmax, ymin, ymax = compute_scene_bounds(

        scene,

    )

    ax.set_xlim(

        xmin,

        xmax,

    )

    ax.set_ylim(

        ymin,

        ymax,

    )

    ax.set_aspect(

        "equal",

        adjustable="box",

    )

    ###########################################################################
    # Clean Paper Style
    ###########################################################################

    ax.grid(False)

    ax.set_xlabel("World X (m)")

    ax.set_ylabel("World Y (m)")

    ax.set_title(

        f"Raw Scene {scene.metadata.sequence_id}",

        fontsize=14,

    )

    ###########################################################################
    # Remove top/right frame
    ###########################################################################

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ###########################################################################
    # Thin remaining frame
    ###########################################################################

    ax.spines["left"].set_linewidth(0.8)

    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(

        direction="out",

        length=4,

        width=0.8,

    )


###############################################################################
# Main Plotter
###############################################################################

def plot_raw_scene(
    scene: RawScene,
    *,
    ax=None,
    show_ids: bool = False,
):
    """
    Draw an Argoverse RawScene.

    Parameters
    ----------
    scene
        Parsed RawScene.

    ax
        Existing matplotlib axis.

    show_ids
        Display track IDs.
    """

    if ax is None:

        _, ax = plt.subplots(

            figsize=(9, 9),

        )

    ###########################################################################
    # Draw HD Map
    ###########################################################################

    plot_lanes(

        ax,

        scene,

    )

    ###########################################################################
    # Draw Actors
    ###########################################################################

    plot_tracks(

        ax,

        scene,

        show_ids=show_ids,

    )

    ###########################################################################
    # Formatting
    ###########################################################################

    style_axes(

        ax,

        scene,

    )

    return ax

