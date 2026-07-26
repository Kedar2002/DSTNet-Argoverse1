"""
visualization.raw_scene_plotter

Publication-quality visualization of an Argoverse 1 RawScene.

Features
--------
- Local HD map visualization
- Nearby actor filtering
- Target agent highlighting
- Autonomous vehicle highlighting
- Direction arrows
- Automatic scene cropping
- Paper-style aesthetics
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from datasets.raw_scene import (
    RawLane,
    RawScene,
    RawTrack,
)

###############################################################################
# Appearance
###############################################################################

LANE_COLOR = "#E6E6E6"

TARGET_OBSERVED_COLOR = "#00897B"

TARGET_FUTURE_COLOR = "#D32F2F"

AV_COLOR = "#F39C12"

VEHICLE_COLOR = "#7A7A7A"

PEDESTRIAN_COLOR = "#4CAF50"

CYCLIST_COLOR = "#8E44AD"

BUS_COLOR = "#1565C0"

UNKNOWN_COLOR = "#909090"

###############################################################################
# Drawing Parameters
###############################################################################

OBSERVED_STEPS = 20

TARGET_WIDTH = 2.2

AV_WIDTH = 2.0

OTHER_WIDTH = 1.1

LANE_WIDTH = 0.65

LANE_ALPHA = 0.95

OTHER_ALPHA = 0.65

TARGET_ALPHA = 1.0

ARROW_INTERVAL = 4

ARROW_SCALE = 8

LOCAL_RADIUS = 35.0

SCENE_MARGIN = 4.0

###############################################################################
# Utilities
###############################################################################


def target_position(
    scene: RawScene,
) -> np.ndarray:
    """
    Last observed target position.
    """

    target = scene.target_track

    index = min(
        OBSERVED_STEPS - 1,
        len(target.positions) - 1,
    )

    return target.positions[index]


###############################################################################
# Nearby Filtering
###############################################################################


def nearby_tracks(
    scene: RawScene,
    radius: float = LOCAL_RADIUS,
) -> list[RawTrack]:
    """
    Return only actors close to the target.

    Target and AV are always included.
    """

    centre = target_position(scene)

    selected: list[RawTrack] = []

    for track in scene.tracks.values():

        if track.is_target:

            selected.append(track)

            continue

        if track.is_av:

            selected.append(track)

            continue

        index = min(

            OBSERVED_STEPS - 1,

            len(track.positions) - 1,

        )

        distance = np.linalg.norm(

            track.positions[index] - centre

        )

        if distance <= radius:

            selected.append(track)

    return selected


def nearby_lanes(
    scene: RawScene,
    radius: float = LOCAL_RADIUS,
) -> list[RawLane]:
    """
    Return lanes surrounding the target.
    """

    centre = target_position(scene)

    selected: list[RawLane] = []

    for lane in scene.lanes.values():

        minimum_distance = np.min(

            np.linalg.norm(

                lane.centerline - centre,

                axis=1,

            )

        )

        if minimum_distance <= radius:

            selected.append(lane)

    return selected


###############################################################################
# Actor Appearance
###############################################################################


def actor_colour(
    track: RawTrack,
) -> tuple[str, str]:

    if track.is_target:

        return (

            TARGET_OBSERVED_COLOR,

            TARGET_FUTURE_COLOR,

        )

    if track.is_av:

        return (

            AV_COLOR,

            AV_COLOR,

        )

    colour = {

        "VEHICLE": VEHICLE_COLOR,

        "PEDESTRIAN": PEDESTRIAN_COLOR,

        "CYCLIST": CYCLIST_COLOR,

        "BUS": BUS_COLOR,

    }.get(

        track.object_type.upper(),

        UNKNOWN_COLOR,

    )

    return (

        colour,

        colour,

    )


def actor_width(
    track: RawTrack,
) -> float:

    if track.is_target:

        return TARGET_WIDTH

    if track.is_av:

        return AV_WIDTH

    return OTHER_WIDTH


def actor_alpha(
    track: RawTrack,
) -> float:

    if track.is_target:

        return TARGET_ALPHA

    if track.is_av:

        return TARGET_ALPHA

    return OTHER_ALPHA


###############################################################################
# Direction Arrows
###############################################################################


def draw_direction_arrows(
    ax,
    trajectory: np.ndarray,
    colour: str,
):
    """
    Draw travel direction arrows.
    """

    if len(trajectory) < 2:

        return

    for i in range(

        ARROW_INTERVAL,

        len(trajectory),

        ARROW_INTERVAL,

    ):

        ax.annotate(

            "",

            xy=trajectory[i],

            xytext=trajectory[i - 1],

            arrowprops=dict(

                arrowstyle="-|>",

                color=colour,

                lw=0.8,

                mutation_scale=ARROW_SCALE,

                shrinkA=0,

                shrinkB=0,

            ),

            zorder=50,

        )

###############################################################################
# Lane Rendering
###############################################################################

def plot_lanes(
    ax,
    scene: RawScene,
):
    """
    Draw only nearby HD lanes.
    """

    lanes = nearby_lanes(scene)

    for lane in lanes:

        centerline = lane.centerline

        #######################################################################
        # Lane polyline
        #######################################################################

        ax.plot(

            centerline[:, 0],

            centerline[:, 1],

            color=LANE_COLOR,

            linewidth=LANE_WIDTH,

            alpha=LANE_ALPHA,

            solid_capstyle="round",

            zorder=1,

        )

        #######################################################################
        # Lane direction arrows
        #######################################################################

        if len(centerline) < 3:

            continue

        step = max(

            4,

            len(centerline) // 6,

        )

        for i in range(

            step,

            len(centerline),

            step,

        ):

            ax.annotate(

                "",

                xy=centerline[i],

                xytext=centerline[i - 1],

                arrowprops=dict(

                    arrowstyle="-",

                    lw=0.7,

                    color=LANE_COLOR,

                ),

                zorder=2,

            )


###############################################################################
# Actor Rendering
###############################################################################

def plot_track(
    ax,
    track: RawTrack,
):
    """
    Draw a single actor.
    """

    trajectory = track.positions

    observed = trajectory[:OBSERVED_STEPS]

    future = trajectory[OBSERVED_STEPS:]

    observed_colour, future_colour = actor_colour(track)

    width = actor_width(track)

    alpha = actor_alpha(track)

    ###########################################################################
    # Observed trajectory
    ###########################################################################

    ax.plot(

        observed[:, 0],

        observed[:, 1],

        color=observed_colour,

        linewidth=width,

        alpha=alpha,

        solid_capstyle="round",

        zorder=20,

    )

    ###########################################################################
    # Future trajectory
    ###########################################################################

    if len(future) > 1:

        ax.plot(

            future[:, 0],

            future[:, 1],

            "--",

            color=future_colour,

            linewidth=max(

                width - 0.3,

                1.0,

            ),

            dashes=(6, 3),

            alpha=alpha,

            solid_capstyle="round",

            zorder=21,

        )

    ###########################################################################
    # Direction arrows
    ###########################################################################

    draw_direction_arrows(

        ax,

        observed,

        observed_colour,

    )


###############################################################################
# All Actors
###############################################################################

def plot_tracks(
    ax,
    scene: RawScene,
    *,
    show_ids: bool = False,
):
    """
    Plot nearby actors only.
    """

    tracks = nearby_tracks(scene)

    ###########################################################################
    # Draw others first
    ###########################################################################

    for track in tracks:

        if track.is_target:

            continue

        plot_track(

            ax,

            track,

        )

    ###########################################################################
    # Draw target last
    ###########################################################################

    plot_track(

        ax,

        scene.target_track,

    )

    ###########################################################################
    # Optional IDs
    ###########################################################################

    if not show_ids:

        return

    for track in tracks:

        index = min(

            OBSERVED_STEPS - 1,

            len(track.positions) - 1,

        )

        point = track.positions[index]

        ax.text(

            point[0],

            point[1],

            track.track_id,

            fontsize=7,

            color="black",

            ha="center",

            va="center",

            zorder=100,

        )

###############################################################################
# Scene Bounds
###############################################################################

def compute_scene_bounds(
    scene: RawScene,
):
    """
    Compute plot limits around the target agent.

    The visualization is intentionally local instead of showing
    the complete Argoverse map.
    """

    centre = target_position(scene)

    xmin = centre[0] - LOCAL_RADIUS - SCENE_MARGIN
    xmax = centre[0] + LOCAL_RADIUS + SCENE_MARGIN

    ymin = centre[1] - LOCAL_RADIUS - SCENE_MARGIN
    ymax = centre[1] + LOCAL_RADIUS + SCENE_MARGIN

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
    Apply paper-style formatting.
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

    ###########################################################################
    # Equal aspect
    ###########################################################################

    ax.set_aspect(

        "equal",

        adjustable="box",

    )

    ###########################################################################
    # Remove grid
    ###########################################################################

    ax.grid(False)

    ###########################################################################
    # Remove tick labels
    ###########################################################################

    ax.set_xticks([])

    ax.set_yticks([])

    ###########################################################################
    # Remove axis labels
    ###########################################################################

    ax.set_xlabel("")

    ax.set_ylabel("")

    ###########################################################################
    # Remove title
    ###########################################################################

    ax.set_title("")

    ###########################################################################
    # Hide all borders
    ###########################################################################

    for spine in ax.spines.values():

        spine.set_visible(False)


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
        Parsed scene.

    ax
        Existing matplotlib axis.

    show_ids
        Display track IDs.
    """

    ###########################################################################
    # Create axis
    ###########################################################################

    if ax is None:

        _, ax = plt.subplots(

            figsize=(8, 8),

        )

    ###########################################################################
    # HD Map
    ###########################################################################

    plot_lanes(

        ax,

        scene,

    )

    ###########################################################################
    # Actors
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


###############################################################################
# Public API
###############################################################################

__all__ = [

    "plot_raw_scene",

]
