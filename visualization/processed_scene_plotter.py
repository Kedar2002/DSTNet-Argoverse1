"""
visualization.processed_scene_plotter

Publication-quality visualization of a processed SceneData.

Displays

- Local-coordinate lane centerlines
- Target agent
- Autonomous Vehicle
- Other agents
- Observed trajectories
- Future trajectories
- Local origin
- Heading axis

Coordinates are shown in the normalized target-agent frame.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from datasets.scene_data import SceneData

###############################################################################
# Plot Style
###############################################################################

LANE_COLOR = "#D6D6D6"

TARGET_OBSERVED = "#0B7A75"      # teal
TARGET_FUTURE = "#C62828"        # red

AV_COLOR = "#F4A641"             # orange

OTHER_COLOR = "#707070"

ORIGIN_COLOR = "#202020"

###############################################################################
# Drawing Parameters
###############################################################################

TARGET_WIDTH = 2.8

AV_WIDTH = 2.2

OTHER_WIDTH = 1.2

LANE_WIDTH = 0.8

ARROW_EVERY = 4

SCENE_MARGIN = 5.0

###############################################################################
# Direction Arrows
###############################################################################


def draw_direction_arrows(
    ax,
    trajectory: np.ndarray,
    colour: str,
    *,
    every: int = ARROW_EVERY,
):
    """
    Draw travel direction arrows.
    """

    if len(trajectory) < 2:
        return

    for i in range(

        every,

        len(trajectory),

        every,

    ):

        ax.annotate(

            "",

            xy=trajectory[i],

            xytext=trajectory[i - 1],

            arrowprops=dict(

                arrowstyle="-|>",

                color=colour,

                lw=1.0,

                mutation_scale=11,

            ),

            zorder=20,

        )

###############################################################################
# Lane Plotting
###############################################################################


def plot_lanes(
    ax,
    scene: SceneData,
):
    """
    Plot normalized lane centerlines.
    """

    for lane in scene.lanes:

        centerline = lane["centerline"]

        ax.plot(

            centerline[:, 0],

            centerline[:, 1],

            color=LANE_COLOR,

            linewidth=LANE_WIDTH,

            solid_capstyle="round",

            zorder=1,

        )

###############################################################################
# Single Agent
###############################################################################


def plot_agent(
    ax,
    agent,
):

    observed = agent["observed"]

    future = agent["future"]

    ###########################################################################
    # Target
    ###########################################################################

    if agent["category"].upper() == "AGENT":

        observed_colour = TARGET_OBSERVED

        future_colour = TARGET_FUTURE

        width = TARGET_WIDTH

    ###########################################################################
    # AV
    ###########################################################################

    elif agent["object_type"].upper() == "AV":

        observed_colour = AV_COLOR

        future_colour = AV_COLOR

        width = AV_WIDTH

    ###########################################################################
    # Others
    ###########################################################################

    else:

        observed_colour = OTHER_COLOR

        future_colour = OTHER_COLOR

        width = OTHER_WIDTH

    ###########################################################################
    # Observed
    ###########################################################################

    ax.plot(

        observed[:, 0],

        observed[:, 1],

        color=observed_colour,

        linewidth=width,

        solid_capstyle="round",

        zorder=15,

    )

    ###########################################################################
    # Future
    ###########################################################################

    if len(future):

        ax.plot(

            future[:, 0],

            future[:, 1],

            "--",

            color=future_colour,

            linewidth=max(width - 0.4, 1.0),

            dashes=(5, 3),

            zorder=16,

        )

    ###########################################################################
    # Direction
    ###########################################################################

    draw_direction_arrows(

        ax,

        observed,

        observed_colour,

    )

###############################################################################
# Agent Collection
###############################################################################

def plot_agents(
    ax,
    scene: SceneData,
    *,
    show_ids: bool = False,
):
    """
    Plot every processed agent.
    """

    for agent in scene.agents:

        plot_agent(

            ax,

            agent,

        )

        #######################################################################
        # Optional Track ID
        #######################################################################

        if show_ids:

            position = agent["observed"][-1]

            ax.text(

                position[0],

                position[1],

                str(agent["track_id"]),

                fontsize=7,

                ha="center",

                va="center",

                color="black",

                zorder=30,

            )


###############################################################################
# Local Coordinate Frame
###############################################################################

def plot_reference_frame(
    ax,
):
    """
    Draw local origin and heading direction.

    In the processed frame:
        Origin  -> (0,0)
        Heading -> +X
    """

    ###########################################################################
    # Origin
    ###########################################################################

    ax.plot(

        0.0,

        0.0,

        marker="+",

        color=ORIGIN_COLOR,

        markersize=12,

        markeredgewidth=2,

        zorder=40,

    )

    ###########################################################################
    # Heading Axis
    ###########################################################################

    ax.annotate(

        "",

        xy=(6.0, 0.0),

        xytext=(0.0, 0.0),

        arrowprops=dict(

            arrowstyle="-|>",

            color=ORIGIN_COLOR,

            lw=1.5,

            mutation_scale=12,

        ),

        zorder=40,

    )

    ax.text(

        6.5,

        0.0,

        "Heading",

        fontsize=9,

        va="center",

    )


###############################################################################
# Scene Bounds
###############################################################################

def compute_scene_bounds(
    scene: SceneData,
):
    """
    Compute automatic axis limits.
    """

    points = []

    ###########################################################################
    # Lanes
    ###########################################################################

    for lane in scene.lanes:

        points.append(

            lane["centerline"],

        )

    ###########################################################################
    # Agents
    ###########################################################################

    for agent in scene.agents:

        points.append(

            agent["observed"],

        )

        if len(agent["future"]):

            points.append(

                agent["future"],

            )

    ###########################################################################
    # Fallback
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
# Axis Styling
###############################################################################

def style_axes(
    ax,
    scene: SceneData,
):
    """
    Publication-style formatting.
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
    # Clean Style
    ###########################################################################

    ax.grid(False)

    ax.set_xlabel("Local X (m)")

    ax.set_ylabel("Local Y (m)")

    ax.set_title(

        f"Processed Scene {scene.sequence_id}",

        fontsize=14,

    )

    ###########################################################################
    # Remove top/right border
    ###########################################################################

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

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

def plot_processed_scene(
    scene: SceneData,
    *,
    ax=None,
    show_ids: bool = False,
):
    """
    Plot a processed SceneData object.

    Parameters
    ----------
    scene
        Processed scene.

    ax
        Existing matplotlib axis.

    show_ids
        Draw track IDs.
    """

    if ax is None:

        _, ax = plt.subplots(

            figsize=(9, 9),

        )

    ###########################################################################
    # Draw HD map
    ###########################################################################

    plot_lanes(

        ax,

        scene,

    )

    ###########################################################################
    # Draw actors
    ###########################################################################

    plot_agents(

        ax,

        scene,

        show_ids=show_ids,

    )

    ###########################################################################
    # Local frame
    ###########################################################################

    plot_reference_frame(

        ax,

    )

    ###########################################################################
    # Style
    ###########################################################################

    style_axes(

        ax,

        scene,

    )

    return ax



