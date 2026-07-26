"""
visualization.processed_scene_plotter

Publication-quality visualization of a processed SceneData.

Features
--------
- Local-coordinate HD map
- Target agent highlighting
- Nearby agents
- Local origin
- Heading axis
- Paper-style appearance
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from datasets.scene_data import SceneData

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

TARGET_WIDTH = 2.2

AV_WIDTH = 2.0

OTHER_WIDTH = 1.1

LANE_WIDTH = 0.65

LANE_ALPHA = 0.95

TARGET_ALPHA = 1.0

OTHER_ALPHA = 0.65

ARROW_INTERVAL = 4

ARROW_SCALE = 8

SCENE_MARGIN = 4.0

LOCAL_RADIUS = 35.0

###############################################################################
# Actor Appearance
###############################################################################


def actor_colour(
    agent,
):

    if agent["category"].upper() == "AGENT":

        return (

            TARGET_OBSERVED_COLOR,

            TARGET_FUTURE_COLOR,

        )

    if agent["object_type"].upper() == "AV":

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

        agent["object_type"].upper(),

        UNKNOWN_COLOR,

    )

    return (

        colour,

        colour,

    )


def actor_width(
    agent,
):

    if agent["category"].upper() == "AGENT":

        return TARGET_WIDTH

    if agent["object_type"].upper() == "AV":

        return AV_WIDTH

    return OTHER_WIDTH


def actor_alpha(
    agent,
):

    if agent["category"].upper() == "AGENT":

        return TARGET_ALPHA

    if agent["object_type"].upper() == "AV":

        return TARGET_ALPHA

    return OTHER_ALPHA

###############################################################################
# Nearby Filtering
###############################################################################


def target_position(
    scene: SceneData,
):

    target = scene.target_agent

    return target["observed"][-1]


def nearby_agents(
    scene: SceneData,
):

    centre = target_position(scene)

    selected = []

    for agent in scene.agents:

        if agent["category"].upper() == "AGENT":

            selected.append(agent)

            continue

        if agent["object_type"].upper() == "AV":

            selected.append(agent)

            continue

        distance = np.linalg.norm(

            agent["observed"][-1]

            - centre

        )

        if distance <= LOCAL_RADIUS:

            selected.append(agent)

    return selected


def nearby_lanes(
    scene: SceneData,
):

    centre = target_position(scene)

    selected = []

    for lane in scene.lanes:

        distance = np.min(

            np.linalg.norm(

                lane["centerline"] - centre,

                axis=1,

            )

        )

        if distance <= LOCAL_RADIUS:

            selected.append(

                lane,

            )

    return selected

###############################################################################
# Direction Arrows
###############################################################################


def draw_direction_arrows(
    ax,
    trajectory,
    colour,
):

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

                lw=0.8,

                color=colour,

                mutation_scale=ARROW_SCALE,

                shrinkA=0,

                shrinkB=0,

            ),

            zorder=40,

        )

###############################################################################
# Lane Rendering
###############################################################################

def plot_lanes(
    ax,
    scene: SceneData,
):
    """
    Draw nearby normalized lane centerlines.
    """

    lanes = nearby_lanes(scene)

    for lane in lanes:

        centerline = lane["centerline"]

        #######################################################################
        # Centerline
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
        # Lane Direction
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
# Single Agent
###############################################################################

def plot_agent(
    ax,
    agent,
):
    """
    Draw one processed actor.
    """

    observed = agent["observed"]

    future = agent["future"]

    observed_colour, future_colour = actor_colour(

        agent,

    )

    width = actor_width(

        agent,

    )

    alpha = actor_alpha(

        agent,

    )

    ###########################################################################
    # Observed
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
    # Future
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
    # Direction Arrows
    ###########################################################################

    draw_direction_arrows(

        ax,

        observed,

        observed_colour,

    )


###############################################################################
# All Agents
###############################################################################

def plot_agents(
    ax,
    scene: SceneData,
    *,
    show_ids: bool = False,
):
    """
    Draw nearby processed actors.
    """

    agents = nearby_agents(

        scene,

    )

    ###########################################################################
    # Draw non-target agents first
    ###########################################################################

    for agent in agents:

        if agent["category"].upper() == "AGENT":

            continue

        plot_agent(

            ax,

            agent,

        )

    ###########################################################################
    # Draw target on top
    ###########################################################################

    plot_agent(

        ax,

        scene.target_agent,

    )

    ###########################################################################
    # Optional IDs
    ###########################################################################

    if not show_ids:

        return

    for agent in agents:

        point = agent["observed"][-1]

        ax.text(

            point[0],

            point[1],

            str(agent["track_id"]),

            fontsize=7,

            color="black",

            ha="center",

            va="center",

            zorder=100,

        )


###############################################################################
# Local Reference Frame
###############################################################################

def plot_reference_frame(
    ax,
):
    """
    Draw the local coordinate frame.

    The target agent is centred at (0,0)
    with heading aligned to +X.
    """

    ###########################################################################
    # Origin
    ###########################################################################

    ax.plot(

        0.0,

        0.0,

        marker="+",

        color="black",

        markersize=12,

        markeredgewidth=2,

        zorder=50,

    )

    ###########################################################################
    # Heading
    ###########################################################################

    ax.annotate(

        "",

        xy=(6.0, 0.0),

        xytext=(0.0, 0.0),

        arrowprops=dict(

            arrowstyle="-|>",

            lw=1.2,

            color="black",

            mutation_scale=10,

        ),

        zorder=50,

    )

###############################################################################
# Scene Bounds
###############################################################################

def compute_scene_bounds(
    scene: SceneData,
):
    """
    Compute local plotting bounds.

    Unlike the raw visualization, the processed scene is already
    normalized around the target. We therefore crop around the
    transformed trajectories instead of using a fixed map extent.
    """

    points = []

    ###########################################################################
    # Target
    ###########################################################################

    target = scene.target_agent

    points.append(

        target["observed"]

    )

    if len(target["future"]):

        points.append(

            target["future"]

        )

    ###########################################################################
    # Nearby agents
    ###########################################################################

    for agent in nearby_agents(scene):

        if agent is target:

            continue

        points.append(

            agent["observed"]

        )

        if len(agent["future"]):

            points.append(

                agent["future"]

            )

    ###########################################################################
    # Nearby lanes
    ###########################################################################

    for lane in nearby_lanes(scene):

        points.append(

            lane["centerline"]

        )

    ###########################################################################
    # Fallback
    ###########################################################################

    if len(points) == 0:

        return (

            -20.0,

            20.0,

            -20.0,

            20.0,

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
    Publication-style appearance.
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
    # Equal scaling
    ###########################################################################

    ax.set_aspect(

        "equal",

        adjustable="box",

    )

    ###########################################################################
    # Clean appearance
    ###########################################################################

    ax.grid(False)

    ax.set_xticks([])

    ax.set_yticks([])

    ax.set_xlabel("")

    ax.set_ylabel("")

    ax.set_title("")

    ###########################################################################
    # Hide frame
    ###########################################################################

    for spine in ax.spines.values():

        spine.set_visible(False)


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
        Display actor IDs.
    """

    ###########################################################################
    # Create axis
    ###########################################################################

    if ax is None:

        _, ax = plt.subplots(

            figsize=(8, 8),

        )

    ###########################################################################
    # HD map
    ###########################################################################

    plot_lanes(

        ax,

        scene,

    )

    ###########################################################################
    # Actors
    ###########################################################################

    plot_agents(

        ax,

        scene,

        show_ids=show_ids,

    )

    ###########################################################################
    # Local coordinate frame
    ###########################################################################

    plot_reference_frame(

        ax,

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

    "plot_processed_scene",

]
