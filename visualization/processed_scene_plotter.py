"""
visualization.processed_scene_plotter

Visualize a processed SceneData in the local reference frame.

Displays

- Normalized lane centerlines
- Target agent
- AV
- Other agents
- Local origin
- Heading direction
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from datasets.scene_data import SceneData

###############################################################################
# Colours
###############################################################################

LANE_COLOR = "lightgray"

TARGET_OBS = "tab:blue"

TARGET_FUTURE = "tab:green"

AV_COLOR = "tab:orange"

OTHER_COLOR = "gray"

ORIGIN_COLOR = "red"

###############################################################################
# Lanes
###############################################################################

def plot_lanes(
    ax,
    scene: SceneData,
):

    for lane in scene.lanes:

        centerline = lane["centerline"]

        ax.plot(

            centerline[:, 0],

            centerline[:, 1],

            color=LANE_COLOR,

            linewidth=1.2,

            zorder=1,

        )

###############################################################################
# Agents
###############################################################################

def plot_agents(
    ax,
    scene: SceneData,
    *,
    show_ids: bool = False,
):

    for agent in scene.agents:

        observed = agent["observed"]

        future = agent["future"]

        #######################################################################
        # Target Agent
        #######################################################################

        if agent["category"].upper() == "AGENT":

            colour_obs = TARGET_OBS

            colour_future = TARGET_FUTURE

            width = 3.0

        #######################################################################
        # AV
        #######################################################################

        elif agent["object_type"].upper() == "AV":

            colour_obs = AV_COLOR

            colour_future = AV_COLOR

            width = 2.5

        #######################################################################
        # Others
        #######################################################################

        else:

            colour_obs = OTHER_COLOR

            colour_future = OTHER_COLOR

            width = 1.2

        ###############################################################
        # Observed
        ###############################################################

        ax.plot(

            observed[:, 0],

            observed[:, 1],

            color=colour_obs,

            linewidth=width,

            zorder=5,

        )

        ###############################################################
        # Future
        ###############################################################

        if len(future) > 0:

            ax.plot(

                future[:, 0],

                future[:, 1],

                "--",

                color=colour_future,

                linewidth=width,

                zorder=6,

            )

        ###############################################################
        # End point
        ###############################################################

        ax.scatter(

            observed[-1, 0],

            observed[-1, 1],

            color=colour_obs,

            s=20,

            zorder=7,

        )

        ###############################################################
        # IDs
        ###############################################################

        if show_ids:

            ax.text(

                observed[-1, 0],

                observed[-1, 1],

                str(agent["track_id"]),

                fontsize=7,

            )

###############################################################################
# Origin
###############################################################################

def plot_origin(
    ax,
):

    ax.scatter(

        0.0,

        0.0,

        marker="*",

        s=120,

        color=ORIGIN_COLOR,

        zorder=20,

        label="Origin",

    )

    ax.arrow(

        0.0,

        0.0,

        5.0,

        0.0,

        width=0.1,

        color=ORIGIN_COLOR,

        zorder=20,

    )

###############################################################################
# Main
###############################################################################

def plot_processed_scene(
    scene: SceneData,
    *,
    ax=None,
    show_ids: bool = False,
):

    if ax is None:

        _, ax = plt.subplots(

            figsize=(10, 10),

        )

    plot_lanes(

        ax,

        scene,

    )

    plot_agents(

        ax,

        scene,

        show_ids=show_ids,

    )

    plot_origin(

        ax,

    )

    ax.set_title(

        f"Processed Scene : {scene.sequence_id}"

    )

    ax.set_xlabel(

        "Local X (m)"

    )

    ax.set_ylabel(

        "Local Y (m)"

    )

    ax.set_aspect(

        "equal",

        adjustable="box",

    )

    ax.grid(

        alpha=0.25,

    )

    return ax
