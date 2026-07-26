"""
visualization.raw_scene_plotter

Visualize a RawScene directly from the Argoverse dataset.

This module plots

- HD lane centerlines
- Agent trajectories
- Target agent
- Autonomous vehicle
- Object categories
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from datasets.raw_scene import RawScene


###############################################################################
# Colours
###############################################################################

COLORS = {

    "VEHICLE": "black",

    "PEDESTRIAN": "tab:green",

    "CYCLIST": "tab:orange",

    "MOTORCYCLIST": "tab:brown",

    "BUS": "purple",

    "AV": "tab:blue",

    "UNKNOWN": "gray",

}


###############################################################################
# Lane Plotting
###############################################################################

def plot_lanes(
    ax,
    scene: RawScene,
):

    for lane in scene.lanes.values():

        centerline = lane.centerline

        ax.plot(

            centerline[:, 0],

            centerline[:, 1],

            color="lightgray",

            linewidth=1.2,

            zorder=1,

        )


###############################################################################
# Agent Plotting
###############################################################################

def plot_tracks(
    ax,
    scene: RawScene,
    *,
    show_ids: bool = False,
):

    for track in scene.tracks.values():

        trajectory = track.positions

        #######################################################################
        # Target Agent
        #######################################################################

        if track.is_target:

            colour = "red"

            width = 3.5

            alpha = 1.0

            zorder = 10

        #######################################################################
        # Autonomous Vehicle
        #######################################################################

        elif track.is_av:

            colour = "tab:blue"

            width = 3.0

            alpha = 1.0

            zorder = 9

        #######################################################################
        # Other Agents
        #######################################################################

        else:

            colour = COLORS.get(

                track.object_type.upper(),

                COLORS["UNKNOWN"],

            )

            width = 1.4

            alpha = 0.7

            zorder = 5

        ###############################################################################
        # Observed Trajectory
        ###############################################################################

        observed = trajectory[:20]

        ax.plot(

            observed[:, 0],

            observed[:, 1],

            color=colour,

            linewidth=width,

            alpha=alpha,

            zorder=zorder,

        )

        ###############################################################################
        # Future Trajectory
        ###############################################################################

        if len(trajectory) > 20:

            future = trajectory[20:]

            ax.plot(

                future[:, 0],

                future[:, 1],

                "--",

                color=colour,

                linewidth=max(1.0, width - 0.5),

                alpha=alpha,

                zorder=zorder,

            )

        #######################################################################
        # Last Position
        #######################################################################

        ax.scatter(

            trajectory[-1, 0],

            trajectory[-1, 1],

            color=colour,

            s=18,

            zorder=zorder + 1,

        )

        ax.scatter(

            trajectory[0, 0],

            trajectory[0, 1],

            color=colour,

            marker="s",

            s=20,

            zorder=zorder + 1,

        )

        #######################################################################
        # Track ID
        #######################################################################

        if show_ids:

            ax.text(

                trajectory[-1, 0],

                trajectory[-1, 1],

                str(track.track_id),

                fontsize=7,

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
    Plot the original Argoverse scene in world coordinates.
    """

    if ax is None:

        _, ax = plt.subplots(

            figsize=(10, 10),

        )

    ###########################################################################
    # Lanes
    ###########################################################################

    plot_lanes(

        ax,

        scene,

    )

    ###########################################################################
    # Tracks
    ###########################################################################

    plot_tracks(

        ax,

        scene,

        show_ids=show_ids,

    )

    ###########################################################################
    # Formatting
    ###########################################################################

    ax.set_title(

        f"Raw Scene : {scene.metadata.sequence_id}"

    )

    ax.set_xlabel(

        "World X (m)"

    )

    ax.set_ylabel(

        "World Y (m)"

    )

    ax.set_aspect(

        "equal",

        adjustable="box",

    )

    ax.grid(

        alpha=0.25,

    )

    legend = [

        Line2D([0], [0], color="red", lw=3, label="Target"),

        Line2D([0], [0], color="tab:blue", lw=3, label="AV"),

        Line2D([0], [0], color="black", lw=2, label="Other"),

    ]

    ax.legend(

        handles=legend,

        loc="upper right",

    )

    return ax
