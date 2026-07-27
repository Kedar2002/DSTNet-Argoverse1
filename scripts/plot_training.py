"""
scripts.plot_training

Plot DSTNet training curves from training_log.csv.

Generates

    • Training Loss
    • minADE
    • minFDE
    • MissRate
    • Learning Rate

Usage
-----

Mini training

python -m scripts.plot_training \
    --log logs/mini/training_log.csv

Production training

python -m scripts.plot_training \
    --log logs/training_log.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
# Default Figure Style
###############################################################################

plt.style.use("default")

FIGSIZE = (8, 5)

DPI = 300

LINEWIDTH = 2.2

MARKERSIZE = 5

GRID_ALPHA = 0.30

###############################################################################
# Argument Parser
###############################################################################

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="DSTNet Training Curve Plotter"
    )

    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to training_log.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory for saving plots",
    )

    return parser

###############################################################################
# Load CSV
###############################################################################

def load_log(csv_path: Path) -> pd.DataFrame:

    if not csv_path.exists():

        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    print()

    print("=" * 80)

    print("Training Log Loaded")

    print("=" * 80)

    print(df.head())

    return df

###############################################################################
# Column Detection
###############################################################################

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    """
    Find the first matching column.
    """

    columns = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in columns:

            return columns[candidate.lower()]

    raise KeyError(

        f"None of the columns found:\n{candidates}"

    )


###############################################################################
# Plot Utility
###############################################################################

def plot_metric(
    *,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
) -> None:
    """
    Plot one metric.
    """

    plt.figure(
        figsize=FIGSIZE,
    )

    plt.plot(

        df[x_column],

        df[y_column],

        marker="o",

        linewidth=LINEWIDTH,

        markersize=MARKERSIZE,

    )

    ###########################################################################
    # Formatting
    ###########################################################################

    plt.title(
        title,
        fontsize=14,
        weight="bold",
    )

    plt.xlabel(
        "Epoch",
        fontsize=12,
    )

    plt.ylabel(
        ylabel,
        fontsize=12,
    )

    plt.grid(
        True,
        alpha=GRID_ALPHA,
    )

    plt.tight_layout()

    ###########################################################################
    # Save
    ###########################################################################

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    save_path = (

        output_dir

        / filename

    )

    plt.savefig(

        save_path,

        dpi=DPI,

    )

    plt.close()

    print(

        f"Saved : {save_path}"

    )


###############################################################################
# Column Mapping
###############################################################################

def get_columns(
    df: pd.DataFrame,
) -> dict[str, str]:
    """
    Automatically detect training log columns.
    """

    return {

        "epoch": find_column(
            df,
            [
                "epoch",
            ],
        ),

        "loss": find_column(
            df,
            [
                "train_loss",
                "loss",
            ],
        ),

        "ade": find_column(
            df,
            [
                "minADE",
                "minade",
            ],
        ),

        "fde": find_column(
            df,
            [
                "minFDE",
                "minfde",
            ],
        ),

        "miss": find_column(
            df,
            [
                "MissRate",
                "missrate",
            ],
        ),

        "lr": find_column(
            df,
            [
                "learning_rate",
                "lr",
            ],
        ),

    }

###############################################################################
# Plot All Metrics
###############################################################################

def generate_plots(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate all training curves.
    """

    columns = get_columns(df)

    ###########################################################################
    # Training Loss
    ###########################################################################

    plot_metric(

        df=df,

        x_column=columns["epoch"],

        y_column=columns["loss"],

        title="Training Loss",

        ylabel="Loss",

        output_dir=output_dir,

        filename="training_loss.png",

    )

    ###########################################################################
    # minADE
    ###########################################################################

    plot_metric(

        df=df,

        x_column=columns["epoch"],

        y_column=columns["ade"],

        title="Validation minADE",

        ylabel="minADE (m)",

        output_dir=output_dir,

        filename="minADE.png",

    )

    ###########################################################################
    # minFDE
    ###########################################################################

    plot_metric(

        df=df,

        x_column=columns["epoch"],

        y_column=columns["fde"],

        title="Validation minFDE",

        ylabel="minFDE (m)",

        output_dir=output_dir,

        filename="minFDE.png",

    )

    ###########################################################################
    # Miss Rate
    ###########################################################################

    plot_metric(

        df=df,

        x_column=columns["epoch"],

        y_column=columns["miss"],

        title="Validation Miss Rate",

        ylabel="Miss Rate",

        output_dir=output_dir,

        filename="miss_rate.png",

    )

    ###########################################################################
    # Learning Rate
    ###########################################################################

    plot_metric(

        df=df,

        x_column=columns["epoch"],

        y_column=columns["lr"],

        title="Learning Rate Schedule",

        ylabel="Learning Rate",

        output_dir=output_dir,

        filename="learning_rate.png",

    )


###############################################################################
# Main
###############################################################################

def main() -> None:

    parser = build_argument_parser()

    args = parser.parse_args()

    csv_path = Path(args.log)

    if args.output is None:

        output_dir = csv_path.parent / "plots"

    else:

        output_dir = Path(args.output)

    df = load_log(

        csv_path,

    )

    print()

    print("=" * 80)

    print("Generating Training Curves")

    print("=" * 80)

    generate_plots(

        df,

        output_dir,

    )

    print()

    print("=" * 80)

    print("Training Curves Generated Successfully")

    print("=" * 80)

    print()

    print(f"Output Directory : {output_dir}")


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()
