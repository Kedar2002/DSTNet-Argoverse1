"""
scripts.evaluate_mini

Mini evaluation script for DSTNet.

Evaluates a trained checkpoint using a small validation subset.

This script reuses the production evaluation pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import Subset

###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(

        0,

        str(PROJECT_ROOT),

    )

###############################################################################
# Import Production Evaluation Pipeline
###############################################################################

from scripts.evaluate import (

    VAL_ROOT,

    build_dataset,

    print_header,

    print_section,

    run_evaluation,

)

###############################################################################
# Mini Configuration
###############################################################################

VAL_SCENES = 64

CHECKPOINT = (

    PROJECT_ROOT

    / "checkpoints"

    / "mini"

    / "best.pth"

)

RESULT_ROOT = (

    PROJECT_ROOT

    / "results"

    / "mini"

)

###############################################################################
# Validation Subset
###############################################################################

def build_validation_subset():

    print_section(

        "Building Mini Validation Dataset",

    )

    dataset = build_dataset(

        VAL_ROOT

    )

    subset_size = min(

        VAL_SCENES,

        len(dataset),

    )

    subset = Subset(

        dataset,

        range(subset_size),

    )

    print(

        f"Original Validation Scenes : {len(dataset):,}"

    )

    print(

        f"Mini Validation Scenes     : {subset_size}"

    )

    return subset

###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet Mini Evaluation"
    )

    ###########################################################################
    # Build Dataset
    ###########################################################################

    validation_dataset = build_validation_subset()

    ###########################################################################
    # Results Directory
    ###########################################################################

    RESULT_ROOT.mkdir(

        parents=True,

        exist_ok=True,

    )

    ###########################################################################
    # Checkpoint
    ###########################################################################

    if not CHECKPOINT.exists():

        raise FileNotFoundError(

            f"Checkpoint not found:\n{CHECKPOINT}"

        )

    print_section(

        "Checkpoint"

    )

    print(

        f"Using : {CHECKPOINT.name}"

    )

    ###########################################################################
    # Run Production Evaluation
    ###########################################################################

    run_evaluation(

        dataset=validation_dataset,

        checkpoint=CHECKPOINT,

        result_root=RESULT_ROOT,

    )


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()
