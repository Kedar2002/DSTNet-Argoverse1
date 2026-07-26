"""
scripts.infer_mini

Mini inference for DSTNet.

Runs inference on a single validation scene using the mini checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
# Import Production Pipeline
###############################################################################

from scripts.infer import (

    DATASET_ROOT,

    print_header,

    run_inference_pipeline,

)

###############################################################################
# Mini Configuration
###############################################################################

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

    / "inference"

)

###############################################################################
# Scene
###############################################################################

DEFAULT_SCENE = (

    DATASET_ROOT

    / "37853.csv"

)

###############################################################################
# Main
###############################################################################

def main():

    print_header(

        "DSTNet Mini Inference",

    )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(

            f"Checkpoint not found:\n{CHECKPOINT}"

        )

    run_inference_pipeline(

        scene_path=DEFAULT_SCENE,

        checkpoint=CHECKPOINT,

        result_root=RESULT_ROOT,

    )

    print()

    print("=" * 80)

    print("✓ MINI INFERENCE COMPLETE")

    print("=" * 80)


###############################################################################
# Entry
###############################################################################

if __name__ == "__main__":

    main()

    
