"""
scripts.infer

Production inference script for DSTNet.

Responsibilities
----------------
- Load trained checkpoint
- Load a single Argoverse-1 scene
- Run preprocessing
- Perform forward inference
- Recover world-coordinate trajectories
- Print prediction summary
- Save inference results

Pipeline

CSV Scene
    │
    ▼
SceneParser
    │
    ▼
ScenePreprocessor
    │
    ▼
Collate
    │
    ▼
DSTNet
    │
    ▼
Best Prediction Mode
    │
    ▼
World Coordinates
    │
    ▼
Prediction Output
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

###############################################################################
# Repository Root
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################
# Dataset
###############################################################################

from datasets.map_loader import MapLoader
from datasets.scene_parser import SceneParser
from datasets.preprocess import ScenePreprocessor
from datasets.transforms import build_eval_transform
from datasets.collate import collate_fn

###############################################################################
# Model
###############################################################################

from models.dstnet import DSTNet

###############################################################################
# Utilities
###############################################################################

from engine.utils import move_to_device

###############################################################################
# Configuration
###############################################################################

MAP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
    / "hd_maps"
    / "map_files"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "argoverse1"
    / "val"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "training"
)

DEFAULT_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "best.pth"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "inference"
)

###############################################################################
# Runtime
###############################################################################

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

###############################################################################
# Default Scene
###############################################################################

DEFAULT_SCENE = (
    DATASET_ROOT
    / "37853.csv"
)

###############################################################################
# Printing
###############################################################################

def print_header(
    title: str,
) -> None:

    print()

    print("=" * 80)

    print(title)

    print("=" * 80)


def print_section(
    title: str,
) -> None:

    print()

    print(title)

    print("-" * 80)

###############################################################################
# Directory
###############################################################################

def create_directories():

    RESULT_ROOT.mkdir(

        parents=True,

        exist_ok=True,

    )

###############################################################################
# Preprocessor
###############################################################################

def build_preprocessor():

    return ScenePreprocessor(

        observation_steps=20,

        prediction_steps=30,

        lane_sample_points=20,

        agent_radius=30.0,

        lane_radius=30.0,

    )

###############################################################################
# Parser
###############################################################################

def build_parser():

    map_loader = MapLoader(

        map_root=MAP_ROOT,

    )

    parser = SceneParser(

        map_loader,

    )

    return parser

###############################################################################
# Model
###############################################################################

def build_model():

    model = DSTNet()

    model.to(
        DEVICE,
    )

    total = sum(

        parameter.numel()

        for parameter in model.parameters()

    )

    trainable = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )

    print_section(
        "Model"
    )

    print(
        f"Device               : {DEVICE}"
    )

    print(
        f"Total Parameters     : {total:,}"
    )

    print(
        f"Trainable Parameters : {trainable:,}"
    )

    return model

###############################################################################
# Scene Loader
###############################################################################

def load_scene(
    scene_path: Path,
):

    print_section(
        "Loading Scene"
    )

    if not scene_path.exists():

        raise FileNotFoundError(

            f"Scene not found:\n"

            f"{scene_path}"

        )

    parser = build_parser()

    preprocessor = build_preprocessor()

    transform = build_eval_transform()

    raw_scene = parser.parse(
        scene_path,
    )

    raw_scene = transform(
        raw_scene,
    )

    processed_scene = preprocessor.preprocess(
        raw_scene,
    )

    batch = collate_fn(
        [
            processed_scene,
        ]
    )

    batch = move_to_device(
        batch,
        DEVICE,
    )

    print(
        f"Sequence ID : {processed_scene.sequence_id}"
    )

    print(
        f"City        : {processed_scene.city}"
    )

    print(
        f"Agents      : {len(processed_scene.agents)}"
    )

    print(
        f"Lanes       : {len(processed_scene.lanes)}"
    )

    return batch, processed_scene

###############################################################################
# Checkpoint Loader
###############################################################################

def load_checkpoint(
    model: DSTNet,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
):
    """
    Load a trained DSTNet checkpoint.
    """

    print_section(
        "Loading Checkpoint"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(

            f"Checkpoint not found:\n"

            f"{checkpoint_path}"

        )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE,

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    model.eval()

    print("✓ Checkpoint Loaded")

    print()

    print(
        f"Epoch      : {checkpoint.get('epoch', 'N/A')}"
    )

    print(
        f"Train Loss : {checkpoint.get('train_loss', 0.0):.6f}"
    )

    return checkpoint


###############################################################################
# Inference
###############################################################################

@torch.no_grad()
def run_inference(
    *,
    model: DSTNet,
    batch,
):
    """
    Run DSTNet on one scene.
    """

    print_section(
        "Running Inference"
    )

    start = time.perf_counter()

    coarse_prediction, refined_prediction = model(

        agent_trajectories=batch["agent_trajectories"],

        lane_centerlines=batch["lane_centerlines"],

        positions=batch["positions"],

        headings=batch["headings"],

        graph=batch["graph"],

        agent_mask=batch.get(
            "agent_mask",
        ),

        lane_mask=batch.get(
            "lane_mask",
        ),
    )

    inference_time = (

        time.perf_counter()

        - start

    )

    return (

        coarse_prediction,

        refined_prediction,

        inference_time,

    )


###############################################################################
# Best Mode Selection
###############################################################################

def select_best_mode(
    refined_prediction,
):
    """
    Select the highest-confidence prediction mode.
    """

    scores = refined_prediction.scores

    trajectories = refined_prediction.trajectories

    ###########################################################################
    # Highest confidence mode
    ###########################################################################

    best_mode = torch.argmax(

        scores,

        dim=-1,

    )

    B, N = best_mode.shape

    prediction_steps = trajectories.shape[-2]

    best_trajectories = torch.zeros(

        (

            B,

            N,

            prediction_steps,

            2,

        ),

        dtype=trajectories.dtype,

        device=trajectories.device,

    )

    best_scores = torch.zeros(

        (

            B,

            N,

        ),

        dtype=scores.dtype,

        device=scores.device,

    )

    for b in range(B):

        for n in range(N):

            mode = int(

                best_mode[b, n]

            )

            best_trajectories[

                b,

                n,

            ] = trajectories[

                b,

                n,

                mode,

            ]

            best_scores[

                b,

                n,

            ] = scores[

                b,

                n,

                mode,

            ]

    return {

        "mode_index": best_mode,

        "scores": best_scores,

        "trajectories": best_trajectories,

    }


###############################################################################
# Inference Statistics
###############################################################################

def print_inference_summary(
    *,
    inference_time: float,
    refined_prediction,
):
    """
    Print inference statistics.
    """

    print_section(
        "Inference Summary"
    )

    print(
        f"Inference Time : "
        f"{inference_time * 1000:.2f} ms"
    )

    print()

    print(
        f"Trajectory Tensor : "
        f"{tuple(refined_prediction.trajectories.shape)}"
    )

    print(
        f"Score Tensor      : "
        f"{tuple(refined_prediction.scores.shape)}"
    )

    print()

    print(
        f"Prediction Modes  : "
        f"{refined_prediction.trajectories.shape[2]}"
    )

    print(
        f"Prediction Steps  : "
        f"{refined_prediction.trajectories.shape[3]}"
    )

###############################################################################
# World Coordinate Recovery
###############################################################################

def recover_world_coordinates(
    *,
    prediction: torch.Tensor,
    processed_scene,
):
    """
    Convert local coordinates back to world coordinates.

    Parameters
    ----------
    prediction
        (N,T,2)

    processed_scene
        ProcessedScene object.

    Returns
    -------
    torch.Tensor
        (N,T,2)
    """

    origin = torch.as_tensor(

        processed_scene.origin,

        dtype=prediction.dtype,

        device=prediction.device,

    )

    heading = float(
        processed_scene.heading
    )

    cosine = torch.cos(
        torch.tensor(
            heading,
            device=prediction.device,
            dtype=prediction.dtype,
        )
    )

    sine = torch.sin(
        torch.tensor(
            heading,
            device=prediction.device,
            dtype=prediction.dtype,
        )
    )

    rotation = torch.tensor(

        [

            [cosine, -sine],

            [sine, cosine],

        ],

        dtype=prediction.dtype,

        device=prediction.device,

    )

    world = torch.matmul(

        prediction,

        rotation.T,

    )

    world += origin

    return world


###############################################################################
# Prediction Summary
###############################################################################

def print_prediction_summary(
    *,
    prediction,
):
    """
    Print prediction summary.
    """

    print_section(
        "Prediction Summary"
    )

    print(

        f"Agents           : "

        f"{prediction['trajectories'].shape[1]}"

    )

    print(

        f"Prediction Steps : "

        f"{prediction['trajectories'].shape[2]}"

    )

    print()

    print(

        "First Agent "

        "First 5 Future Points"

    )

    print("-" * 40)

    first = prediction[
        "trajectories"
    ][0, 0]

    for point in first[:5]:

        print(

            f"({point[0]:8.3f}, "

            f"{point[1]:8.3f})"

        )


###############################################################################
# Save JSON
###############################################################################

def save_prediction_json(
    *,
    prediction,
    processed_scene,
):
    """
    Save prediction JSON.
    """

    output = {

        "sequence_id": processed_scene.sequence_id,

        "city": processed_scene.city,

        "trajectories":

            prediction[
                "trajectories"
            ]

            .cpu()

            .tolist(),

        "scores":

            prediction[
                "scores"
            ]

            .cpu()

            .tolist(),

    }

    output_file = (

        RESULT_ROOT

        / f"{processed_scene.sequence_id}.json"

    )

    with open(

        output_file,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            output,

            file,

            indent=4,

        )

    print()

    print(

        f"Prediction JSON : "

        f"{output_file}"

    )


###############################################################################
# Save CSV
###############################################################################

def save_prediction_csv(
    *,
    prediction,
    processed_scene,
):
    """
    Save prediction CSV.
    """

    import csv

    output_file = (

        RESULT_ROOT

        / f"{processed_scene.sequence_id}.csv"

    )

    with open(

        output_file,

        "w",

        newline="",

    ) as file:

        writer = csv.writer(file)

        writer.writerow(

            [

                "agent",

                "step",

                "x",

                "y",

                "score",

            ]

        )

        trajectories = prediction[
            "trajectories"
        ]

        scores = prediction[
            "scores"
        ]

        B, N, T, _ = trajectories.shape

        for agent in range(N):

            score = float(

                scores[0, agent]

            )

            for step in range(T):

                point = trajectories[

                    0,

                    agent,

                    step,

                ]

                writer.writerow(

                    [

                        agent,

                        step,

                        float(point[0]),

                        float(point[1]),

                        score,

                    ]

                )

    print(

        f"Prediction CSV  : "

        f"{output_file}"

    )


###############################################################################
# Save Results
###############################################################################

def save_predictions(
    *,
    prediction,
    processed_scene,
):
    """
    Save inference outputs.
    """

    save_prediction_json(

        prediction=prediction,

        processed_scene=processed_scene,

    )

    save_prediction_csv(

        prediction=prediction,

        processed_scene=processed_scene,

    )

###############################################################################
# Reusable Inference Pipeline
###############################################################################

def run_inference_pipeline(
    *,
    scene_path: Path = DEFAULT_SCENE,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    result_root: Path | None = None,
):
    """
    Reusable production inference pipeline.
    """

    global RESULT_ROOT

    if result_root is not None:

        RESULT_ROOT = result_root

    create_directories()

    ###########################################################################
    # Model
    ###########################################################################

    model = build_model()

    ###########################################################################
    # Checkpoint
    ###########################################################################

    load_checkpoint(

        model,

        checkpoint,

    )

    ###########################################################################
    # Scene
    ###########################################################################

    batch, processed_scene = load_scene(

        scene_path,

    )

    ###########################################################################
    # Forward
    ###########################################################################

    _, refined_prediction, inference_time = (

        run_inference(

            model=model,

            batch=batch,

        )

    )

    ###########################################################################
    # Best Mode
    ###########################################################################

    prediction = select_best_mode(

        refined_prediction,

    )

    ###########################################################################
    # Recover World Coordinates
    ###########################################################################

    prediction["trajectories"] = recover_world_coordinates(

        prediction=prediction["trajectories"],

        processed_scene=processed_scene,

    )

    ###########################################################################
    # Print
    ###########################################################################

    print_inference_summary(

        inference_time=inference_time,

        refined_prediction=refined_prediction,

    )

    print_prediction_summary(

        prediction=prediction,

    )

    ###########################################################################
    # Save
    ###########################################################################

    save_predictions(

        prediction=prediction,

        processed_scene=processed_scene,

    )

    return prediction

###############################################################################
# Main
###############################################################################

def main() -> None:

    print_header(
        "DSTNet Production Inference"
    )

    run_inference_pipeline(

        scene_path=DEFAULT_SCENE,

        checkpoint=DEFAULT_CHECKPOINT,

        result_root=RESULT_ROOT,

    )

    print()

    print("=" * 80)

    print("✓ INFERENCE COMPLETE")

    print("=" * 80)


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()
