"""
scripts.test_dstnet

Complete DSTNet integration test.

Pipeline

CSV
 ↓
SceneParser
 ↓
ScenePreprocessor
 ↓
ArgoverseDataset
 ↓
DataLoader
 ↓
DSTNet
 ↓
Forward Pass

This script is intended to verify that the entire repository
is correctly wired together before training.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.collate import collate_fn
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser

from datasets.map_loader import MapLoader

from models.dstnet import DSTNet

###########################################################################
# Configuration
###########################################################################

DATA_ROOT = Path("data/argoverse1/train")

MAP_ROOT = Path("data/argoverse1/hd_maps/map_files")

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 2

OBSERVATION_STEPS = 20

PREDICTION_STEPS = 30

LANE_POINTS = 20

HIDDEN_DIM = 256

NUM_HEADS = 8

NUM_LAYERS = 2

NUM_MODES = 6

###########################################################################
# Dataset
###########################################################################

print("=" * 80)
print("Building Dataset")
print("=" * 80)

###########################################################################
# Load HD Maps
###########################################################################

print("Map root:", MAP_ROOT.resolve())
print("XML files found:")

for xml in MAP_ROOT.glob("pruned_argoverse_*_vector_map.xml"):
    print("  ", xml.name)

print("\nLoading HD Maps...\n")

map_loader = MapLoader(
    map_root=MAP_ROOT,
)

print(map_loader.summary())

###########################################################################
# Scene Parser
###########################################################################

parser = SceneParser(
    map_api=map_loader,
)

preprocessor = ScenePreprocessor(
    observation_steps=OBSERVATION_STEPS,
    prediction_steps=PREDICTION_STEPS,
    lane_sample_points=LANE_POINTS,
    agent_radius=30.0,
    lane_radius=20.0,
)

dataset = ArgoverseDataset(
    root=DATA_ROOT,
    parser=parser,
    preprocessor=preprocessor,
)

print(dataset.summary())

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

###########################################################################
# Model
###########################################################################

print("\nBuilding Model...\n")

model = DSTNet(
    observation_steps=OBSERVATION_STEPS,
    prediction_steps=PREDICTION_STEPS,
    lane_points=LANE_POINTS,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_encoder_layers=NUM_LAYERS,
    num_modes=NUM_MODES,
).to(DEVICE)

model.eval()

print(model)

###########################################################################
# Load One Batch
###########################################################################

print("\nLoading one batch...\n")

batch = next(
    iter(loader)
)

print("Batch Keys")

for key in batch:

    value = batch[key]

    if torch.is_tensor(value):

        print(
            f"{key:25s} {tuple(value.shape)}"
        )

    elif isinstance(value, list):

        print(
            f"{key:25s} list[{len(value)}]"
        )

    else:

        print(
            f"{key:25s} {type(value)}"
        )

###########################################################################
# Move tensors to device
###########################################################################

for key in [

    "agent_trajectories",

    "lane_centerlines",

    "positions",

    "headings",

    "agent_mask",

    "lane_mask",

]:

    batch[key] = batch[key].to(
        DEVICE,
    )

print("\nInput Shapes")

print(
    "agent_trajectories :",
    batch["agent_trajectories"].shape,
)

print(
    "lane_centerlines   :",
    batch["lane_centerlines"].shape,
)

print(
    "positions          :",
    batch["positions"].shape,
)

print(
    "headings           :",
    batch["headings"].shape,
)

###########################################################################
# Forward Pass
###########################################################################

print("\nRunning DSTNet...\n")

try:

    with torch.no_grad():

        coarse, refined = model(

            agent_trajectories=batch[
                "agent_trajectories"
            ],

            lane_centerlines=batch[
                "lane_centerlines"
            ],

            positions=batch[
                "positions"
            ],

            headings=batch[
                "headings"
            ],

            graph=batch[
                "graph"
            ],

            agent_mask=batch[
                "agent_mask"
            ],

            lane_mask=batch[
                "lane_mask"
            ],
        )

    print("\nSUCCESS\n")

    print(
        "Coarse trajectories :",
        coarse.trajectories.shape,
    )

    print(
        "Coarse scores       :",
        coarse.scores.shape,
    )

    print(
        "Refined trajectories:",
        refined.trajectories.shape,
    )

    print(
        "Offsets             :",
        refined.offsets.shape,
    )

except Exception:

    print("\nFAILED\n")

    traceback.print_exc()


