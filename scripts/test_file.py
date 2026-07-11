from pathlib import Path

from datasets.scene_parser import SceneParser
from datasets.transforms import (
    build_eval_transform,
    build_train_transform,
)


scene = SceneParser(None)(
    Path("data/argoverse1/train/1.csv")
)

print("Original")

print(scene.target_track.last_position)

train = build_train_transform()

scene2 = train(scene)

print()

print("Augmented")

print(scene2.target_track.last_position)

print()

print("Evaluation")

scene3 = build_eval_transform()(scene)

print(scene3.target_track.last_position)
