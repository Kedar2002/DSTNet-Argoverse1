from pathlib import Path

from datasets.collate import collate_fn
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser


def build_scene(csv_path: Path):

    parser = SceneParser(None)

    raw = parser(csv_path)

    preprocessor = ScenePreprocessor(
        observation_steps=20,
        prediction_steps=30,
        lane_sample_points=20,
        agent_radius=30.0,
        lane_radius=20.0,
    )

    return preprocessor.preprocess(raw)


def main():

    scene1 = build_scene(
        Path("data/argoverse1/train/1.csv")
    )

    scene2 = build_scene(
        Path("data/argoverse1/train/2.csv")
    )

    batch = collate_fn(
        [scene1, scene2]
    )

    print("=" * 70)
    print("Batch Summary")
    print("=" * 70)

    print(batch["agents"]["observed"].shape)
    print(batch["agents"]["future"].shape)
    print(batch["agents"]["velocity"].shape)
    print(batch["agents"]["speed"].shape)
    print(batch["agents"]["acceleration"].shape)
    print(batch["agents"]["heading"].shape)

    print()

    print(batch["metadata"]["sequence_id"])


if __name__ == "__main__":
    main()
