from pathlib import Path

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.cache_manager import CacheManager
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser


def main():

    parser = SceneParser(None)

    preprocessor = ScenePreprocessor(
        observation_steps=20,
        prediction_steps=30,
        lane_sample_points=20,
        agent_radius=30.0,
        lane_radius=20.0,
    )

    dataset = ArgoverseDataset(
        root=Path("data/argoverse1/train"),
        parser=parser,
        preprocessor=preprocessor,
        cache=CacheManager("cache"),
    )

    print(dataset)

    print(dataset.summary())

    print()

    print("Dataset Size")

    print(len(dataset))

    print()

    scene = dataset[0]

    print(scene)

    print()

    print(scene.summary())


if __name__ == "__main__":
    main()
