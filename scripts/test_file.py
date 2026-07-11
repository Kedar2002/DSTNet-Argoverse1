from pathlib import Path

from datasets.cache_manager import CacheManager
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser


def main() -> None:

    parser = SceneParser(None)

    scene = parser(
        Path("data/argoverse1/train/1.csv")
    )

    preprocessor = ScenePreprocessor(
        observation_steps=20,
        prediction_steps=30,
        lane_sample_points=20,
        agent_radius=30.0,
        lane_radius=20.0,
    )

    processed = preprocessor.preprocess(scene)

    cache = CacheManager("cache")

    cache.save(processed)

    loaded = cache.load(
        processed.sequence_id
    )

    print("=" * 70)
    print("Cache Summary")
    print("=" * 70)

    print(cache.summary())

    print()

    print("Loaded Scene")

    print(loaded)

    print()

    print("Agents :", loaded.num_agents)

    print("Lanes  :", loaded.num_lanes)

    print()

    print("Cache Exists :", cache.exists(processed.sequence_id))


if __name__ == "__main__":
    main()
