"""Shared settings for the processed Argoverse cache format."""

CACHE_VERSION = "2.0"

# Keep this in one place so the Kaggle cache builder and training entry point
# preprocess scenes with exactly the same geometry and tensor dimensions.
KAGGLE_PREPROCESSING_CONFIG = {
    "observation_steps": 20,
    "prediction_steps": 30,
    "frame_rate": 10.0,
    "map_sample_points": 20,
    "spatial_radius": 30.0,
    "map_radius": 30.0,
}
