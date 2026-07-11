"""
datasets.geometry

Geometry utilities for DSTNet.

All functions operate on NumPy arrays and are deterministic.

Coordinate System
-----------------
Raw Argoverse coordinates

        ↓

Target Agent Centered

        ↓

Heading Aligned

        ↓

Normalized Coordinates
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-8


###############################################################################
# Rotation
###############################################################################


def rotation_matrix(
    theta: float,
) -> np.ndarray:
    """
    Compute a 2D rotation matrix.

    Parameters
    ----------
    theta
        Rotation angle in radians.

    Returns
    -------
    ndarray
        Shape (2, 2)
    """

    c = math.cos(theta)
    s = math.sin(theta)

    return np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float32,
    )


def rotate_points(
    points: np.ndarray,
    theta: float,
) -> np.ndarray:
    """
    Rotate 2D points.

    Parameters
    ----------
    points
        Shape (...,2)

    theta
        Rotation angle.
    """

    r = rotation_matrix(theta)

    return points @ r.T


def translate_points(
    points: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Translate points.

    Parameters
    ----------
    points
        Shape (...,2)

    translation
        Shape (2,)
    """

    return points + translation


###############################################################################
# Coordinate Transform
###############################################################################


def transform_points(
    points: np.ndarray,
    origin: np.ndarray,
    heading: float,
) -> np.ndarray:
    """
    Transform world coordinates into the local target frame.

    The transformation is:

        translate
            ↓
        rotate
    """

    translated = points - origin

    return rotate_points(
        translated,
        -heading,
    )


def inverse_transform_points(
    points: np.ndarray,
    origin: np.ndarray,
    heading: float,
) -> np.ndarray:
    """
    Convert local coordinates back to world coordinates.
    """

    rotated = rotate_points(
        points,
        heading,
    )

    return rotated + origin


###############################################################################
# Angle Utilities
###############################################################################


def normalize_angle(
    theta: float,
) -> float:
    """
    Normalize angle to [-pi, pi].
    """

    return math.atan2(
        math.sin(theta),
        math.cos(theta),
    )


def angle_difference(
    theta1: float,
    theta2: float,
) -> float:
    """
    Smallest signed angle difference.
    """

    return normalize_angle(
        theta1 - theta2,
    )


def heading_vector(
    theta: float,
) -> np.ndarray:
    """
    Unit vector from heading angle.
    """

    return np.array(
        [
            math.cos(theta),
            math.sin(theta),
        ],
        dtype=np.float32,
    )

###############################################################################
# Heading Estimation
###############################################################################


def compute_heading(
    points: np.ndarray,
) -> float:
    """
    Estimate heading from the last valid trajectory segment.

    Parameters
    ----------
    points
        Shape (N, 2)

    Returns
    -------
    float
        Heading angle in radians.
    """

    if len(points) < 2:
        return 0.0

    delta = points[-1] - points[-2]

    if np.linalg.norm(delta) < EPS:
        return 0.0

    return math.atan2(delta[1], delta[0])


def compute_headings(
    trajectory: np.ndarray,
) -> np.ndarray:
    """
    Compute heading at every trajectory point.

    Parameters
    ----------
    trajectory
        Shape (N, 2)

    Returns
    -------
    ndarray
        Shape (N,)
    """

    n = len(trajectory)

    headings = np.zeros(
        n,
        dtype=np.float32,
    )

    if n < 2:
        return headings

    for i in range(1, n):

        delta = trajectory[i] - trajectory[i - 1]

        if np.linalg.norm(delta) < EPS:

            headings[i] = headings[i - 1]

        else:

            headings[i] = math.atan2(
                delta[1],
                delta[0],
            )

    headings[0] = headings[1]

    return headings


###############################################################################
# Vector Utilities
###############################################################################


def safe_normalize(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Normalize a vector safely.
    """

    norm = np.linalg.norm(vector)

    if norm < EPS:
        return np.zeros_like(vector)

    return vector / norm


###############################################################################
# Velocity
###############################################################################


def compute_velocity(
    points: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Compute velocity for every point.

    Parameters
    ----------
    points
        Shape (N,2)

    dt
        Sampling interval.

    Returns
    -------
    ndarray
        Shape (N,2)
    """

    velocity = np.zeros_like(
        points,
        dtype=np.float32,
    )

    if len(points) < 2:
        return velocity

    velocity[1:] = (
        points[1:] - points[:-1]
    ) / dt

    velocity[0] = velocity[1]

    return velocity


###############################################################################
# Speed
###############################################################################


def compute_speed(
    points: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Compute scalar speed.
    """

    velocity = compute_velocity(
        points,
        dt,
    )

    return np.linalg.norm(
        velocity,
        axis=1,
    )


###############################################################################
# Acceleration
###############################################################################


def compute_acceleration(
    points: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Compute acceleration.
    """

    velocity = compute_velocity(
        points,
        dt,
    )

    acceleration = np.zeros_like(
        velocity,
        dtype=np.float32,
    )

    if len(points) < 2:
        return acceleration

    acceleration[1:] = (
        velocity[1:] - velocity[:-1]
    ) / dt

    acceleration[0] = acceleration[1]

    return acceleration


###############################################################################
# Trajectory Statistics
###############################################################################


def trajectory_length(
    points: np.ndarray,
) -> float:
    """
    Compute total trajectory length.
    """

    if len(points) < 2:
        return 0.0

    segments = np.diff(
        points,
        axis=0,
    )

    return float(
        np.linalg.norm(
            segments,
            axis=1,
        ).sum()
    )

###############################################################################
# Distance Utilities
###############################################################################


def euclidean_distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:
    """
    Compute Euclidean distance between two points.

    Parameters
    ----------
    point_a
        Shape (2,)

    point_b
        Shape (2,)

    Returns
    -------
    float
    """

    return float(
        np.linalg.norm(point_a - point_b)
    )


def pairwise_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.

    Parameters
    ----------
    points_a
        Shape (N,2)

    points_b
        Shape (M,2)

    Returns
    -------
    ndarray
        Shape (N,M)
    """

    diff = (
        points_a[:, None, :]
        - points_b[None, :, :]
    )

    return np.linalg.norm(
        diff,
        axis=-1,
    )


def closest_point(
    points: np.ndarray,
    query: np.ndarray,
) -> tuple[int, float]:
    """
    Find the closest point in a point set.

    Parameters
    ----------
    points
        Shape (N,2)

    query
        Shape (2,)

    Returns
    -------
    index
        Index of closest point.

    distance
        Euclidean distance.
    """

    distances = np.linalg.norm(
        points - query,
        axis=1,
    )

    index = int(np.argmin(distances))

    return index, float(distances[index])


###############################################################################
# Lane Utilities
###############################################################################


def lane_length(
    centerline: np.ndarray,
) -> float:
    """
    Compute total lane centerline length.

    Parameters
    ----------
    centerline
        Shape (N,2)

    Returns
    -------
    float
    """

    return trajectory_length(centerline)


def lane_direction(
    centerline: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized lane direction.

    Returns
    -------
    ndarray
        Shape (2,)
    """

    if len(centerline) < 2:

        return np.zeros(
            2,
            dtype=np.float32,
        )

    direction = (
        centerline[-1]
        - centerline[0]
    )

    return safe_normalize(direction)


def sample_centerline(
    centerline: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Uniformly sample a lane centerline using arc-length interpolation.

    Parameters
    ----------
    centerline
        Shape (N, 2)

    num_points
        Number of output samples.

    Returns
    -------
    ndarray
        Shape (num_points, 2)
    """

    centerline = np.asarray(
        centerline,
        dtype=np.float32,
    )

    if centerline.ndim != 2:
        raise ValueError(
            "centerline must have shape (N,2)."
        )

    if centerline.shape[1] != 2:
        raise ValueError(
            "centerline must have shape (N,2)."
        )

    if len(centerline) == 0:
        raise ValueError(
            "centerline cannot be empty."
        )

    if len(centerline) == 1:
        return np.repeat(
            centerline,
            num_points,
            axis=0,
        )

    # Segment lengths
    segments = np.diff(
        centerline,
        axis=0,
    )

    lengths = np.linalg.norm(
        segments,
        axis=1,
    )

    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(lengths),
        )
    )

    total_length = cumulative[-1]

    if total_length < EPS:

        return np.repeat(
            centerline[:1],
            num_points,
            axis=0,
        )

    target_distances = np.linspace(
        0.0,
        total_length,
        num_points,
    )

    sampled = np.empty(
        (num_points, 2),
        dtype=np.float32,
    )

    for i, distance in enumerate(target_distances):

        segment = np.searchsorted(
            cumulative,
            distance,
            side="right",
        ) - 1

        segment = min(
            segment,
            len(lengths) - 1,
        )

        local_distance = (
            distance
            - cumulative[segment]
        )

        if lengths[segment] < EPS:

            sampled[i] = centerline[segment]

            continue

        alpha = (
            local_distance
            / lengths[segment]
        )

        sampled[i] = (
            (1.0 - alpha)
            * centerline[segment]
            + alpha
            * centerline[segment + 1]
        )

    return sampled


###############################################################################
# Bounding Utilities
###############################################################################


def bounding_box(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box.

    Returns
    -------
    minimum
        Shape (2,)

    maximum
        Shape (2,)
    """

    return (
        np.min(points, axis=0),
        np.max(points, axis=0),
    )


def inside_radius(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Determine whether each point lies inside a radius.

    Parameters
    ----------
    points
        Shape (N,2)

    center
        Shape (2,)

    radius
        Radius in meters.

    Returns
    -------
    ndarray
        Boolean mask of shape (N,)
    """

    distances = np.linalg.norm(
        points - center,
        axis=1,
    )

    return distances <= radius
