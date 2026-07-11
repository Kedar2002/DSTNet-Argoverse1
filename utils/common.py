"""
Common utility functions used across the DSTNet repository.

This module intentionally contains only generic helper functions that
are independent of any specific model or dataset component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str | Path
        Directory path.

    Returns
    -------
    Path
        Created directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file(path: str | Path) -> Path:
    """
    Validate that a file exists.

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    return path


def validate_directory(path: str | Path) -> Path:
    """
    Validate that a directory exists.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Path
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(path)

    return path


def list_files(
    directory: str | Path,
    suffix: str | tuple[str, ...],
) -> list[Path]:
    """
    Return sorted files matching a suffix.

    Parameters
    ----------
    directory
        Input directory.

    suffix
        File extension(s).

    Returns
    -------
    list[Path]
    """
    directory = validate_directory(directory)

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in (
            suffix if isinstance(suffix, tuple) else (suffix,)
        )
    )


def save_json(
    obj: dict[str, Any],
    path: str | Path,
) -> None:
    """
    Save dictionary as JSON.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load JSON file.
    """
    path = validate_file(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load YAML file.
    """
    path = validate_file(path)

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(
    obj: dict[str, Any],
    path: str | Path,
) -> None:
    """
    Save YAML file.
    """
    path = Path(path)

    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            obj,
            f,
            sort_keys=False,
        )


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = False,
) -> int:
    """
    Count model parameters.

    Parameters
    ----------
    trainable_only : bool
        Count only trainable parameters.
    """
    if trainable_only:
        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

    return sum(
        p.numel()
        for p in model.parameters()
    )


def move_to_device(
    obj: Any,
    device: torch.device,
) -> Any:
    """
    Recursively move tensors to a device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {
            k: move_to_device(v, device)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            move_to_device(v, device)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            move_to_device(v, device)
            for v in obj
        )

    return obj


def has_nan_or_inf(tensor: torch.Tensor) -> bool:
    """
    Check whether a tensor contains NaN or Inf values.

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    bool
        True if the tensor contains NaN or Inf values.
    """
    return bool(
        torch.isnan(tensor).any().item()
        or torch.isinf(tensor).any().item()
    )


def assert_tensor_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Raise an error if tensor contains NaN or Inf.
    """
    if has_nan_or_inf(tensor):
        raise ValueError(
            f"Tensor '{name}' contains NaN or Inf values."
        )
