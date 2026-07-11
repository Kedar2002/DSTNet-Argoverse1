"""
utils/config.py

Lightweight configuration system for DSTNet.

Features
--------
- Recursive attribute access
- Recursive dictionary merge
- Immutable configuration
- YAML loading
- Configuration overrides
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
from collections.abc import ItemsView, KeysView, ValuesView

import yaml

from utils.logger import get_logger

LOGGER = get_logger(__name__)

###############################################################################
# Repository Paths
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "configs"


###############################################################################
# Config Node
###############################################################################


class ConfigNode:
    """
    Recursive configuration object.

    Example
    -------

    cfg.dataset.root

    cfg.training.learning_rate

    cfg.runtime.device
    """

    __slots__ = (
        "_data",
        "_frozen",
    )

    def __init__(
        self,
        data: dict[str, Any],
    ) -> None:

        object.__setattr__(
            self,
            "_data",
            {},
        )

        object.__setattr__(
            self,
            "_frozen",
            False,
        )

        for key, value in data.items():

            if isinstance(value, dict):

                value = ConfigNode(value)

            self._data[key] = value

    ###########################################################################
    # Attribute Access
    ###########################################################################

    def __getattr__(
        self,
        name: str,
    ) -> Any:

        try:
            return self._data[name]

        except KeyError as exc:

            raise AttributeError(
                name
            ) from exc

    def __getitem__(
        self,
        key: str,
    ) -> Any:

        return self._data[key]

    def __contains__(
        self,
        key: str,
    ) -> bool:

        return key in self._data

    ###########################################################################
    # Iteration
    ###########################################################################

    def __iter__(
        self,
    ) -> Iterator[str]:

        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> ItemsView[str, Any]:
        return self._data.items()

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[Any]:
        return self._data.values()

    ###########################################################################
    # Mutation
    ###########################################################################

    def __setattr__(
        self,
        key: str,
        value: Any,
    ) -> None:

        if self._frozen:

            raise AttributeError(
                "Configuration is immutable."
            )

        if isinstance(
            value,
            dict,
        ):
            value = ConfigNode(value)

        self._data[key] = value

    ###########################################################################
    # Freeze
    ###########################################################################

    def freeze(
        self,
    ) -> None:
        """
        Make configuration immutable.
        """

        object.__setattr__(
            self,
            "_frozen",
            True,
        )

        for value in self._data.values():

            if isinstance(
                value,
                ConfigNode,
            ):
                value.freeze()

    ###########################################################################
    # Utilities
    ###########################################################################

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert recursively to a Python dict.
        """

        result = {}

        for key, value in self._data.items():

            if isinstance(
                value,
                ConfigNode,
            ):
                result[key] = value.to_dict()

            else:
                result[key] = value

        return result

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Dictionary-style get().
        """
        return self._data.get(key, default)

    def copy(self) -> "ConfigNode":
        """
        Return a mutable deep copy of this configuration.
        """
        return ConfigNode(self.to_dict())

    def as_dict(self) -> dict[str, Any]:
        """
        Alias for to_dict().
        """
        return self.to_dict()

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}"
            f"({self._data})"
        )


###############################################################################
# YAML Utilities
###############################################################################


def _load_yaml(
    path: Path,
) -> dict[str, Any]:
    """
    Load a YAML file.
    """

    if not path.exists():

        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(file)

    if data is None:

        return {}

    if not isinstance(
        data,
        dict,
    ):

        raise TypeError(
            f"{path.name} must contain a mapping."
        )

    return data


def _recursive_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge dictionaries.
    """

    result = deepcopy(base)

    for key, value in override.items():

        if (
            key in result
            and isinstance(
                result[key],
                dict,
            )
            and isinstance(
                value,
                dict,
            )
        ):

            result[key] = _recursive_merge(
                result[key],
                value,
            )

        else:

            result[key] = deepcopy(value)

    return result

###############################################################################
# Configuration Loading
###############################################################################


_BASE_CONFIGS = (
    "dataset",
    "model",
    "training",
    "runtime",
)

_OVERRIDE_CONFIGS = (
    "paper",
    "debug",
    "local_cpu",
)


def _config_path(name: str) -> Path:
    """
    Return the absolute path of a configuration file.
    """

    return CONFIG_DIR / f"{name}.yaml"


def _load_base_config() -> dict[str, Any]:
    """
    Load the repository base configuration.

    The following files are merged in order:

        dataset.yaml
        model.yaml
        training.yaml
        runtime.yaml
    """

    config: dict[str, Any] = {}

    for filename in _BASE_CONFIGS:

        LOGGER.info(
            "Loading %s.yaml",
            filename,
        )

        config = _recursive_merge(
            config,
            _load_yaml(
                _config_path(filename),
            ),
        )

    return config


def _load_override(
    override_name: str,
) -> dict[str, Any]:
    """
    Load one override configuration.
    """

    if override_name not in _OVERRIDE_CONFIGS:

        raise ValueError(
            f"Unknown override '{override_name}'."
        )

    LOGGER.info(
        "Applying override: %s.yaml",
        override_name,
    )

    return _load_yaml(
        _config_path(
            override_name,
        )
    )


def _build_config(
    override: str | None = None,
) -> ConfigNode:
    """
    Build a configuration object.

    Parameters
    ----------
    override
        Optional override configuration.
    """

    config = _load_base_config()

    if override is not None:

        config = _recursive_merge(
            config,
            _load_override(
                override,
            ),
        )

    cfg = ConfigNode(config)

    cfg.freeze()

    return cfg


###############################################################################
# Cached Loaders
###############################################################################


@lru_cache(maxsize=1)
def load_config() -> ConfigNode:
    """
    Load the default configuration.
    """

    LOGGER.info(
        "Loading repository configuration..."
    )

    cfg = _build_config()

    LOGGER.info(
        "Configuration loaded successfully."
    )

    return cfg


@lru_cache(maxsize=1)
def load_paper_config() -> ConfigNode:
    """
    Load the paper reproduction configuration.
    """

    LOGGER.info(
        "Loading paper configuration..."
    )

    return _build_config(
        "paper",
    )


@lru_cache(maxsize=1)
def load_debug_config() -> ConfigNode:
    """
    Load the debug configuration.
    """

    LOGGER.info(
        "Loading debug configuration..."
    )

    return _build_config(
        "debug",
    )


@lru_cache(maxsize=1)
def load_local_cpu_config() -> ConfigNode:
    """
    Load the local CPU configuration.
    """

    LOGGER.info(
        "Loading local CPU configuration..."
    )

    return _build_config(
        "local_cpu",
    )


###############################################################################
# Public Accessors
###############################################################################


def get_config() -> ConfigNode:
    """
    Return the default repository configuration.
    """

    return load_config()


def get_paper_config() -> ConfigNode:
    """
    Return the paper configuration.
    """

    return load_paper_config()


def get_debug_config() -> ConfigNode:
    """
    Return the debug configuration.
    """

    return load_debug_config()


def get_local_cpu_config() -> ConfigNode:
    """
    Return the local CPU configuration.
    """

    return load_local_cpu_config()

###############################################################################
# Reload Helpers
###############################################################################


def reload_config() -> ConfigNode:
    """
    Reload the default configuration.
    """

    load_config.cache_clear()

    return load_config()


def reload_all() -> None:
    """
    Clear every cached configuration.
    """

    load_config.cache_clear()

    load_paper_config.cache_clear()

    load_debug_config.cache_clear()

    load_local_cpu_config.cache_clear()


###############################################################################
# Validation Helpers
###############################################################################


def validate_paths(
    cfg: ConfigNode,
) -> None:
    """
    Validate required dataset paths.
    """

    required = (
        cfg.dataset.root,
        cfg.dataset.train_dir,
        cfg.dataset.val_dir,
        cfg.dataset.test_dir,
    )

    for path in required:

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Required path does not exist:\n{path}"
            )


###############################################################################
# Summary
###############################################################################


def print_summary(
    cfg: ConfigNode,
) -> None:
    """
    Print repository configuration summary.
    """

    LOGGER.info("=" * 70)

    LOGGER.info("DSTNet Configuration")

    LOGGER.info("=" * 70)

    if "experiment" in cfg:

        LOGGER.info(
            "Experiment : %s",
            cfg.experiment.name,
        )

        LOGGER.info("")

    LOGGER.info("Dataset")

    LOGGER.info(
        "  Root              : %s",
        cfg.dataset.root,
    )

    LOGGER.info(
        "  Observation       : %d",
        cfg.dataset.observation_steps,
    )

    LOGGER.info(
        "  Prediction        : %d",
        cfg.dataset.prediction_steps,
    )

    LOGGER.info(
        "  Max Agents        : %d",
        cfg.dataset.max_agents,
    )

    LOGGER.info(
        "  Max Lanes         : %d",
        cfg.dataset.max_lanes,
    )

    LOGGER.info("")

    LOGGER.info("Model")

    LOGGER.info(
        "  Hidden Dim        : %d",
        cfg.model.hidden_dim,
    )

    LOGGER.info(
        "  Modes             : %d",
        cfg.model.num_modes,
    )

    LOGGER.info(
        "  Refinement        : %s",
        cfg.model.enable_refinement,
    )

    LOGGER.info("")

    LOGGER.info("Training")

    LOGGER.info(
        "  Optimizer         : %s",
        cfg.training.optimizer,
    )

    LOGGER.info(
        "  Scheduler         : %s",
        cfg.training.scheduler,
    )

    LOGGER.info(
        "  LR                : %.6f",
        cfg.training.learning_rate,
    )

    LOGGER.info(
        "  Batch Size        : %d",
        cfg.training.batch_size,
    )

    LOGGER.info(
        "  Epochs            : %d",
        cfg.training.epochs,
    )

    LOGGER.info("")

    LOGGER.info("Runtime")

    LOGGER.info(
        "  Device            : %s",
        cfg.runtime.device,
    )

    LOGGER.info(
        "  Workers           : %d",
        cfg.runtime.num_workers,
    )

    LOGGER.info(
        "  Seed              : %d",
        cfg.runtime.seed,
    )

    LOGGER.info("=" * 70)


###############################################################################
# Module Exports
###############################################################################

__all__ = [
    "ConfigNode",
    "load_config",
    "load_paper_config",
    "load_debug_config",
    "load_local_cpu_config",
    "reload_config",
    "reload_all",
    "get_config",
    "get_paper_config",
    "get_debug_config",
    "get_local_cpu_config",
    "validate_paths",
    "print_summary",
]
