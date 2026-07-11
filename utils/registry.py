"""
Generic registry implementation used throughout the DSTNet repository.

The registry provides a simple mechanism for registering classes or
factory functions by name and retrieving them later.

Example
-------
from utils.registry import Registry

MODELS = Registry("models")

@MODELS.register()
class DSTNet:
    ...

model_cls = MODELS.get("DSTNet")
model = model_cls(...)
"""

from __future__ import annotations

from typing import Any, Callable

RegistryItem = Callable[..., Any]


class Registry:
    """
    Generic object registry.

    Parameters
    ----------
    name : str
        Human-readable registry name.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, RegistryItem] = {}

    @property
    def name(self) -> str:
        """
        Registry name.
        """
        return self._name

    @property
    def registered_names(self) -> list[str]:
        """
        Return all registered names.
        """
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self._name}', "
            f"items={len(self)})"
        )

    def register(
        self,
        name: str | None = None,
    ) -> Callable[[RegistryItem], RegistryItem]:
        """
        Register a class or callable.

        Parameters
        ----------
        name : str, optional
            Custom registration name.

        Returns
        -------
        Callable
            Decorator.
        """

        def decorator(obj: RegistryItem) -> RegistryItem:
            key = name or obj.__name__

            if key in self._registry:
                raise KeyError(
                    f"'{key}' is already registered "
                    f"in registry '{self._name}'."
                )

            self._registry[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> RegistryItem:
        """
        Retrieve a registered object.

        Parameters
        ----------
        name : str

        Returns
        -------
        Registered object.
        """
        if name not in self._registry:
            available = ", ".join(self.registered_names)

            raise KeyError(
                f"'{name}' is not registered in "
                f"'{self._name}'. "
                f"Available: [{available}]"
            )

        return self._registry[name]

    def build(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Instantiate a registered class.

        Parameters
        ----------
        name : str
            Registered class name.

        Returns
        -------
        Any
            Constructed object.
        """
        cls = self.get(name)
        return cls(*args, **kwargs)

    def clear(self) -> None:
        """
        Remove all registered objects.

        Mainly useful for testing.
        """
        self._registry.clear()


###############################################################################
# Global registries
###############################################################################

MODELS = Registry("models")

DATASETS = Registry("datasets")

LOSSES = Registry("losses")

OPTIMIZERS = Registry("optimizers")

SCHEDULERS = Registry("schedulers")

METRICS = Registry("metrics")
