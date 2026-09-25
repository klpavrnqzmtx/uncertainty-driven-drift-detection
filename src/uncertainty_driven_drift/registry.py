"""Minimal component registry.

Components are registered under a *kind* (``dataset``, ``model``,
``uncertainty``, ``detector``) and a *name*. The runner looks them up by
name at build time, which keeps the runner independent of any specific
implementation and makes adding new components a one-decorator change.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

Kind = str
Name = str
Factory = Callable[..., Any]

_REGISTRY: Dict[Kind, Dict[Name, Factory]] = {
    "dataset": {},
    "model": {},
    "uncertainty": {},
    "detector": {},
}


def register(kind: Kind, name: Name) -> Callable[[Factory], Factory]:
    """Decorator to register a factory (class or function) under ``kind``/``name``."""
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown kind {kind!r}; expected one of {list(_REGISTRY)}")

    def _decorator(factory: Factory) -> Factory:
        if name in _REGISTRY[kind]:
            raise ValueError(f"{kind}/{name} already registered")
        _REGISTRY[kind][name] = factory
        return factory

    return _decorator


def build(kind: Kind, name: Name, /, **kwargs: Any) -> Any:
    """Instantiate a component by its registered name."""
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown kind {kind!r}")
    try:
        factory = _REGISTRY[kind][name]
    except KeyError as exc:
        available = sorted(_REGISTRY[kind])
        raise KeyError(
            f"No {kind} named {name!r}. Available: {available}"
        ) from exc
    return factory(**kwargs)


def available(kind: Kind) -> list[Name]:
    return sorted(_REGISTRY[kind])
