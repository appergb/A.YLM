"""Offline video navigation demo orchestration for A-YLM.

保持包级导入轻量化，避免在 import 时拉起重依赖。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DEFAULT_MLX_MODEL, NavigationDemoConfig
    from .runner import NavigationDemoRunner

__all__ = [
    "DEFAULT_MLX_MODEL",
    "NavigationDemoConfig",
    "NavigationDemoRunner",
]

_LAZY_ATTRS = {
    "DEFAULT_MLX_MODEL": "aylm.navigation_demo.config",
    "NavigationDemoConfig": "aylm.navigation_demo.config",
    "NavigationDemoRunner": "aylm.navigation_demo.runner",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(
            f"module 'aylm.navigation_demo' has no attribute {name!r}"
        )
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())))
