"""Offline video navigation demo orchestration for A-YLM.

保持包级导入轻量化，避免在 import 时拉起重依赖。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .calibrator import CalibrationContext, FrameFeedback, SessionCalibrator
    from .config import DEFAULT_MLX_MODEL, NavigationDemoConfig
    from .learning_store import BaselineSnapshot, LearningStore, SessionRecord
    from .prompt_calibrator import PromptCalibrator, PromptPatch
    from .runner import NavigationDemoRunner

__all__ = [
    "BaselineSnapshot",
    "CalibrationContext",
    "DEFAULT_MLX_MODEL",
    "FrameFeedback",
    "LearningStore",
    "NavigationDemoConfig",
    "NavigationDemoRunner",
    "PromptCalibrator",
    "PromptPatch",
    "SessionCalibrator",
    "SessionRecord",
]

_LAZY_ATTRS = {
    "BaselineSnapshot": "aylm.navigation_demo.learning_store",
    "CalibrationContext": "aylm.navigation_demo.calibrator",
    "DEFAULT_MLX_MODEL": "aylm.navigation_demo.config",
    "FrameFeedback": "aylm.navigation_demo.calibrator",
    "LearningStore": "aylm.navigation_demo.learning_store",
    "NavigationDemoConfig": "aylm.navigation_demo.config",
    "NavigationDemoRunner": "aylm.navigation_demo.runner",
    "PromptCalibrator": "aylm.navigation_demo.prompt_calibrator",
    "PromptPatch": "aylm.navigation_demo.prompt_calibrator",
    "SessionCalibrator": "aylm.navigation_demo.calibrator",
    "SessionRecord": "aylm.navigation_demo.learning_store",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'aylm.navigation_demo' has no attribute {name!r}")
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())))
