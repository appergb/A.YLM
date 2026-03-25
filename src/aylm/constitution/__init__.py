"""A-YLM 几何宪法式 AI 核心模块。

本模块提供开放式接口，允许用户自定义：
- 宪法原则（ConstitutionPrinciple）
- 安全打分器（SafetyScorer）
- 训练信号生成器（TrainingSignalGenerator）
- 指令解析器（CommandParser）
- 指令验证器（CommandValidator）

设计理念：
- 我们提供基础设施和接口定义
- 具体的打分逻辑和训练方式由用户根据需求实现
- 支持插件机制，便于第三方扩展

导入此模块即自动注册所有内置原则、打分器、训练信号生成器和指令解析器。
"""

# 导入 principles 子包，触发装饰器自注册
from . import principles  # noqa: F401
from .adapter import ConstitutionObstacle, ObstacleMotion
from .base import (
    ConstitutionPrinciple,
    Severity,
    ViolationResult,
)
from .command_parser import CommandParser, JSONCommandParser, NaturalLanguageParser
from .config import ConstitutionConfig
from .default_generator import DefaultTrainingSignalGenerator
from .evaluator import ConstitutionEvaluator, EvaluationResult
from .registry import ConstitutionRegistry
from .scorer import RecommendedAction, SafetyScore, SafetyScorer
from .training import SignalType, TrainingSignal, TrainingSignalGenerator
from .types import AIDecision, EgoState, SceneState, TrajectoryPoint
from .validator import CommandValidator, ValidationResult
from .weighted_scorer import WeightedSafetyScorer

__all__ = [
    # 基础类型
    "Severity",
    "ViolationResult",
    "RecommendedAction",
    "SignalType",
    # 抽象基类
    "ConstitutionPrinciple",
    "SafetyScorer",
    "TrainingSignalGenerator",
    "CommandParser",
    # 数据类型
    "SceneState",
    "AIDecision",
    "EgoState",
    "TrajectoryPoint",
    "SafetyScore",
    "TrainingSignal",
    "ConstitutionConfig",
    # 适配器
    "ConstitutionObstacle",
    "ObstacleMotion",
    # 具体实现
    "WeightedSafetyScorer",
    "DefaultTrainingSignalGenerator",
    "JSONCommandParser",
    "NaturalLanguageParser",
    # 编排器
    "ConstitutionEvaluator",
    "EvaluationResult",
    # 验证器
    "CommandValidator",
    "ValidationResult",
    # 注册机制
    "ConstitutionRegistry",
]
