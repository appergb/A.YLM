"""A-YLM 几何宪法式 AI 核心模块。

本模块提供开放式接口，允许用户自定义：
- 宪法原则（ConstitutionPrinciple）
- 安全打分器（SafetyScorer）
- 训练信号生成器（TrainingSignalGenerator）

设计理念：
- 我们提供基础设施和接口定义
- 具体的打分逻辑和训练方式由用户根据需求实现
- 支持插件机制，便于第三方扩展
"""

from .base import (
    ConstitutionPrinciple,
    Severity,
    ViolationResult,
)
from .config import ConstitutionConfig
from .registry import ConstitutionRegistry
from .scorer import SafetyScore, SafetyScorer
from .training import TrainingSignal, TrainingSignalGenerator

__all__ = [
    # 基础类型
    "Severity",
    "ViolationResult",
    # 抽象基类
    "ConstitutionPrinciple",
    "SafetyScorer",
    "TrainingSignalGenerator",
    # 数据类
    "SafetyScore",
    "TrainingSignal",
    "ConstitutionConfig",
    # 注册机制
    "ConstitutionRegistry",
]
