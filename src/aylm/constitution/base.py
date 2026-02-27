"""宪法原则基类定义。

本模块定义了几何宪法式 AI 的核心抽象：
- Severity: 违规严重性等级
- ViolationResult: 违规检测结果
- ConstitutionPrinciple: 宪法原则基类

用户可以继承 ConstitutionPrinciple 实现自定义的安全规则。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import AIDecision, SceneState


class Severity(Enum):
    """违规严重性等级。

    用于定义宪法原则被违反时的严重程度，
    影响最终安全分数的计算和推荐动作。
    """

    CRITICAL = 1.0  # 关键：必须立即阻止（如即将碰撞）
    HIGH = 0.8  # 高：强烈警告，建议干预
    MEDIUM = 0.5  # 中：一般警告，需要注意
    LOW = 0.2  # 低：轻微提示，可忽略


@dataclass
class ViolationResult:
    """违规检测结果。

    Attributes:
        violated: 是否违反原则
        severity: 违规严重性
        confidence: 检测置信度 (0.0-1.0)
        description: 违规描述（人类可读）
        metrics: 相关度量值（如距离、TTC 等）
        correction_hint: 纠正建议（可选，用于生成训练信号）
    """

    violated: bool
    severity: Severity
    confidence: float
    description: str
    metrics: dict[str, float] = field(default_factory=dict)
    correction_hint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "violated": self.violated,
            "severity": self.severity.name,
            "severity_weight": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "metrics": self.metrics,
            "correction_hint": self.correction_hint,
        }


class ConstitutionPrinciple(ABC):
    """宪法原则基类。

    所有安全规则都应继承此类并实现 evaluate 方法。
    这是 A-YLM 几何宪法式 AI 的核心抽象。

    Example:
        >>> class NoCollisionPrinciple(ConstitutionPrinciple):
        ...     @property
        ...     def name(self) -> str:
        ...         return "no_collision"
        ...
        ...     @property
        ...     def severity(self) -> Severity:
        ...         return Severity.CRITICAL
        ...
        ...     def evaluate(self, state, decision) -> ViolationResult:
        ...         # 实现碰撞检测逻辑
        ...         collision = check_collision(state.obstacles, decision.trajectory)
        ...         return ViolationResult(
        ...             violated=collision,
        ...             severity=self.severity,
        ...             confidence=0.95,
        ...             description="检测到碰撞风险" if collision else "安全",
        ...         )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """原则名称（唯一标识符）。"""
        pass

    @property
    @abstractmethod
    def severity(self) -> Severity:
        """默认严重性等级。"""
        pass

    @property
    def description(self) -> str:
        """原则描述（可选覆盖）。"""
        return f"宪法原则: {self.name}"

    @property
    def enabled(self) -> bool:
        """是否启用（可动态控制）。"""
        return True

    @abstractmethod
    def evaluate(
        self,
        state: "SceneState",
        decision: "AIDecision",
    ) -> ViolationResult:
        """评估 AI 决策是否违反此原则。

        Args:
            state: 当前场景状态（包含障碍物、自车状态等）
            decision: AI 的决策（包含规划轨迹、控制指令等）

        Returns:
            ViolationResult: 违规检测结果
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, severity={self.severity.name})"
