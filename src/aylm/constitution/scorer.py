"""安全打分器模块。

本模块定义了安全打分的抽象接口：
- SafetyScore: 安全评分结果
- SafetyScorer: 安全打分器基类

用户可以继承 SafetyScorer 实现自定义的打分逻辑。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tools.obstacle_marker import ObstacleBox3D
    from .types import AIDecision, EgoState


class RecommendedAction(Enum):
    """推荐动作。"""

    SAFE = "safe"  # 安全，无需干预
    CAUTION = "caution"  # 注意，轻微风险
    WARNING = "warning"  # 警告，需要关注
    INTERVENTION = "intervention"  # 干预，建议接管
    EMERGENCY_STOP = "emergency_stop"  # 紧急停车


@dataclass
class SafetyScore:
    """安全评分结果。

    Attributes:
        overall: 综合安全分数 (0.0-1.0，1.0 最安全)
        collision_score: 碰撞风险分数
        ttc_score: TTC（碰撞时间）分数
        boundary_score: 边界合规分数
        violations: 违规的原则名称列表
        violation_details: 违规详情
        recommended_action: 推荐动作
        confidence: 评分置信度
    """

    overall: float
    collision_score: float = 1.0
    ttc_score: float = 1.0
    boundary_score: float = 1.0
    violations: list[str] = field(default_factory=list)
    violation_details: list[dict[str, Any]] = field(default_factory=list)
    recommended_action: RecommendedAction = RecommendedAction.SAFE
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "overall": self.overall,
            "scores": {
                "collision": self.collision_score,
                "ttc": self.ttc_score,
                "boundary": self.boundary_score,
            },
            "violations": self.violations,
            "violation_details": self.violation_details,
            "recommended_action": self.recommended_action.value,
            "confidence": self.confidence,
        }

    @property
    def is_safe(self) -> bool:
        """是否安全（无违规）。"""
        return len(self.violations) == 0

    @property
    def needs_intervention(self) -> bool:
        """是否需要干预。"""
        return self.recommended_action in (
            RecommendedAction.INTERVENTION,
            RecommendedAction.EMERGENCY_STOP,
        )


class SafetyScorer(ABC):
    """安全打分器基类。

    用户可以继承此类实现自定义的打分逻辑。
    不同的 AI 系统可能有不同的安全标准和打分方式。

    Example:
        >>> class MySafetyScorer(SafetyScorer):
        ...     def score(self, obstacles, ego_state, ai_decision) -> SafetyScore:
        ...         # 实现自定义打分逻辑
        ...         collision_risk = self._check_collision(obstacles, ai_decision)
        ...         ttc = self._compute_ttc(obstacles, ego_state)
        ...         return SafetyScore(
        ...             overall=min(collision_risk, ttc),
        ...             collision_score=collision_risk,
        ...             ttc_score=ttc,
        ...         )
    """

    @abstractmethod
    def score(
        self,
        obstacles: list["ObstacleBox3D"],
        ego_state: "EgoState",
        ai_decision: "AIDecision",
    ) -> SafetyScore:
        """计算安全分数。

        Args:
            obstacles: 3D 障碍物列表（来自 A-YLM 感知模块）
            ego_state: 自车状态（位置、速度、航向等）
            ai_decision: AI 的决策（规划轨迹、控制指令等）

        Returns:
            SafetyScore: 安全评分结果
        """
        pass

    def score_from_scene(
        self,
        scene_data: dict[str, Any],
    ) -> SafetyScore:
        """从场景数据计算安全分数（便捷方法）。

        Args:
            scene_data: 场景数据字典，包含 obstacles、ego_state、ai_decision

        Returns:
            SafetyScore: 安全评分结果
        """
        return self.score(
            obstacles=scene_data.get("obstacles", []),
            ego_state=scene_data.get("ego_state"),
            ai_decision=scene_data.get("ai_decision"),
        )
