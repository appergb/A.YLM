"""速度限制检测宪法原则。

提供速度限制检测的参考实现。
"""

from typing import TYPE_CHECKING

from ..base import ConstitutionPrinciple, Severity, ViolationResult
from ..registry import ConstitutionRegistry

if TYPE_CHECKING:
    from ..types import AIDecision, SceneState


@ConstitutionRegistry.register_principle("speed_limit")
class SpeedLimitPrinciple(ConstitutionPrinciple):
    """速度限制原则。

    检测自车速度和目标速度是否超过速度限制。

    数学表述：
        v_ego ≤ v_limit
        v_target ≤ v_limit

    参数：
        speed_limit: 速度限制（m/s），默认 13.89 即 50km/h
        warning_ratio: 预警比例（0-1），接近限速时触发警告
    """

    def __init__(
        self,
        speed_limit: float = 13.89,
        warning_ratio: float = 0.9,
    ):
        self.speed_limit = speed_limit
        self.warning_ratio = warning_ratio

    @property
    def name(self) -> str:
        return "speed_limit"

    @property
    def severity(self) -> Severity:
        return Severity.MEDIUM

    @property
    def description(self) -> str:
        return "检测车辆速度是否超过限速"

    def evaluate(
        self,
        state: "SceneState",
        decision: "AIDecision",
    ) -> ViolationResult:
        """评估速度合规性。

        Args:
            state: 场景状态（包含自车速度）
            decision: AI 决策（包含目标速度）

        Returns:
            ViolationResult: 违规检测结果
        """
        ego_speed = state.ego_state.speed
        target_speed = decision.target_speed
        speed_ratio = ego_speed / self.speed_limit if self.speed_limit > 0 else 0.0

        metrics = {
            "ego_speed": float(ego_speed),
            "target_speed": float(target_speed) if target_speed is not None else -1,
            "speed_limit": self.speed_limit,
            "speed_ratio": float(speed_ratio),
        }

        # 检查 1：当前速度超过限速 → 关键违规
        if ego_speed > self.speed_limit:
            return ViolationResult(
                violated=True,
                severity=Severity.CRITICAL,
                confidence=0.95,
                description=(
                    f"超速: {ego_speed:.1f}m/s > 限速 {self.speed_limit:.1f}m/s"
                ),
                metrics=metrics,
                correction_hint={
                    "action": "reduce_speed",
                    "target_speed": self.speed_limit,
                },
            )

        # 检查 2：目标速度超过限速 → 高严重性违规
        if target_speed is not None and target_speed > self.speed_limit:
            return ViolationResult(
                violated=True,
                severity=Severity.HIGH,
                confidence=0.9,
                description=(
                    f"目标速度超限: {target_speed:.1f}m/s > "
                    f"限速 {self.speed_limit:.1f}m/s"
                ),
                metrics=metrics,
                correction_hint={
                    "action": "reduce_target_speed",
                    "target_speed": self.speed_limit,
                },
            )

        # 检查 3：接近限速 → 中等警告
        if ego_speed > self.speed_limit * self.warning_ratio:
            return ViolationResult(
                violated=True,
                severity=Severity.MEDIUM,
                confidence=0.8,
                description=(
                    f"接近限速: {ego_speed:.1f}m/s " f"({speed_ratio:.0%} 限速)"
                ),
                metrics=metrics,
                correction_hint={
                    "action": "maintain_speed",
                    "max_speed": self.speed_limit * self.warning_ratio,
                },
            )

        # 速度安全
        return ViolationResult(
            violated=False,
            severity=self.severity,
            confidence=0.95,
            description=f"速度合规: {ego_speed:.1f}m/s",
            metrics=metrics,
        )
