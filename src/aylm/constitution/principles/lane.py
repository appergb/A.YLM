"""车道合规检测宪法原则。

提供车道偏离检测的参考实现。
"""

from typing import TYPE_CHECKING

from ..base import ConstitutionPrinciple, Severity, ViolationResult
from ..registry import ConstitutionRegistry

if TYPE_CHECKING:
    from ..types import AIDecision, SceneState


@ConstitutionRegistry.register_principle("lane_compliance")
class LaneCompliancePrinciple(ConstitutionPrinciple):
    """车道合规原则。

    检测 AI 决策轨迹是否保持在车道范围内。

    数学表述：
        max_offset = (w_lane / 2) * r_max
        ∀p ∈ trajectory: |p.y| < max_offset

    参数：
        lane_width: 车道宽度（米）
        max_offset_ratio: 最大偏移比例（0-1）
    """

    def __init__(
        self,
        lane_width: float = 3.5,
        max_offset_ratio: float = 0.9,
    ):
        self.lane_width = lane_width
        self.max_offset_ratio = max_offset_ratio

    @property
    def name(self) -> str:
        return "lane_compliance"

    @property
    def severity(self) -> Severity:
        return Severity.MEDIUM

    @property
    def description(self) -> str:
        return "检测轨迹是否偏离车道"

    def evaluate(
        self,
        state: "SceneState",
        decision: "AIDecision",
    ) -> ViolationResult:
        """评估车道合规性。

        Args:
            state: 场景状态（包含车道边界信息）
            decision: AI 决策（包含规划轨迹）

        Returns:
            ViolationResult: 违规检测结果
        """
        if state.lane_boundaries is None:
            return ViolationResult(
                violated=False,
                severity=self.severity,
                confidence=0.3,
                description="无车道数据，跳过检测",
            )

        if not decision.trajectory:
            return ViolationResult(
                violated=False,
                severity=self.severity,
                confidence=0.5,
                description="无轨迹信息，无法评估",
            )

        max_lateral_offset = self.lane_width / 2 * self.max_offset_ratio

        # 检测轨迹点的横向偏移
        max_lateral_deviation = 0.0

        for traj_point in decision.trajectory:
            # 机器人坐标系中 Y 分量为横向偏移
            lateral = abs(traj_point.position[1])

            if lateral > max_lateral_deviation:
                max_lateral_deviation = lateral

        # 判断是否违规（bool() 确保为 Python bool 而非 np.bool_）
        violated = bool(max_lateral_deviation > max_lateral_offset)

        return ViolationResult(
            violated=violated,
            severity=self.severity,
            confidence=0.8,
            description=(
                f"车道偏离: 最大偏移 {max_lateral_deviation:.2f}m > {max_lateral_offset:.2f}m"
                if violated
                else f"车道合规: 最大偏移 {max_lateral_deviation:.2f}m"
            ),
            metrics={
                "max_lateral_deviation": float(max_lateral_deviation),
                "lane_width": self.lane_width,
                "max_offset": float(max_lateral_offset),
            },
            correction_hint=(
                {
                    "action": "return_to_lane_center",
                    "max_lateral_offset": max_lateral_offset,
                }
                if violated
                else None
            ),
        )
