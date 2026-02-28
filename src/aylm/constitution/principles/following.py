"""跟车距离检测宪法原则。

提供安全跟车距离检测的参考实现。
"""

from typing import TYPE_CHECKING

import numpy as np

from ..base import ConstitutionPrinciple, Severity, ViolationResult
from ..registry import ConstitutionRegistry

if TYPE_CHECKING:
    from ..types import AIDecision, SceneState


@ConstitutionRegistry.register_principle("safe_following")
class SafeFollowingPrinciple(ConstitutionPrinciple):
    """安全跟车距离原则。

    检测自车与前方动态障碍物的跟车距离是否满足安全要求。

    数学表述：
        d_required = v_ego * t_gap + d_min
        ∀o ∈ O_ahead: dist(ego, o) > d_required

    参数：
        time_gap: 时间间隔系数（秒）
        min_gap: 最小安全间距（米）
    """

    def __init__(
        self,
        time_gap: float = 2.0,
        min_gap: float = 2.0,
    ):
        self.time_gap = time_gap
        self.min_gap = min_gap

    @property
    def name(self) -> str:
        return "safe_following"

    @property
    def severity(self) -> Severity:
        return Severity.HIGH

    @property
    def description(self) -> str:
        return "检测与前方车辆的跟车距离是否安全"

    def evaluate(
        self,
        state: "SceneState",
        decision: "AIDecision",
    ) -> ViolationResult:
        """评估跟车距离安全性。

        Args:
            state: 场景状态（包含障碍物列表和自车状态）
            decision: AI 决策（包含规划轨迹）

        Returns:
            ViolationResult: 违规检测结果
        """
        if not state.obstacles:
            return ViolationResult(
                violated=False,
                severity=self.severity,
                confidence=1.0,
                description="无障碍物",
            )

        if not decision.trajectory:
            return ViolationResult(
                violated=False,
                severity=self.severity,
                confidence=0.5,
                description="无轨迹信息，无法评估",
            )

        ego_pos = state.ego_state.position
        ego_speed = state.ego_state.speed
        required_distance = self.time_gap * ego_speed + self.min_gap

        min_distance = float("inf")

        for obstacle in state.obstacles:
            # 获取障碍物中心（机器人坐标系）
            if hasattr(obstacle, "center_robot"):
                obs_center = np.array(obstacle.center_robot)
            elif hasattr(obstacle, "center"):
                obs_center = np.array(obstacle.center)
            else:
                continue

            # 只检测前方障碍物（机器人坐标系 X 轴为前进方向）
            if obs_center[0] <= 0:
                continue

            # 只检测动态障碍物（非静止）
            if hasattr(obstacle, "motion") and obstacle.motion:
                if obstacle.motion.is_stationary:
                    continue
            else:
                continue

            # 计算距离
            distance = np.linalg.norm(obs_center - ego_pos)

            if distance < min_distance:
                min_distance = distance

        # 无符合条件的前方动态障碍物
        if min_distance == float("inf"):
            return ViolationResult(
                violated=False,
                severity=self.severity,
                confidence=1.0,
                description="前方无动态障碍物",
            )

        # 判断是否违规（bool() 确保为 Python bool 而非 np.bool_）
        violated = bool(min_distance < required_distance)

        return ViolationResult(
            violated=violated,
            severity=self.severity,
            confidence=0.85,
            description=(
                f"跟车距离不足: {min_distance:.2f}m < {required_distance:.2f}m"
                if violated
                else f"跟车距离安全: {min_distance:.2f}m"
            ),
            metrics={
                "min_distance": float(min_distance),
                "required_distance": float(required_distance),
            },
            correction_hint=(
                {
                    "action": "increase_following_distance",
                    "target_distance": required_distance,
                }
                if violated
                else None
            ),
        )
