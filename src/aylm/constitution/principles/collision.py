"""碰撞检测宪法原则。

提供碰撞检测的参考实现，用户可以继承或修改。
"""

from typing import TYPE_CHECKING

import numpy as np

from ..base import ConstitutionPrinciple, Severity, ViolationResult
from ..registry import ConstitutionRegistry

if TYPE_CHECKING:
    from ..types import AIDecision, SceneState


@ConstitutionRegistry.register_principle("no_collision")
class NoCollisionPrinciple(ConstitutionPrinciple):
    """无碰撞原则。

    检测 AI 决策轨迹是否与障碍物发生碰撞。

    数学表述：
        ∀t: V_ego(t) ∩ V_obstacle(t) = ∅

    参数：
        safety_margin: 安全边距（米）
        prediction_horizon: 预测时间范围（秒）
    """

    def __init__(
        self,
        safety_margin: float = 0.5,
        prediction_horizon: float = 2.0,
    ):
        self.safety_margin = safety_margin
        self.prediction_horizon = prediction_horizon

    @property
    def name(self) -> str:
        return "no_collision"

    @property
    def severity(self) -> Severity:
        return Severity.CRITICAL

    @property
    def description(self) -> str:
        return "检测 AI 决策是否会导致碰撞"

    def evaluate(
        self,
        state: "SceneState",
        decision: "AIDecision",
    ) -> ViolationResult:
        """评估碰撞风险。

        Args:
            state: 场景状态（包含障碍物列表）
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

        # 检测轨迹与障碍物的最小距离
        min_distance = float("inf")
        collision_obstacle = None

        for traj_point in decision.trajectory:
            if traj_point.timestamp > self.prediction_horizon:
                break

            for obstacle in state.obstacles:
                # 获取障碍物中心（支持不同格式）
                if hasattr(obstacle, "center_robot"):
                    obs_center = np.array(obstacle.center_robot)
                elif hasattr(obstacle, "center"):
                    obs_center = np.array(obstacle.center)
                else:
                    continue

                # 计算距离
                distance = np.linalg.norm(traj_point.position - obs_center)

                # 考虑障碍物尺寸
                if hasattr(obstacle, "dimensions"):
                    # 简化：使用最大维度作为半径
                    obs_radius = max(obstacle.dimensions) / 2
                    distance -= obs_radius

                if distance < min_distance:
                    min_distance = distance
                    collision_obstacle = obstacle

        # 判断是否碰撞
        collision = min_distance < self.safety_margin

        return ViolationResult(
            violated=collision,
            severity=self.severity,
            confidence=0.9 if collision else 0.95,
            description=(
                f"检测到碰撞风险，最小距离 {min_distance:.2f}m"
                if collision
                else f"安全，最小距离 {min_distance:.2f}m"
            ),
            metrics={
                "min_distance": float(min_distance),
                "safety_margin": self.safety_margin,
            },
            correction_hint=(
                {
                    "action": "avoid",
                    "obstacle_position": (
                        collision_obstacle.center_robot
                        if hasattr(collision_obstacle, "center_robot")
                        else None
                    ),
                }
                if collision and collision_obstacle
                else None
            ),
        )
