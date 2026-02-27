"""TTC（碰撞时间）安全原则。

提供 TTC 计算的参考实现。
"""

from typing import TYPE_CHECKING

import numpy as np

from ..base import ConstitutionPrinciple, Severity, ViolationResult
from ..registry import ConstitutionRegistry

if TYPE_CHECKING:
    from ..types import AIDecision, SceneState


@ConstitutionRegistry.register_principle("ttc_safety")
class TTCSafetyPrinciple(ConstitutionPrinciple):
    """TTC 安全原则。

    计算与动态障碍物的碰撞时间（Time To Collision）。

    数学表述：
        ∀o ∈ O_dynamic: TTC(ego, o) > τ_min

    其中：
        TTC = (d - d_safe) / v_rel

    参数：
        warning_threshold: 警告阈值（秒）
        critical_threshold: 关键阈值（秒）
        min_safe_distance: 最小安全距离（米）
    """

    def __init__(
        self,
        warning_threshold: float = 3.0,
        critical_threshold: float = 1.5,
        min_safe_distance: float = 2.0,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_safe_distance = min_safe_distance

    @property
    def name(self) -> str:
        return "ttc_safety"

    @property
    def severity(self) -> Severity:
        return Severity.HIGH

    @property
    def description(self) -> str:
        return "检测与动态障碍物的碰撞时间是否安全"

    def evaluate(
        self,
        state: "SceneState",
        decision: "AIDecision",
    ) -> ViolationResult:
        """评估 TTC 安全性。

        Args:
            state: 场景状态
            decision: AI 决策

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

        min_ttc = float("inf")

        ego_pos = state.ego_state.position
        ego_vel = state.ego_state.velocity

        for obstacle in state.obstacles:
            # 只检测动态障碍物
            if hasattr(obstacle, "motion") and obstacle.motion:
                motion = obstacle.motion
                if motion.is_stationary:
                    continue

                # 获取障碍物位置和速度
                if hasattr(obstacle, "center_robot"):
                    obs_pos = np.array(obstacle.center_robot)
                else:
                    continue

                obs_vel = np.array(motion.velocity_robot)

                # 计算相对位置和速度
                rel_pos = obs_pos - ego_pos
                rel_vel = obs_vel - ego_vel

                # 计算距离
                distance = np.linalg.norm(rel_pos)

                # 计算相对速度在连线方向的分量
                if distance > 0:
                    direction = rel_pos / distance
                    closing_speed = -np.dot(rel_vel, direction)

                    # 只有在接近时才计算 TTC
                    if closing_speed > 0.1:  # 接近速度阈值
                        ttc = (distance - self.min_safe_distance) / closing_speed

                        if ttc < min_ttc:
                            min_ttc = ttc
                            _ = obstacle  # 保留用于未来扩展

        # 判断违规
        if min_ttc < self.critical_threshold:
            violated = True
            severity = Severity.CRITICAL
            desc = f"TTC 关键警告: {min_ttc:.1f}s < {self.critical_threshold}s"
        elif min_ttc < self.warning_threshold:
            violated = True
            severity = Severity.HIGH
            desc = f"TTC 警告: {min_ttc:.1f}s < {self.warning_threshold}s"
        else:
            violated = False
            severity = self.severity
            desc = f"TTC 安全: {min_ttc:.1f}s"

        return ViolationResult(
            violated=violated,
            severity=severity,
            confidence=0.85,
            description=desc,
            metrics={
                "min_ttc": float(min_ttc) if min_ttc != float("inf") else -1,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
            },
            correction_hint=(
                {
                    "action": "slow_down",
                    "target_ttc": self.warning_threshold,
                }
                if violated
                else None
            ),
        )
