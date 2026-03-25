"""Pipeline 适配器层。

将 A-YLM 感知流水线的数据转换为宪法模块类型。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..tools.motion_estimator import MotionVector
    from ..tools.obstacle_marker import ObstacleBox3D


@dataclass
class ObstacleMotion:
    """障碍物运动信息（宪法模块内部使用）。"""

    velocity_robot: NDArray[np.float32]  # 机器人坐标系速度 [vx, vy, vz] m/s
    speed: float  # 标量速度 m/s
    heading: float  # 航向角（弧度）
    is_stationary: bool  # 是否静止


@dataclass
class ConstitutionObstacle:
    """宪法模块统一障碍物表示。

    统一不同来源的障碍物数据（Python 对象或 JSON 字典），
    提供给宪法原则评估使用。
    """

    center_robot: NDArray[np.float32]  # 机器人坐标系中心 [x, y, z]
    dimensions: NDArray[np.float32]  # 尺寸 [length, width, height]（机器人坐标系）
    label: str  # 语义标签名称
    confidence: float  # 置信度
    track_id: Optional[int] = None
    motion: Optional[ObstacleMotion] = None

    @classmethod
    def from_pipeline_obstacle(
        cls,
        obstacle: "ObstacleBox3D",
        motion: "Optional[MotionVector]" = None,
    ) -> "ConstitutionObstacle":
        """从 Pipeline ObstacleBox3D 转换。"""
        center = np.array(obstacle.center_robot, dtype=np.float32)
        dims = np.array(obstacle.dimensions_robot, dtype=np.float32)

        obs_motion = None
        if motion is not None:
            obs_motion = ObstacleMotion(
                velocity_robot=np.array(motion.velocity_robot, dtype=np.float32),
                speed=motion.speed,
                heading=motion.heading,
                is_stationary=motion.is_stationary,
            )

        return cls(
            center_robot=center,
            dimensions=dims,
            label=obstacle.label.name,
            confidence=obstacle.confidence,
            track_id=obstacle.track_id,
            motion=obs_motion,
        )

    @classmethod
    def from_obstacle_dict(cls, data: dict[str, Any]) -> "ConstitutionObstacle":
        """从 Pipeline JSON 输出字典转换。"""
        center = np.array(data["center_robot"], dtype=np.float32)
        dims = np.array(
            data.get("dimensions_robot", data.get("dimensions_cv", [1, 1, 1])),
            dtype=np.float32,
        )

        obs_motion = None
        motion_data = data.get("motion")
        if motion_data:
            vel = motion_data.get("velocity_robot", [0, 0, 0])
            obs_motion = ObstacleMotion(
                velocity_robot=np.array(vel, dtype=np.float32),
                speed=motion_data.get("speed", 0.0),
                heading=motion_data.get("heading", 0.0),
                is_stationary=motion_data.get("is_stationary", True),
            )

        return cls(
            center_robot=center,
            dimensions=dims,
            label=data.get("_label", data.get("category", "unknown")),
            confidence=data.get("confidence", 0.0),
            track_id=data.get("track_id"),
            motion=obs_motion,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        result: dict[str, Any] = {
            "center_robot": self.center_robot.tolist(),
            "dimensions": self.dimensions.tolist(),
            "label": self.label,
            "confidence": self.confidence,
        }
        if self.track_id is not None:
            result["track_id"] = self.track_id
        if self.motion is not None:
            result["motion"] = {
                "velocity_robot": self.motion.velocity_robot.tolist(),
                "speed": float(self.motion.speed),
                "heading": float(self.motion.heading),
                "is_stationary": bool(self.motion.is_stationary),
            }
        return result
