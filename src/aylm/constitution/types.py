"""宪法式 AI 类型定义。

定义场景状态、AI 决策等核心数据结构。
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class EgoState:
    """自车/机器人状态。

    Attributes:
        position: 位置 [x, y, z]（机器人坐标系）
        velocity: 速度 [vx, vy, vz]（m/s）
        heading: 航向角（弧度）
        speed: 标量速度（m/s）
        acceleration: 加速度 [ax, ay, az]（m/s²）
        dimensions: 尺寸 [length, width, height]（米）
    """

    position: NDArray[np.float32]
    velocity: NDArray[np.float32]
    heading: float
    speed: float
    acceleration: NDArray[np.float32] | None = None
    dimensions: NDArray[np.float32] | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "heading": float(self.heading),
            "speed": float(self.speed),
            "acceleration": (
                self.acceleration.tolist() if self.acceleration is not None else None
            ),
            "dimensions": (
                self.dimensions.tolist() if self.dimensions is not None else None
            ),
        }


@dataclass
class TrajectoryPoint:
    """轨迹点。"""

    position: NDArray[np.float32]  # [x, y, z]
    velocity: NDArray[np.float32] | None = None  # [vx, vy, vz]
    timestamp: float = 0.0  # 相对时间（秒）

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryPoint":
        """从字典创建。

        支持格式：
        - {"position": [x,y,z], "velocity": [vx,vy,vz], "timestamp": t}
        - {"position": [x,y,z], "timestamp": t}
        """
        pos = np.array(data["position"], dtype=np.float32)
        vel = None
        if data.get("velocity") is not None:
            vel = np.array(data["velocity"], dtype=np.float32)
        return cls(
            position=pos,
            velocity=vel,
            timestamp=float(data.get("timestamp", 0.0)),
        )


@dataclass
class AIDecision:
    """AI 决策。

    表示端到端 AI 系统输出的决策，可以是轨迹规划或控制指令。

    Attributes:
        decision_type: 决策类型（trajectory, control, waypoint）
        trajectory: 规划轨迹（轨迹点列表）
        control: 控制指令（steering, throttle, brake）
        target_speed: 目标速度（m/s）
        confidence: 决策置信度
        metadata: 额外元数据
    """

    decision_type: str  # "trajectory", "control", "waypoint"
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    control: dict[str, float] = field(default_factory=dict)  # steering, throttle, brake
    target_speed: float | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "decision_type": self.decision_type,
            "trajectory": [
                {
                    "position": p.position.tolist(),
                    "velocity": p.velocity.tolist() if p.velocity is not None else None,
                    "timestamp": p.timestamp,
                }
                for p in self.trajectory
            ],
            "control": self.control,
            "target_speed": self.target_speed,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AIDecision":
        """从字典创建（支持外部 JSON 输入）。

        Args:
            data: 包含 decision_type, trajectory, control 等字段的字典

        Returns:
            AIDecision 实例
        """
        trajectory = []
        for pt in data.get("trajectory", []):
            trajectory.append(TrajectoryPoint.from_dict(pt))

        return cls(
            decision_type=data.get("decision_type", "trajectory"),
            trajectory=trajectory,
            control=data.get("control", {}),
            target_speed=data.get("target_speed"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SceneState:
    """场景状态。

    包含当前帧的所有感知信息，供宪法原则评估使用。

    Attributes:
        frame_id: 帧 ID
        timestamp: 时间戳（秒）
        ego_state: 自车状态
        obstacles: 障碍物列表（来自 A-YLM 感知模块）
        lane_boundaries: 车道边界（可选）
        traffic_signs: 交通标志（可选）
        metadata: 额外元数据
    """

    frame_id: int
    timestamp: float
    ego_state: EgoState
    obstacles: list[Any] = field(default_factory=list)  # ObstacleBox3D 列表
    lane_boundaries: list[Any] | None = None
    traffic_signs: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "ego_state": self.ego_state.to_dict(),
            "obstacles": [
                obs.to_dict() if hasattr(obs, "to_dict") else obs
                for obs in self.obstacles
            ],
            "lane_boundaries": self.lane_boundaries,
            "traffic_signs": self.traffic_signs,
            "metadata": self.metadata,
        }
