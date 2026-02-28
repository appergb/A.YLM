"""指令解析器模块。

将外部系统（LLM / 端到端模型 / ROS）的指令解析为 AIDecision，
供宪法评估器评估安全性。

支持插件扩展：用户可继承 CommandParser 并通过
@ConstitutionRegistry.register_command_parser() 注册自定义格式。
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from .registry import ConstitutionRegistry
from .types import AIDecision, TrajectoryPoint


class CommandParser(ABC):
    """指令解析器基类。

    用户可扩展以支持自定义指令格式（如 ROS message、protobuf 等）。

    Example:
        >>> @ConstitutionRegistry.register_command_parser("ros")
        ... class ROSCommandParser(CommandParser):
        ...     def can_parse(self, command):
        ...         return isinstance(command, dict) and "header" in command
        ...     def parse(self, command, **context):
        ...         # 从 ROS 消息解析轨迹
        ...         ...
    """

    @abstractmethod
    def can_parse(self, command: str | dict) -> bool:
        """检查是否能解析该指令格式。

        Args:
            command: 外部指令（字符串或字典）

        Returns:
            是否能解析
        """

    @abstractmethod
    def parse(self, command: str | dict, **context: Any) -> AIDecision:
        """将外部指令解析为 AIDecision。

        Args:
            command: 外部指令
            **context: 额外上下文（如 ego_speed, ego_heading）

        Returns:
            AIDecision: 解析后的 AI 决策

        Raises:
            ValueError: 指令格式无法解析
        """


@ConstitutionRegistry.register_command_parser("json")
class JSONCommandParser(CommandParser):
    """JSON 格式指令解析器。

    支持格式：
    - trajectory: {"type": "trajectory", "points": [[x,y,z,t], ...]}
    - control: {"type": "control", "steering": 0.1, "throttle": 0.5, "brake": 0.0}
    - waypoint: {"type": "waypoint", "target": [x,y,z], "speed": 5.0}
    """

    def can_parse(self, command: str | dict) -> bool:
        if isinstance(command, dict):
            return "type" in command
        return False

    def parse(self, command: str | dict, **context: Any) -> AIDecision:
        if not isinstance(command, dict):
            raise ValueError("JSONCommandParser 只接受 dict 类型指令")

        cmd_type = command.get("type", "")
        ego_speed = float(context.get("ego_speed", 0.0))
        ego_heading = float(context.get("ego_heading", 0.0))

        if cmd_type == "trajectory":
            return self._parse_trajectory(command)
        elif cmd_type == "control":
            return self._parse_control(command, ego_speed, ego_heading)
        elif cmd_type == "waypoint":
            return self._parse_waypoint(command, ego_speed, ego_heading)
        else:
            raise ValueError(f"不支持的 JSON 指令类型: {cmd_type}")

    def _parse_trajectory(self, command: dict) -> AIDecision:
        points = command.get("points", [])
        trajectory = []
        for pt in points:
            if len(pt) >= 3:
                pos = np.array(pt[:3], dtype=np.float32)
                ts = float(pt[3]) if len(pt) > 3 else 0.0
                trajectory.append(TrajectoryPoint(position=pos, timestamp=ts))

        return AIDecision(
            decision_type="trajectory",
            trajectory=trajectory,
            target_speed=command.get("target_speed"),
            confidence=command.get("confidence", 1.0),
            metadata={"source": "json_command", "raw_type": "trajectory"},
        )

    def _parse_control(
        self, command: dict, ego_speed: float, ego_heading: float
    ) -> AIDecision:
        steering = float(command.get("steering", 0.0))
        throttle = float(command.get("throttle", 0.0))
        brake = float(command.get("brake", 0.0))

        # 从控制指令生成预测轨迹（简化运动学模型）
        trajectory = self._control_to_trajectory(
            steering, throttle, brake, ego_speed, ego_heading
        )

        target_speed = ego_speed + (throttle - brake) * 5.0
        target_speed = max(0.0, target_speed)

        return AIDecision(
            decision_type="control",
            trajectory=trajectory,
            control={
                "steering": steering,
                "throttle": throttle,
                "brake": brake,
            },
            target_speed=target_speed,
            confidence=command.get("confidence", 1.0),
            metadata={"source": "json_command", "raw_type": "control"},
        )

    def _parse_waypoint(
        self, command: dict, ego_speed: float, ego_heading: float
    ) -> AIDecision:
        target = command.get("target", [0, 0, 0])
        speed = float(command.get("speed", ego_speed))
        target_pos = np.array(target[:3], dtype=np.float32)

        # 生成从原点到目标的简单轨迹
        distance = float(np.linalg.norm(target_pos))
        if distance < 0.01 or speed < 0.01:
            trajectory = [
                TrajectoryPoint(
                    position=np.array([0, 0, 0], dtype=np.float32),
                    timestamp=0.0,
                )
            ]
        else:
            duration = distance / speed
            steps = max(2, min(10, int(duration / 0.1)))
            trajectory = []
            for i in range(steps):
                t = duration * i / (steps - 1)
                ratio = i / (steps - 1)
                pos = target_pos * ratio
                trajectory.append(
                    TrajectoryPoint(
                        position=pos.astype(np.float32),
                        timestamp=float(t),
                    )
                )

        return AIDecision(
            decision_type="waypoint",
            trajectory=trajectory,
            target_speed=speed,
            confidence=command.get("confidence", 1.0),
            metadata={"source": "json_command", "raw_type": "waypoint"},
        )

    @staticmethod
    def _control_to_trajectory(
        steering: float,
        throttle: float,
        brake: float,
        ego_speed: float,
        ego_heading: float,
        dt: float = 0.1,
        horizon: float = 2.0,
    ) -> list[TrajectoryPoint]:
        """从控制指令生成预测轨迹（简化自行车模型）。"""
        wheelbase = 2.5  # 轴距（米）
        x, y, heading = 0.0, 0.0, ego_heading
        speed = ego_speed

        trajectory = []
        t = 0.0
        while t <= horizon:
            trajectory.append(
                TrajectoryPoint(
                    position=np.array([x, y, 0], dtype=np.float32),
                    timestamp=t,
                )
            )
            # 更新状态
            accel = (throttle - brake) * 5.0  # 简化加速度
            speed = max(0.0, speed + accel * dt)
            if abs(steering) > 1e-6 and speed > 0.1:
                turn_radius = wheelbase / math.tan(steering)
                d_heading = speed * dt / turn_radius
            else:
                d_heading = 0.0
            heading += d_heading
            x += speed * dt * math.cos(heading)
            y += speed * dt * math.sin(heading)
            t += dt

        return trajectory


@ConstitutionRegistry.register_command_parser("natural_language")
class NaturalLanguageParser(CommandParser):
    """自然语言指令解析器。

    使用关键词规则引擎解析，不依赖外部 LLM。
    支持中英文关键词。

    支持指令示例：
    - "向左转弯30度" / "turn left 30 degrees"
    - "加速到60km/h" / "accelerate to 60"
    - "紧急刹车" / "emergency brake"
    - "变道到右侧" / "change lane right"
    - "保持当前速度" / "maintain speed"
    - "后退" / "reverse"
    - "停车" / "stop"
    """

    # 关键词规则
    _PATTERNS: ClassVar[list[tuple[str, str, dict[str, Any]]]] = [
        # (正则, 动作类型, 默认参数)
        (r"紧急刹车|emergency\s*brak", "emergency_brake", {}),
        (r"停车|stop", "stop", {}),
        (r"后退|reverse|倒车", "reverse", {}),
        (
            r"(?:向)?左转(?:弯)?(\d+)?度?|turn\s*left\s*(\d+)?",
            "turn_left",
            {"angle": 30},
        ),
        (
            r"(?:向)?右转(?:弯)?(\d+)?度?|turn\s*right\s*(\d+)?",
            "turn_right",
            {"angle": 30},
        ),
        (
            r"加速(?:到)?(\d+)|accelerat\w*\s*(?:to\s*)?(\d+)",
            "accelerate",
            {"target_kmh": 60},
        ),
        (r"减速|slow\s*down|decelerat", "decelerate", {}),
        (r"变道.*右|change\s*lane.*right|右变道", "lane_change_right", {}),
        (r"变道.*左|change\s*lane.*left|左变道", "lane_change_left", {}),
        (r"保持|maintain|keep", "maintain", {}),
        (r"直行|go\s*straight|前进|forward", "go_straight", {}),
    ]

    def can_parse(self, command: str | dict) -> bool:
        if not isinstance(command, str):
            return False
        cmd = command.strip()
        if not cmd:
            return False
        for pattern, _, _ in self._PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return True
        return False

    def parse(self, command: str | dict, **context: Any) -> AIDecision:
        if not isinstance(command, str):
            raise ValueError("NaturalLanguageParser 只接受 str 类型指令")

        ego_speed = float(context.get("ego_speed", 0.0))
        ego_heading = float(context.get("ego_heading", 0.0))
        cmd = command.strip()

        for pattern, action_type, defaults in self._PATTERNS:
            match = re.search(pattern, cmd, re.IGNORECASE)
            if match:
                return self._build_decision(
                    action_type, match, defaults, ego_speed, ego_heading, cmd
                )

        raise ValueError(f"无法解析自然语言指令: {command}")

    def _build_decision(
        self,
        action_type: str,
        match: re.Match,
        defaults: dict,
        ego_speed: float,
        ego_heading: float,
        raw_command: str,
    ) -> AIDecision:
        """根据解析的动作类型构建 AIDecision。"""
        metadata = {
            "source": "natural_language",
            "raw_command": raw_command,
            "parsed_action": action_type,
        }

        if action_type == "emergency_brake":
            return self._make_brake_decision(ego_speed, metadata)

        if action_type == "stop":
            return self._make_stop_decision(ego_speed, metadata)

        if action_type == "reverse":
            return self._make_reverse_decision(metadata)

        if action_type in ("turn_left", "turn_right"):
            angle = defaults["angle"]
            for g in match.groups():
                if g is not None:
                    angle = int(g)
                    break
            sign = -1 if action_type == "turn_left" else 1
            return self._make_turn_decision(
                angle, sign, ego_speed, ego_heading, metadata
            )

        if action_type == "accelerate":
            target_kmh = defaults.get("target_kmh", 60)
            for g in match.groups():
                if g is not None:
                    target_kmh = int(g)
                    break
            target_speed = target_kmh / 3.6
            return self._make_straight_decision(target_speed, ego_heading, metadata)

        if action_type == "decelerate":
            target_speed = max(0.0, ego_speed * 0.5)
            return self._make_straight_decision(target_speed, ego_heading, metadata)

        if action_type in ("lane_change_left", "lane_change_right"):
            sign = -1 if "left" in action_type else 1
            return self._make_lane_change_decision(
                sign, ego_speed, ego_heading, metadata
            )

        if action_type in ("maintain", "go_straight"):
            return self._make_straight_decision(ego_speed, ego_heading, metadata)

        return self._make_straight_decision(ego_speed, ego_heading, metadata)

    @staticmethod
    def _make_brake_decision(ego_speed: float, metadata: dict) -> AIDecision:
        trajectory = [
            TrajectoryPoint(
                position=np.array([ego_speed * t * 0.3, 0, 0], dtype=np.float32),
                timestamp=t,
            )
            for t in [0.0, 0.2, 0.5, 1.0]
        ]
        return AIDecision(
            decision_type="control",
            trajectory=trajectory,
            control={"steering": 0.0, "throttle": 0.0, "brake": 1.0},
            target_speed=0.0,
            metadata=metadata,
        )

    @staticmethod
    def _make_stop_decision(ego_speed: float, metadata: dict) -> AIDecision:
        trajectory = [
            TrajectoryPoint(
                position=np.array([ego_speed * t * 0.5, 0, 0], dtype=np.float32),
                timestamp=t,
            )
            for t in [0.0, 0.5, 1.0, 2.0]
        ]
        return AIDecision(
            decision_type="control",
            trajectory=trajectory,
            control={"steering": 0.0, "throttle": 0.0, "brake": 0.8},
            target_speed=0.0,
            metadata=metadata,
        )

    @staticmethod
    def _make_reverse_decision(metadata: dict) -> AIDecision:
        trajectory = [
            TrajectoryPoint(
                position=np.array([-1.0 * t, 0, 0], dtype=np.float32),
                timestamp=t,
            )
            for t in [0.0, 0.5, 1.0, 2.0]
        ]
        return AIDecision(
            decision_type="trajectory",
            trajectory=trajectory,
            target_speed=1.0,
            metadata=metadata,
        )

    @staticmethod
    def _make_turn_decision(
        angle_deg: float,
        sign: int,
        ego_speed: float,
        ego_heading: float,
        metadata: dict,
    ) -> AIDecision:
        angle_rad = math.radians(angle_deg) * sign
        speed = max(ego_speed, 2.0)
        turn_radius = 10.0
        arc_length = abs(angle_rad) * turn_radius
        duration = arc_length / speed if speed > 0.1 else 2.0

        trajectory = []
        steps = max(4, int(duration / 0.1))
        for i in range(steps):
            t = duration * i / (steps - 1) if steps > 1 else 0.0
            frac = i / (steps - 1) if steps > 1 else 0.0
            x = turn_radius * math.sin(angle_rad * frac)
            y = turn_radius * (1 - math.cos(angle_rad * frac)) * (-sign)
            trajectory.append(
                TrajectoryPoint(
                    position=np.array([x, y, 0], dtype=np.float32),
                    timestamp=float(t),
                )
            )

        metadata["turn_angle_deg"] = angle_deg * sign
        return AIDecision(
            decision_type="trajectory",
            trajectory=trajectory,
            target_speed=speed,
            metadata=metadata,
        )

    @staticmethod
    def _make_lane_change_decision(
        sign: int,
        ego_speed: float,
        ego_heading: float,
        metadata: dict,
    ) -> AIDecision:
        lane_width = 3.5
        speed = max(ego_speed, 2.0)
        duration = 3.0

        trajectory = []
        steps = 10
        for i in range(steps):
            t = duration * i / (steps - 1)
            frac = i / (steps - 1)
            x = speed * t
            # S-curve 横向移动
            y = sign * lane_width * (3 * frac**2 - 2 * frac**3)
            trajectory.append(
                TrajectoryPoint(
                    position=np.array([x, y, 0], dtype=np.float32),
                    timestamp=float(t),
                )
            )

        metadata["lane_change_direction"] = "right" if sign > 0 else "left"
        return AIDecision(
            decision_type="trajectory",
            trajectory=trajectory,
            target_speed=speed,
            metadata=metadata,
        )

    @staticmethod
    def _make_straight_decision(
        target_speed: float,
        ego_heading: float,
        metadata: dict,
    ) -> AIDecision:
        speed = max(target_speed, 0.0)
        trajectory = [
            TrajectoryPoint(
                position=np.array(
                    [
                        speed * t * math.cos(ego_heading),
                        speed * t * math.sin(ego_heading),
                        0,
                    ],
                    dtype=np.float32,
                ),
                timestamp=float(t),
            )
            for t in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
        ]
        return AIDecision(
            decision_type="trajectory",
            trajectory=trajectory,
            target_speed=speed,
            metadata=metadata,
        )
