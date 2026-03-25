"""宪法评估会话管理器。

有状态的评估会话，支持：
- 单帧评估（接收速度、障碍物、可选指令）
- 多帧时序评估（按时间顺序逐帧输入）
- 动态参数变更（运行时修改速度/航向）
- 评估历史与趋势分析

纯 Python 实现，不依赖 FastAPI / HTTP。

Example:
    >>> from aylm.api import ConstitutionSession
    >>> session = ConstitutionSession(ego_speed=10.0)
    >>>
    >>> # 单帧评估
    >>> result = session.evaluate(
    ...     obstacles=[{"center_robot": [5,0,0], "dimensions_robot": [1,1,1],
    ...                 "_label": "VEHICLE", "confidence": 0.9}],
    ... )
    >>> print(result["approved"], result["safety_score"])
    >>>
    >>> # 动态修改速度
    >>> session.update_ego(speed=15.0, heading=0.1)
    >>>
    >>> # 带指令评估
    >>> result = session.evaluate(
    ...     command={"type": "trajectory", "points": [[3,0,0,0.5]]},
    ...     obstacles=[...],
    ... )
    >>>
    >>> # 查看趋势
    >>> print(session.trend)  # "declining" / "stable" / "improving"
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    """单帧评估记录。"""

    frame_id: int
    timestamp: float
    safety_score: float
    approved: bool
    recommended_action: str
    violation_count: int
    ego_speed: float


class ConstitutionSession:
    """有状态的宪法评估会话。

    封装 CommandValidator，提供有状态的多帧评估、动态参数变更和趋势分析。

    Args:
        ego_speed: 初始自车速度 m/s
        ego_heading: 初始自车航向（弧度）
        approval_threshold: 安全分批准阈值（默认 0.6）
        config_path: 宪法配置文件路径（YAML/JSON），None 使用默认
        history_size: 保留的历史帧数（默认 100）
    """

    def __init__(
        self,
        ego_speed: float = 0.0,
        ego_heading: float = 0.0,
        approval_threshold: float = 0.6,
        config_path: str | None = None,
        history_size: int = 100,
    ):
        self._ego_speed = ego_speed
        self._ego_heading = ego_heading
        self._approval_threshold = approval_threshold
        self._frame_counter = 0
        self._start_time = time.monotonic()
        self._history: deque[FrameRecord] = deque(maxlen=history_size)

        # 延迟初始化 validator
        self._validator: Any = None
        self._config_path = config_path
        self._init_validator()

    def _init_validator(self) -> None:
        """延迟导入并初始化 CommandValidator。"""
        try:
            from aylm.constitution import CommandValidator

            config = None
            if self._config_path:
                from aylm.tools.constitution_integration import (
                    ConstitutionIntegration,
                )

                config = ConstitutionIntegration.load_config(self._config_path)

            self._validator = CommandValidator(
                config=config,
                approval_threshold=self._approval_threshold,
            )
            logger.info(
                "宪法评估会话已初始化 (speed=%.1f m/s, threshold=%.2f)",
                self._ego_speed,
                self._approval_threshold,
            )
        except Exception as e:
            logger.error("宪法评估会话初始化失败: %s", e)
            self._validator = None

    @property
    def is_available(self) -> bool:
        """评估器是否可用。"""
        return self._validator is not None

    @property
    def ego_speed(self) -> float:
        """当前自车速度 m/s。"""
        return self._ego_speed

    @property
    def ego_heading(self) -> float:
        """当前自车航向（弧度）。"""
        return self._ego_heading

    @property
    def frame_count(self) -> int:
        """已评估帧数。"""
        return self._frame_counter

    @property
    def history(self) -> list[FrameRecord]:
        """评估历史记录（最近 N 帧）。"""
        return list(self._history)

    @property
    def trend(self) -> str:
        """安全趋势分析。

        Returns:
            "improving" — 安全分上升趋势
            "declining" — 安全分下降趋势
            "stable"    — 安全分稳定
            "unknown"   — 数据不足
        """
        if len(self._history) < 3:
            return "unknown"

        recent = list(self._history)
        n = min(len(recent), 10)
        scores = [r.safety_score for r in recent[-n:]]

        # 计算简单线性趋势
        mid = n // 2
        first_half_avg = sum(scores[:mid]) / mid if mid > 0 else scores[0]
        second_half_avg = sum(scores[mid:]) / (n - mid) if (n - mid) > 0 else scores[-1]

        diff = second_half_avg - first_half_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"

    @property
    def summary(self) -> dict[str, Any]:
        """会话摘要统计。"""
        if not self._history:
            return {
                "frame_count": 0,
                "trend": "unknown",
                "avg_score": 0.0,
                "min_score": 0.0,
                "approval_rate": 0.0,
                "ego_speed": self._ego_speed,
            }

        scores = [r.safety_score for r in self._history]
        approved = sum(1 for r in self._history if r.approved)
        return {
            "frame_count": self._frame_counter,
            "trend": self.trend,
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "approval_rate": approved / len(self._history),
            "total_violations": sum(r.violation_count for r in self._history),
            "ego_speed": self._ego_speed,
        }

    def update_ego(
        self,
        speed: float | None = None,
        heading: float | None = None,
    ) -> None:
        """动态更新自车参数。

        Args:
            speed: 新的自车速度 m/s（None 保持不变）
            heading: 新的自车航向弧度（None 保持不变）
        """
        if speed is not None:
            self._ego_speed = float(speed)
        if heading is not None:
            self._ego_heading = float(heading)
        logger.debug(
            "自车参数已更新: speed=%.1f m/s, heading=%.2f rad",
            self._ego_speed,
            self._ego_heading,
        )

    def evaluate(
        self,
        obstacles: list[dict[str, Any]] | None = None,
        command: str | dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """评估单帧安全性。

        Args:
            obstacles: 障碍物列表（JSON 字典格式）
            command: 外部指令（字符串或字典），None 则评估默认直行
            timestamp: 时间戳（秒），None 则自动计算

        Returns:
            评估结果字典，包含:
            - approved: bool — 是否批准
            - safety_score: float — 安全分 0-1
            - recommended_action: str — 推荐动作
            - reason: str — 原因说明
            - violations: list — 违规详情
            - frame_id: int — 帧序号
            - timestamp: float — 时间戳
            - trend: str — 安全趋势
        """
        if not self.is_available:
            return {
                "approved": False,
                "safety_score": 0.0,
                "recommended_action": "error",
                "reason": "宪法评估器未初始化",
                "violations": [],
                "frame_id": self._frame_counter,
                "timestamp": 0.0,
                "trend": "unknown",
            }

        # 自动时间戳
        if timestamp is None:
            timestamp = time.monotonic() - self._start_time

        frame_id = self._frame_counter
        self._frame_counter += 1

        # 构建评估参数
        eval_kwargs: dict[str, Any] = {
            "ego_speed": self._ego_speed,
            "ego_heading": self._ego_heading,
            "obstacles": obstacles or [],
        }

        # 如果有指令，走 CommandValidator；否则评估默认直行
        if command is not None:
            result = self._validator.validate(command=command, **eval_kwargs)
        else:
            # 无指令时构建默认直行指令进行评估
            default_cmd = {
                "type": "trajectory",
                "points": [
                    [
                        self._ego_speed * t,
                        0.0,
                        0.0,
                        t,
                    ]
                    for t in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
                ],
            }
            result = self._validator.validate(command=default_cmd, **eval_kwargs)

        # 记录到历史
        record = FrameRecord(
            frame_id=frame_id,
            timestamp=timestamp,
            safety_score=result.safety_score,
            approved=result.approved,
            recommended_action=result.recommended_action,
            violation_count=len(result.violations),
            ego_speed=self._ego_speed,
        )
        self._history.append(record)

        # 构建返回结果
        output = result.to_dict()
        output["frame_id"] = frame_id
        output["timestamp"] = timestamp
        output["trend"] = self.trend

        return output

    def evaluate_batch(
        self,
        frames: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """批量评估多帧（按时序）。

        Args:
            frames: 帧列表，每帧包含:
                - obstacles: list[dict] — 障碍物
                - command: str | dict | None — 指令（可选）
                - timestamp: float | None — 时间戳（可选）
                - ego_speed: float | None — 该帧速度（可选，动态覆盖）
                - ego_heading: float | None — 该帧航向（可选，动态覆盖）

        Returns:
            评估结果列表
        """
        results = []
        for frame in frames:
            # 动态更新 ego 参数
            if "ego_speed" in frame:
                self.update_ego(speed=frame["ego_speed"])
            if "ego_heading" in frame:
                self.update_ego(heading=frame["ego_heading"])

            result = self.evaluate(
                obstacles=frame.get("obstacles"),
                command=frame.get("command"),
                timestamp=frame.get("timestamp"),
            )
            results.append(result)

        return results

    def reset(self) -> None:
        """重置会话状态（清空历史，保留配置）。"""
        self._frame_counter = 0
        self._start_time = time.monotonic()
        self._history.clear()
        logger.info("会话已重置")
