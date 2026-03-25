"""跨会话学习存储。

持久化校准结果到 JSON 文件，使下一次运行从更优的基线启动。
每次运行结束时保存会话摘要，并自动选择历史最优配置作为基线。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STORE_VERSION = 1


@dataclass(frozen=True)
class SessionRecord:
    """不可变的单次会话摘要。

    Attributes:
        session_id: 唯一标识
        timestamp: ISO 格式时间戳
        frame_count: 处理帧数
        avg_score: 平均安全分
        approval_rate: 批准率
        violation_pattern: 各原则违规计数
        final_threshold: 会话结束时的阈值
        final_weights: 会话结束时的权重
        effective_hints: 有效的安全提示
    """

    session_id: str
    timestamp: str
    frame_count: int
    avg_score: float
    approval_rate: float
    violation_pattern: dict[str, int]
    final_threshold: float
    final_weights: dict[str, float]
    effective_hints: list[str]

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "frame_count": self.frame_count,
            "avg_score": self.avg_score,
            "approval_rate": self.approval_rate,
            "violation_pattern": dict(self.violation_pattern),
            "final_threshold": self.final_threshold,
            "final_weights": dict(self.final_weights),
            "effective_hints": list(self.effective_hints),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionRecord:
        """从字典反序列化。"""
        return cls(
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            frame_count=data["frame_count"],
            avg_score=data["avg_score"],
            approval_rate=data["approval_rate"],
            violation_pattern=data.get("violation_pattern", {}),
            final_threshold=data["final_threshold"],
            final_weights=data.get("final_weights", {}),
            effective_hints=data.get("effective_hints", []),
        )


@dataclass(frozen=True)
class BaselineSnapshot:
    """不可变的基线配置快照。

    Attributes:
        threshold: 推荐阈值
        weights: 推荐权重
        hints: 推荐安全提示
        avg_score: 产生此基线的会话平均分
    """

    threshold: float
    weights: dict[str, float]
    hints: list[str]
    avg_score: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "threshold": self.threshold,
            "weights": dict(self.weights),
            "hints": list(self.hints),
            "avg_score": self.avg_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineSnapshot:
        """从字典反序列化。"""
        return cls(
            threshold=data["threshold"],
            weights=data.get("weights", {}),
            hints=data.get("hints", []),
            avg_score=data["avg_score"],
        )


class LearningStore:
    """跨会话学习持久化存储。

    使用单个 JSON 文件存储历史会话记录和最优基线配置。
    每次会话结束时调用 save_session() 保存结果，
    下次运行时调用 load_baseline() 加载最优配置。

    Args:
        store_path: JSON 存储文件路径
        max_sessions: 保留的最大会话记录数
    """

    def __init__(
        self,
        store_path: Path,
        max_sessions: int = 20,
    ):
        self._store_path = Path(store_path)
        self._max_sessions = max_sessions

    def load_baseline(self) -> BaselineSnapshot | None:
        """加载历史最优基线配置。

        Returns:
            BaselineSnapshot 或 None（首次运行）
        """
        data = self._read_store()
        if data is None:
            return None

        baseline_data = data.get("current_baseline")
        if not baseline_data:
            return None

        try:
            return BaselineSnapshot.from_dict(baseline_data)
        except (KeyError, TypeError) as exc:
            logger.warning("基线数据格式异常，忽略: %s", exc)
            return None

    def save_session(self, record: SessionRecord) -> None:
        """保存会话记录，并在更优时更新基线。

        Args:
            record: 会话摘要记录
        """
        data = self._read_store() or self._empty_store()

        # 追加会话记录
        sessions = data.get("sessions", [])
        sessions.append(record.to_dict())
        if len(sessions) > self._max_sessions:
            sessions = sessions[-self._max_sessions :]
        data["sessions"] = sessions

        # 更新聚合违规
        aggregated = data.get("aggregated_violations", {})
        for principle, count in record.violation_pattern.items():
            aggregated[principle] = aggregated.get(principle, 0) + count
        data["aggregated_violations"] = aggregated

        # 如果本次更优，更新基线
        current_baseline = data.get("current_baseline")
        should_update = current_baseline is None or (
            record.avg_score > current_baseline.get("avg_score", 0.0)
        )
        if should_update:
            data["current_baseline"] = BaselineSnapshot(
                threshold=record.final_threshold,
                weights=record.final_weights,
                hints=record.effective_hints,
                avg_score=record.avg_score,
            ).to_dict()
            logger.info(
                "学习存储基线已更新: avg_score=%.3f, threshold=%.2f",
                record.avg_score,
                record.final_threshold,
            )

        self._write_store(data)

    def get_aggregated_violations(self) -> dict[str, int]:
        """返回跨会话聚合违规计数。"""
        data = self._read_store()
        if data is None:
            return {}
        return dict(data.get("aggregated_violations", {}))

    def get_session_count(self) -> int:
        """返回已存储的会话数。"""
        data = self._read_store()
        if data is None:
            return 0
        return len(data.get("sessions", []))

    def _read_store(self) -> dict[str, Any] | None:
        """读取存储文件，损坏时返回 None。"""
        if not self._store_path.exists():
            return None
        try:
            text = self._store_path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict):
                logger.warning("学习存储文件格式异常，重置")
                return None
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("学习存储文件读取失败，重置: %s", exc)
            return None

    def _write_store(self, data: dict[str, Any]) -> None:
        """写入存储文件。"""
        data["version"] = _STORE_VERSION
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._store_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(self._store_path)
        except OSError as exc:
            logger.error("学习存储写入失败: %s", exc)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _empty_store() -> dict[str, Any]:
        """创建空存储结构。"""
        return {
            "version": _STORE_VERSION,
            "sessions": [],
            "current_baseline": None,
            "aggregated_violations": {},
        }
