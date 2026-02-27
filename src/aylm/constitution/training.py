"""训练信号生成模块。

本模块定义了自监督学习的训练信号接口：
- TrainingSignal: 训练信号数据结构
- TrainingSignalGenerator: 训练信号生成器基类

这是实现"几何宪法式 AI 自循环自训练"的关键模块。
用户可以根据自己的 AI 训练框架实现具体的信号生成逻辑。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scorer import SafetyScore
    from .types import AIDecision, SceneState


class SignalType(Enum):
    """训练信号类型。"""

    POSITIVE = "positive"  # 正样本：安全决策
    NEGATIVE = "negative"  # 负样本：违规决策
    CORRECTION = "correction"  # 纠正样本：包含正确行为指导


@dataclass
class TrainingSignal:
    """训练信号。

    用于自监督学习的训练数据，可上传到云端用于 AI 模型改进。

    Attributes:
        timestamp: 时间戳
        frame_id: 帧 ID
        signal_type: 信号类型（正/负/纠正）
        safety_score: 安全分数
        violations: 违规的原则列表
        scene_context: 场景上下文（障碍物、自车状态等）
        ai_decision: AI 的原始决策
        correction_target: 纠正目标（如果是纠正样本）
        metadata: 额外元数据
    """

    timestamp: float
    frame_id: int
    signal_type: SignalType
    safety_score: float
    violations: list[str] = field(default_factory=list)
    scene_context: dict[str, Any] = field(default_factory=dict)
    ai_decision: dict[str, Any] = field(default_factory=dict)
    correction_target: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式（用于序列化）。"""
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "signal_type": self.signal_type.value,
            "safety_score": self.safety_score,
            "violations": self.violations,
            "scene_context": self.scene_context,
            "ai_decision": self.ai_decision,
            "correction_target": self.correction_target,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingSignal":
        """从字典创建实例。"""
        return cls(
            timestamp=data["timestamp"],
            frame_id=data["frame_id"],
            signal_type=SignalType(data["signal_type"]),
            safety_score=data["safety_score"],
            violations=data.get("violations", []),
            scene_context=data.get("scene_context", {}),
            ai_decision=data.get("ai_decision", {}),
            correction_target=data.get("correction_target"),
            metadata=data.get("metadata", {}),
        )


class TrainingSignalGenerator(ABC):
    """训练信号生成器基类。

    用户可以继承此类实现自定义的训练信号生成逻辑。
    不同的 AI 训练框架可能需要不同格式的训练数据。

    Example:
        >>> class MySignalGenerator(TrainingSignalGenerator):
        ...     def generate(self, safety_result, scene_state, ai_decision):
        ...         if not safety_result.is_safe:
        ...             return TrainingSignal(
        ...                 timestamp=scene_state.timestamp,
        ...                 frame_id=scene_state.frame_id,
        ...                 signal_type=SignalType.NEGATIVE,
        ...                 safety_score=safety_result.overall,
        ...                 violations=safety_result.violations,
        ...             )
        ...         return None
    """

    @abstractmethod
    def generate(
        self,
        safety_result: "SafetyScore",
        scene_state: "SceneState",
        ai_decision: "AIDecision",
    ) -> TrainingSignal | None:
        """生成训练信号。

        Args:
            safety_result: 安全评分结果
            scene_state: 当前场景状态
            ai_decision: AI 的决策

        Returns:
            TrainingSignal: 训练信号，如果不需要生成则返回 None
        """
        pass

    @abstractmethod
    def export(
        self,
        signals: list[TrainingSignal],
        output_path: str,
        format: str = "json",
    ) -> None:
        """导出训练信号。

        Args:
            signals: 训练信号列表
            output_path: 输出路径
            format: 导出格式（json, tfrecord, parquet 等）
        """
        pass

    def should_generate_positive(self) -> bool:
        """是否生成正样本（可覆盖）。

        默认不生成正样本，因为正样本数量通常远大于负样本。
        用户可以根据需要覆盖此方法。
        """
        return False

    def filter_signals(
        self,
        signals: list[TrainingSignal],
        min_score: float = 0.0,
        max_score: float = 1.0,
        signal_types: list[SignalType] | None = None,
    ) -> list[TrainingSignal]:
        """过滤训练信号。

        Args:
            signals: 原始信号列表
            min_score: 最小安全分数
            max_score: 最大安全分数
            signal_types: 允许的信号类型

        Returns:
            过滤后的信号列表
        """
        filtered = []
        for signal in signals:
            if not (min_score <= signal.safety_score <= max_score):
                continue
            if signal_types and signal.signal_type not in signal_types:
                continue
            filtered.append(signal)
        return filtered
