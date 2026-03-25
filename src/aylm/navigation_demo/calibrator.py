"""会话级自校准器。

累积帧级评估反馈，检测违规模式和安全趋势，
动态调整审批阈值、打分权重和 VLM 安全提示。
实现导航决策的在线自校准闭环。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 权重调整上下界（相对于基线的倍率）
_WEIGHT_MIN_RATIO = 0.3
_WEIGHT_MAX_RATIO = 2.0

# 阈值调整上下界
_THRESHOLD_MIN = 0.4
_THRESHOLD_MAX = 0.85

# 阈值调整步长
_THRESHOLD_TIGHTEN_STEP = 0.05
_THRESHOLD_RELAX_STEP = 0.03

# 权重增加比率（高频违规原则）
_WEIGHT_BOOST_RATIO = 0.15

# 违规频率触发阈值（占总帧数比例）
_VIOLATION_RATE_TRIGGER = 0.3

# 安全趋势的分差阈值
_TREND_THRESHOLD = 0.05

# 高批准率阈值（可放松时）
_HIGH_APPROVAL_RATE = 0.8

# 默认权重
DEFAULT_WEIGHTS: dict[str, float] = {
    "collision": 1.0,
    "ttc": 0.8,
    "boundary": 0.5,
}


@dataclass(frozen=True)
class CalibrationContext:
    """不可变的校准快照，反馈到下一帧提议循环。

    Attributes:
        safety_hints: 安全提示列表
        adjusted_threshold: 校准后的审批阈值
        adjusted_weights: 校准后的打分权重
        violation_summary: 人类可读的违规摘要
        frame_count: 已分析帧数
        calibration_round: 校准轮次
    """

    safety_hints: list[str] = field(default_factory=list)
    adjusted_threshold: float = 0.6
    adjusted_weights: dict[str, float] = field(default_factory=dict)
    violation_summary: str = ""
    frame_count: int = 0
    calibration_round: int = 0


@dataclass(frozen=True)
class FrameFeedback:
    """不可变的单帧评估反馈。

    Attributes:
        frame_index: 帧序号
        timestamp: 时间戳
        proposed_approved: 提议是否被批准
        proposal_score: 提议安全分
        executed_score: 执行后安全分
        violations: 违规原则名称列表
        signal_type: 训练信号类型
        correction_hints: 纠正提示列表
    """

    frame_index: int
    timestamp: float
    proposed_approved: bool
    proposal_score: float
    executed_score: float
    violations: list[str] = field(default_factory=list)
    signal_type: str | None = None
    correction_hints: list[str] = field(default_factory=list)


class SessionCalibrator:
    """会话级自校准器。

    累积帧反馈，在达到校准间隔或违规次数阈值时自动重校准。
    生成 CalibrationContext 供提议器和验证器使用。

    Args:
        base_threshold: 初始审批阈值
        base_weights: 初始打分权重
        calibration_interval: 每 N 帧触发一次校准
        violation_count_trigger: 累计 M 次违规触发校准
        learning_store: 可选的跨会话学习存储
    """

    def __init__(
        self,
        base_threshold: float = 0.6,
        base_weights: dict[str, float] | None = None,
        calibration_interval: int = 10,
        violation_count_trigger: int = 5,
        learning_store: Any = None,
    ):
        self._base_threshold = base_threshold
        self._base_weights = dict(base_weights or DEFAULT_WEIGHTS)
        self._calibration_interval = max(1, calibration_interval)
        self._violation_count_trigger = max(1, violation_count_trigger)
        self._learning_store = learning_store

        # 当前校准状态（内部可变，对外通过 get_context() 暴露不可变快照）
        self._current_threshold = base_threshold
        self._current_weights = dict(self._base_weights)
        self._current_hints: list[str] = []
        self._violation_summary = ""
        self._calibration_round = 0

        # 累积反馈
        self._feedbacks: list[FrameFeedback] = []
        self._violations_since_last_calibration = 0

    def apply_baseline(self, baseline: Any) -> None:
        """应用跨会话基线配置。

        Args:
            baseline: BaselineSnapshot 实例
        """
        self._current_threshold = baseline.threshold
        if baseline.weights:
            self._current_weights = dict(baseline.weights)
        if baseline.hints:
            self._current_hints = list(baseline.hints)
        logger.info(
            "已应用跨会话基线: threshold=%.2f, weights=%s",
            baseline.threshold,
            baseline.weights,
        )

    def record_frame(self, feedback: FrameFeedback) -> None:
        """记录单帧评估反馈。

        如果累计帧数或违规次数达到阈值，自动触发重校准。

        Args:
            feedback: 帧反馈
        """
        self._feedbacks.append(feedback)
        self._violations_since_last_calibration += len(feedback.violations)

        if self._should_recalibrate():
            self._recalibrate()

    def get_context(self) -> CalibrationContext:
        """返回当前校准上下文（不可变快照）。"""
        return CalibrationContext(
            safety_hints=list(self._current_hints),
            adjusted_threshold=self._current_threshold,
            adjusted_weights=dict(self._current_weights),
            violation_summary=self._violation_summary,
            frame_count=len(self._feedbacks),
            calibration_round=self._calibration_round,
        )

    def finalize(self) -> CalibrationContext:
        """会话结束时调用。执行最终校准并持久化到学习存储。

        Returns:
            最终校准上下文
        """
        # 如果有未校准的反馈，执行最终校准
        if self._feedbacks and self._violations_since_last_calibration > 0:
            self._recalibrate()

        # 持久化到学习存储
        if self._learning_store is not None and self._feedbacks:
            self._persist_to_store()

        return self.get_context()

    def _should_recalibrate(self) -> bool:
        """判断是否应触发重校准。"""
        frame_count = len(self._feedbacks)

        # 帧数间隔触发
        if frame_count > 0 and frame_count % self._calibration_interval == 0:
            return True

        # 违规次数触发
        return self._violations_since_last_calibration >= self._violation_count_trigger

    def _recalibrate(self) -> None:
        """执行重校准：分析违规模式、调整阈值和权重、生成安全提示。"""
        self._calibration_round += 1
        self._violations_since_last_calibration = 0

        violation_counts = self._analyze_violation_patterns()
        trend = self._compute_score_trend()
        approval_rate = self._compute_approval_rate()

        # 调整阈值
        self._current_threshold = self._adjust_threshold(trend, approval_rate)

        # 调整权重
        self._current_weights = self._adjust_weights(violation_counts)

        # 生成安全提示
        self._current_hints = self._generate_safety_hints(violation_counts)

        # 生成违规摘要
        self._violation_summary = self._generate_violation_summary(violation_counts)

        logger.info(
            "校准轮次 %d: trend=%s, threshold=%.2f, " "top_violations=%s",
            self._calibration_round,
            trend,
            self._current_threshold,
            dict(sorted(violation_counts.items(), key=lambda x: -x[1])[:3]),
        )

    def _analyze_violation_patterns(self) -> dict[str, int]:
        """统计各原则的违规次数。"""
        counts: dict[str, int] = {}
        for feedback in self._feedbacks:
            for violation in feedback.violations:
                counts[violation] = counts.get(violation, 0) + 1
        return counts

    def _compute_score_trend(self) -> str:
        """计算安全分趋势（与 ConstitutionSession.trend 逻辑一致）。"""
        if len(self._feedbacks) < 3:
            return "unknown"

        n = min(len(self._feedbacks), 10)
        recent = self._feedbacks[-n:]
        scores = [f.executed_score for f in recent]

        mid = n // 2
        first_avg = sum(scores[:mid]) / mid if mid > 0 else scores[0]
        second_avg = sum(scores[mid:]) / (n - mid) if (n - mid) > 0 else scores[-1]

        diff = second_avg - first_avg
        if diff > _TREND_THRESHOLD:
            return "improving"
        if diff < -_TREND_THRESHOLD:
            return "declining"
        return "stable"

    def _compute_approval_rate(self) -> float:
        """计算提议批准率。"""
        if not self._feedbacks:
            return 1.0
        approved = sum(1 for f in self._feedbacks if f.proposed_approved)
        return approved / len(self._feedbacks)

    def _adjust_threshold(self, trend: str, approval_rate: float) -> float:
        """根据趋势和批准率调整审批阈值。

        - 趋势下降 → 收紧（+0.05），提高安全要求
        - 趋势上升且高批准率 → 放松（-0.03），允许更多操作
        - 其他情况 → 保持不变
        """
        threshold = self._current_threshold

        if trend == "declining":
            threshold = min(threshold + _THRESHOLD_TIGHTEN_STEP, _THRESHOLD_MAX)
        elif trend == "improving" and approval_rate >= _HIGH_APPROVAL_RATE:
            threshold = max(threshold - _THRESHOLD_RELAX_STEP, _THRESHOLD_MIN)

        return round(threshold, 3)

    def _adjust_weights(self, violation_counts: dict[str, int]) -> dict[str, float]:
        """根据违规频率调整打分权重。

        高频违规原则 → 增加其类别权重（最高 2x 基线）
        从未违规原则 → 权重衰减回基线（-5%）
        """
        frame_count = len(self._feedbacks)
        if frame_count == 0:
            return dict(self._current_weights)

        # 原则名到类别的映射
        principle_to_category: dict[str, str] = {
            "no_collision": "collision",
            "safe_following": "collision",
            "ttc_safety": "ttc",
            "lane_compliance": "boundary",
            "speed_limit": "boundary",
        }

        # 统计每个类别的违规频率
        category_violation_rate: dict[str, float] = {}
        for principle, count in violation_counts.items():
            category = principle_to_category.get(principle, "boundary")
            rate = count / frame_count
            current = category_violation_rate.get(category, 0.0)
            category_violation_rate[category] = max(current, rate)

        new_weights = dict(self._current_weights)
        violated_categories = set(category_violation_rate.keys())

        for category in new_weights:
            base = self._base_weights.get(category, 0.5)
            current = new_weights[category]

            if category in violated_categories:
                rate = category_violation_rate[category]
                if rate >= _VIOLATION_RATE_TRIGGER:
                    # 高频违规：增加权重
                    boosted = current * (1.0 + _WEIGHT_BOOST_RATIO)
                    new_weights[category] = min(boosted, base * _WEIGHT_MAX_RATIO)
            else:
                # 从未违规：缓慢衰减回基线
                if current > base:
                    decayed = current * 0.95
                    new_weights[category] = max(decayed, base)

            # 确保不低于下界
            new_weights[category] = max(new_weights[category], base * _WEIGHT_MIN_RATIO)
            new_weights[category] = round(new_weights[category], 4)

        return new_weights

    def _generate_safety_hints(self, violation_counts: dict[str, int]) -> list[str]:
        """从违规模式和纠正提示生成安全提示。"""
        hints: list[str] = []

        # 从高频违规生成提示
        sorted_violations = sorted(violation_counts.items(), key=lambda x: -x[1])
        for principle, count in sorted_violations[:5]:
            hints.append(
                f"Principle '{principle}' violated {count} times — "
                f"increase caution for this constraint"
            )

        # 从纠正信号中提取提示
        correction_hints_seen: set[str] = set()
        for feedback in self._feedbacks:
            for hint in feedback.correction_hints:
                if hint not in correction_hints_seen:
                    correction_hints_seen.add(hint)
                    hints.append(hint)

        # 基于批准率的通用提示
        approval_rate = self._compute_approval_rate()
        if approval_rate < 0.5:
            hints.append(
                "More than half of proposed commands were rejected — "
                "prefer conservative, low-speed commands"
            )

        return hints

    def _generate_violation_summary(self, violation_counts: dict[str, int]) -> str:
        """生成人类可读的违规摘要。"""
        if not violation_counts:
            return ""

        sorted_violations = sorted(violation_counts.items(), key=lambda x: -x[1])
        lines = []
        for principle, count in sorted_violations[:5]:
            rate = count / len(self._feedbacks) * 100 if self._feedbacks else 0
            lines.append(f"  {principle}: {count} violations ({rate:.0f}%)")

        return "\n".join(lines)

    def _persist_to_store(self) -> None:
        """将会话结果持久化到学习存储。"""
        import datetime

        from .learning_store import SessionRecord

        scores = [f.executed_score for f in self._feedbacks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        approval_rate = self._compute_approval_rate()
        violation_counts = self._analyze_violation_patterns()

        session_id = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        record = SessionRecord(
            session_id=session_id,
            timestamp=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            frame_count=len(self._feedbacks),
            avg_score=round(avg_score, 4),
            approval_rate=round(approval_rate, 4),
            violation_pattern=violation_counts,
            final_threshold=self._current_threshold,
            final_weights=dict(self._current_weights),
            effective_hints=list(self._current_hints),
        )

        try:
            self._learning_store.save_session(record)
            logger.info(
                "会话记录已保存到学习存储: id=%s, avg_score=%.3f",
                session_id,
                avg_score,
            )
        except Exception as exc:
            logger.warning("学习存储保存失败: %s", exc)
