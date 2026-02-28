"""默认训练信号生成器。

提供基于违规结果的训练信号生成参考实现。
"""

import json
import logging
from typing import TYPE_CHECKING

from .registry import ConstitutionRegistry
from .scorer import SafetyScore
from .training import SignalType, TrainingSignal, TrainingSignalGenerator

if TYPE_CHECKING:
    from .types import AIDecision, SceneState

logger = logging.getLogger(__name__)


@ConstitutionRegistry.register_generator("default")
class DefaultTrainingSignalGenerator(TrainingSignalGenerator):
    """默认训练信号生成器。

    规则：
    - 存在带 correction_hint 的违规 → CORRECTION 信号
    - 存在违规但无 correction_hint → NEGATIVE 信号
    - 安全且 generate_positive=True → POSITIVE 信号
    - 安全且 generate_positive=False → None（不生成）
    """

    def __init__(self, generate_positive: bool = False):
        self.generate_positive = generate_positive

    def should_generate_positive(self) -> bool:
        return self.generate_positive

    def generate(
        self,
        safety_result: SafetyScore,
        scene_state: "SceneState",
        ai_decision: "AIDecision",
    ) -> TrainingSignal | None:
        """生成训练信号。"""
        has_violations = len(safety_result.violations) > 0
        has_correction = any(
            d.get("correction_hint") is not None
            for d in safety_result.violation_details
        )

        if has_violations:
            if has_correction:
                signal_type = SignalType.CORRECTION
                corrections = [
                    d["correction_hint"]
                    for d in safety_result.violation_details
                    if d.get("correction_hint") is not None
                ]
                correction_target = {"corrections": corrections}
            else:
                signal_type = SignalType.NEGATIVE
                correction_target = None
        elif self.generate_positive:
            signal_type = SignalType.POSITIVE
            correction_target = None
        else:
            return None

        return TrainingSignal(
            timestamp=scene_state.timestamp,
            frame_id=scene_state.frame_id,
            signal_type=signal_type,
            safety_score=safety_result.overall,
            violations=safety_result.violations,
            scene_context=scene_state.to_dict(),
            ai_decision=ai_decision.to_dict(),
            correction_target=correction_target,
        )

    def export(
        self,
        signals: list[TrainingSignal],
        output_path: str,
        format: str = "json",
    ) -> None:
        """导出训练信号为 JSON Lines 格式。"""
        with open(output_path, "w", encoding="utf-8") as f:
            for signal in signals:
                f.write(json.dumps(signal.to_dict(), ensure_ascii=False) + "\n")
        logger.info("导出 %d 条训练信号到 %s", len(signals), output_path)
