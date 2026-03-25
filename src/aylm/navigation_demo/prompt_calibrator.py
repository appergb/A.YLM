"""提示词校准器。

将违规模式和安全趋势转化为 VLM 提示词补丁，
注入到下一帧的导航指令提议中，引导模型生成更安全的决策。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptPatch:
    """不可变的提示词补丁。

    Attributes:
        section: 补丁分类 ("safety_hints" | "avoid_patterns" | "score_context")
        content: 注入文本
        source_round: 产生此补丁的校准轮次
    """

    section: str
    content: str
    source_round: int


class PromptCalibrator:
    """从校准上下文构建提示词后缀。

    将 CalibrationContext 中的违规模式和安全提示转化为自然语言，
    追加到 VLM 提示词末尾。对 heuristic 提供者无效但无害。

    Args:
        max_hint_lines: 最大安全提示行数（防止提示词过长）
    """

    def __init__(self, max_hint_lines: int = 8):
        self._max_hint_lines = max_hint_lines

    def build_prompt_suffix(self, context: Any) -> str:
        """从校准上下文构建提示词后缀。

        Args:
            context: CalibrationContext 实例

        Returns:
            提示词后缀字符串。无需校准时返回空字符串。
        """
        if context.calibration_round == 0 and not context.safety_hints:
            return ""

        sections: list[str] = []

        violation_text = self._format_violation_summary(context)
        if violation_text:
            sections.append(violation_text)

        hints_text = self._format_safety_hints(context.safety_hints)
        if hints_text:
            sections.append(hints_text)

        score_text = self._format_score_context(context)
        if score_text:
            sections.append(score_text)

        if not sections:
            return ""

        return "\n--- Safety Calibration Context ---\n" + "\n\n".join(sections)

    def build_patches(self, context: Any) -> list[PromptPatch]:
        """返回结构化补丁列表（用于检查/日志）。

        Args:
            context: CalibrationContext 实例

        Returns:
            PromptPatch 列表
        """
        patches: list[PromptPatch] = []
        calibration_round = context.calibration_round

        violation_text = self._format_violation_summary(context)
        if violation_text:
            patches.append(
                PromptPatch(
                    section="avoid_patterns",
                    content=violation_text,
                    source_round=calibration_round,
                )
            )

        hints_text = self._format_safety_hints(context.safety_hints)
        if hints_text:
            patches.append(
                PromptPatch(
                    section="safety_hints",
                    content=hints_text,
                    source_round=calibration_round,
                )
            )

        score_text = self._format_score_context(context)
        if score_text:
            patches.append(
                PromptPatch(
                    section="score_context",
                    content=score_text,
                    source_round=calibration_round,
                )
            )

        return patches

    def _format_violation_summary(self, context: Any) -> str:
        """格式化违规摘要段落。"""
        summary = context.violation_summary
        if not summary:
            return ""
        return f"Recent violations:\n{summary}"

    def _format_safety_hints(self, hints: list[str]) -> str:
        """格式化安全提示列表。"""
        if not hints:
            return ""
        truncated = hints[: self._max_hint_lines]
        lines = ["Safety hints based on recent experience:"]
        for hint in truncated:
            lines.append(f"- {hint}")
        if len(hints) > self._max_hint_lines:
            lines.append(f"  ... ({len(hints) - self._max_hint_lines} more)")
        return "\n".join(lines)

    def _format_score_context(self, context: Any) -> str:
        """格式化安全分趋势上下文。"""
        if context.frame_count == 0:
            return ""
        parts = [f"Frames analyzed: {context.frame_count}"]
        parts.append(f"Calibration round: {context.calibration_round}")
        parts.append(f"Current approval threshold: {context.adjusted_threshold:.2f}")
        return "Score context: " + ", ".join(parts)
