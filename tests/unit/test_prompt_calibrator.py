"""Tests for the prompt calibrator module."""

from __future__ import annotations

from dataclasses import dataclass, field

from aylm.navigation_demo.prompt_calibrator import PromptCalibrator, PromptPatch


@dataclass(frozen=True)
class _FakeContext:
    """Minimal CalibrationContext stand-in for tests."""

    safety_hints: list[str] = field(default_factory=list)
    adjusted_threshold: float = 0.6
    adjusted_weights: dict[str, float] = field(default_factory=dict)
    violation_summary: str = ""
    frame_count: int = 0
    calibration_round: int = 0


class TestPromptCalibrator:
    """Tests for PromptCalibrator."""

    def test_empty_context_returns_empty_string(self):
        cal = PromptCalibrator()
        ctx = _FakeContext()
        assert cal.build_prompt_suffix(ctx) == ""

    def test_context_with_hints_returns_suffix(self):
        cal = PromptCalibrator()
        ctx = _FakeContext(
            safety_hints=["Avoid high speed near obstacles"],
            calibration_round=1,
            frame_count=10,
        )
        suffix = cal.build_prompt_suffix(ctx)
        assert "Safety Calibration Context" in suffix
        assert "Avoid high speed near obstacles" in suffix

    def test_violation_summary_included(self):
        cal = PromptCalibrator()
        ctx = _FakeContext(
            violation_summary="  no_collision: 4 violations (40%)",
            calibration_round=1,
            frame_count=10,
        )
        suffix = cal.build_prompt_suffix(ctx)
        assert "no_collision" in suffix
        assert "Recent violations" in suffix

    def test_score_context_included(self):
        cal = PromptCalibrator()
        ctx = _FakeContext(
            calibration_round=2,
            frame_count=20,
            adjusted_threshold=0.7,
            safety_hints=["Be careful"],
        )
        suffix = cal.build_prompt_suffix(ctx)
        assert "Frames analyzed: 20" in suffix
        assert "0.70" in suffix

    def test_max_hint_lines_truncation(self):
        cal = PromptCalibrator(max_hint_lines=3)
        hints = [f"Hint {i}" for i in range(10)]
        ctx = _FakeContext(
            safety_hints=hints,
            calibration_round=1,
        )
        suffix = cal.build_prompt_suffix(ctx)
        assert "Hint 0" in suffix
        assert "Hint 2" in suffix
        assert "Hint 9" not in suffix
        assert "7 more" in suffix

    def test_build_patches_returns_immutable(self):
        cal = PromptCalibrator()
        ctx = _FakeContext(
            safety_hints=["hint1"],
            violation_summary="summary",
            calibration_round=3,
            frame_count=30,
        )
        patches = cal.build_patches(ctx)
        assert len(patches) == 3
        for patch in patches:
            assert isinstance(patch, PromptPatch)
            assert patch.source_round == 3

    def test_build_patches_empty_context(self):
        cal = PromptCalibrator()
        ctx = _FakeContext()
        patches = cal.build_patches(ctx)
        assert patches == []

    def test_no_score_context_when_zero_frames(self):
        cal = PromptCalibrator()
        ctx = _FakeContext(
            safety_hints=["hint"],
            calibration_round=1,
            frame_count=0,
        )
        suffix = cal.build_prompt_suffix(ctx)
        assert "Frames analyzed" not in suffix
        assert "hint" in suffix
