"""Tests for the session calibrator module."""

from __future__ import annotations

import pytest

from aylm.navigation_demo.calibrator import (
    CalibrationContext,
    FrameFeedback,
    SessionCalibrator,
)
from aylm.navigation_demo.learning_store import BaselineSnapshot


def _safe_feedback(
    frame_index: int = 0,
    score: float = 0.9,
) -> FrameFeedback:
    """Create a safe (no violations) feedback."""
    return FrameFeedback(
        frame_index=frame_index,
        timestamp=float(frame_index) * 0.1,
        proposed_approved=True,
        proposal_score=score,
        executed_score=score,
        violations=[],
    )


def _violation_feedback(
    frame_index: int = 0,
    score: float = 0.3,
    violations: list[str] | None = None,
    correction_hints: list[str] | None = None,
) -> FrameFeedback:
    """Create a feedback with violations."""
    return FrameFeedback(
        frame_index=frame_index,
        timestamp=float(frame_index) * 0.1,
        proposed_approved=False,
        proposal_score=score,
        executed_score=score,
        violations=violations or ["no_collision"],
        signal_type="negative",
        correction_hints=correction_hints or [],
    )


class TestCalibrationContext:
    """Tests for CalibrationContext immutability."""

    def test_frozen_dataclass(self):
        ctx = CalibrationContext()
        with pytest.raises(AttributeError):
            ctx.frame_count = 99  # type: ignore[misc]

    def test_defaults(self):
        ctx = CalibrationContext()
        assert ctx.adjusted_threshold == 0.6
        assert ctx.calibration_round == 0
        assert ctx.safety_hints == []
        assert ctx.frame_count == 0


class TestFrameFeedback:
    """Tests for FrameFeedback immutability."""

    def test_frozen_dataclass(self):
        fb = _safe_feedback()
        with pytest.raises(AttributeError):
            fb.frame_index = 99  # type: ignore[misc]


class TestSessionCalibrator:
    """Tests for SessionCalibrator core logic."""

    def test_initial_context_returns_defaults(self):
        cal = SessionCalibrator(base_threshold=0.6)
        ctx = cal.get_context()
        assert ctx.adjusted_threshold == 0.6
        assert ctx.calibration_round == 0
        assert ctx.frame_count == 0

    def test_records_frame_without_error(self):
        cal = SessionCalibrator()
        cal.record_frame(_safe_feedback())
        ctx = cal.get_context()
        assert ctx.frame_count == 1

    def test_recalibrates_after_interval(self):
        cal = SessionCalibrator(calibration_interval=5)
        for i in range(5):
            cal.record_frame(_safe_feedback(frame_index=i))
        ctx = cal.get_context()
        assert ctx.calibration_round == 1
        assert ctx.frame_count == 5

    def test_recalibrates_after_violation_trigger(self):
        cal = SessionCalibrator(
            calibration_interval=100,
            violation_count_trigger=3,
        )
        for i in range(3):
            cal.record_frame(_violation_feedback(frame_index=i))
        ctx = cal.get_context()
        assert ctx.calibration_round >= 1

    def test_tightens_threshold_on_declining_trend(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
        )
        # First 5 frames: good scores
        for i in range(5):
            cal.record_frame(_safe_feedback(frame_index=i, score=0.9))
        # Next 5 frames: declining scores
        for i in range(5, 10):
            cal.record_frame(_violation_feedback(frame_index=i, score=0.2))

        ctx = cal.get_context()
        assert ctx.adjusted_threshold > 0.6

    def test_relaxes_threshold_on_improving_trend(self):
        cal = SessionCalibrator(
            base_threshold=0.7,
            calibration_interval=10,
        )
        # First 5 frames: poor scores
        for i in range(5):
            cal.record_frame(_safe_feedback(frame_index=i, score=0.5))
        # Next 5 frames: improving scores
        for i in range(5, 10):
            cal.record_frame(_safe_feedback(frame_index=i, score=0.95))

        ctx = cal.get_context()
        assert ctx.adjusted_threshold < 0.7

    def test_threshold_bounded(self):
        cal = SessionCalibrator(
            base_threshold=0.8,
            calibration_interval=3,
        )
        # Many declining frames to push threshold up
        for i in range(30):
            cal.record_frame(_violation_feedback(frame_index=i, score=0.1))
        ctx = cal.get_context()
        assert ctx.adjusted_threshold <= 0.85

    def test_increases_weight_for_frequent_violations(self):
        cal = SessionCalibrator(
            calibration_interval=10,
            base_weights={"collision": 1.0, "ttc": 0.8, "boundary": 0.5},
        )
        # 7/10 frames violate collision → >30% rate
        for i in range(7):
            cal.record_frame(
                _violation_feedback(
                    frame_index=i,
                    violations=["no_collision"],
                )
            )
        for i in range(7, 10):
            cal.record_frame(_safe_feedback(frame_index=i))

        ctx = cal.get_context()
        assert ctx.adjusted_weights["collision"] > 1.0

    def test_weight_bounded(self):
        cal = SessionCalibrator(
            calibration_interval=3,
            base_weights={"collision": 1.0, "ttc": 0.8, "boundary": 0.5},
        )
        for i in range(30):
            cal.record_frame(
                _violation_feedback(
                    frame_index=i,
                    violations=["no_collision"],
                )
            )
        ctx = cal.get_context()
        assert ctx.adjusted_weights["collision"] <= 2.0

    def test_generates_safety_hints_from_violations(self):
        cal = SessionCalibrator(
            calibration_interval=5,
        )
        for i in range(5):
            cal.record_frame(
                _violation_feedback(
                    frame_index=i,
                    violations=["no_collision", "ttc_safety"],
                )
            )
        ctx = cal.get_context()
        assert len(ctx.safety_hints) > 0
        assert any("no_collision" in h for h in ctx.safety_hints)

    def test_generates_hints_from_correction_hints(self):
        cal = SessionCalibrator(
            calibration_interval=3,
        )
        for i in range(3):
            cal.record_frame(
                _violation_feedback(
                    frame_index=i,
                    correction_hints=["Reduce speed below 1.0 m/s"],
                )
            )
        ctx = cal.get_context()
        assert any("Reduce speed" in h for h in ctx.safety_hints)

    def test_violation_summary_generated(self):
        cal = SessionCalibrator(calibration_interval=3)
        for i in range(3):
            cal.record_frame(
                _violation_feedback(frame_index=i, violations=["no_collision"])
            )
        ctx = cal.get_context()
        assert "no_collision" in ctx.violation_summary

    def test_finalize_returns_context(self):
        cal = SessionCalibrator(calibration_interval=100)
        cal.record_frame(_violation_feedback())
        ctx = cal.finalize()
        assert isinstance(ctx, CalibrationContext)
        assert ctx.frame_count == 1

    def test_apply_baseline(self):
        cal = SessionCalibrator(base_threshold=0.6)
        baseline = BaselineSnapshot(
            threshold=0.7,
            weights={"collision": 1.2, "ttc": 0.9, "boundary": 0.6},
            hints=["loaded hint"],
            avg_score=0.85,
        )
        cal.apply_baseline(baseline)
        ctx = cal.get_context()
        assert ctx.adjusted_threshold == 0.7
        assert "loaded hint" in ctx.safety_hints

    def test_finalize_persists_to_learning_store(self, tmp_path):
        from aylm.navigation_demo.learning_store import LearningStore

        store = LearningStore(tmp_path / "store.json")
        cal = SessionCalibrator(
            calibration_interval=100,
            learning_store=store,
        )
        for i in range(5):
            cal.record_frame(_safe_feedback(frame_index=i, score=0.9))

        cal.finalize()
        assert store.get_session_count() == 1
        baseline = store.load_baseline()
        assert baseline is not None
        assert baseline.avg_score > 0.8

    def test_low_approval_rate_generates_conservative_hint(self):
        cal = SessionCalibrator(
            calibration_interval=10,
        )
        # All rejected
        for i in range(10):
            cal.record_frame(
                _violation_feedback(frame_index=i, violations=["no_collision"])
            )
        ctx = cal.get_context()
        assert any("rejected" in h.lower() for h in ctx.safety_hints)

    def test_multiple_calibration_rounds(self):
        cal = SessionCalibrator(calibration_interval=5)
        for i in range(15):
            cal.record_frame(_safe_feedback(frame_index=i))
        ctx = cal.get_context()
        assert ctx.calibration_round == 3
