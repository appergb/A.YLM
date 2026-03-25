"""Tests for the cross-session learning store module."""

from __future__ import annotations

from aylm.navigation_demo.learning_store import (
    BaselineSnapshot,
    LearningStore,
    SessionRecord,
)


def _make_record(
    session_id: str = "test-001",
    avg_score: float = 0.75,
    approval_rate: float = 0.8,
    final_threshold: float = 0.65,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        timestamp="2026-03-24T10:00:00Z",
        frame_count=20,
        avg_score=avg_score,
        approval_rate=approval_rate,
        violation_pattern={"no_collision": 3, "ttc_safety": 1},
        final_threshold=final_threshold,
        final_weights={"collision": 1.1, "ttc": 0.8, "boundary": 0.5},
        effective_hints=["Be careful near obstacles"],
    )


class TestSessionRecord:
    """Tests for SessionRecord serialization."""

    def test_round_trip(self):
        record = _make_record()
        data = record.to_dict()
        restored = SessionRecord.from_dict(data)
        assert restored.session_id == record.session_id
        assert restored.avg_score == record.avg_score
        assert restored.violation_pattern == record.violation_pattern


class TestBaselineSnapshot:
    """Tests for BaselineSnapshot serialization."""

    def test_round_trip(self):
        snapshot = BaselineSnapshot(
            threshold=0.65,
            weights={"collision": 1.1},
            hints=["hint"],
            avg_score=0.75,
        )
        data = snapshot.to_dict()
        restored = BaselineSnapshot.from_dict(data)
        assert restored.threshold == 0.65
        assert restored.avg_score == 0.75


class TestLearningStore:
    """Tests for LearningStore persistence."""

    def test_load_baseline_returns_none_when_empty(self, tmp_path):
        store = LearningStore(tmp_path / "store.json")
        assert store.load_baseline() is None

    def test_creates_file_on_first_save(self, tmp_path):
        store_path = tmp_path / "store.json"
        store = LearningStore(store_path)
        store.save_session(_make_record())
        assert store_path.exists()

    def test_saves_and_loads_session(self, tmp_path):
        store = LearningStore(tmp_path / "store.json")
        store.save_session(_make_record(avg_score=0.75))
        baseline = store.load_baseline()
        assert baseline is not None
        assert baseline.avg_score == 0.75
        assert baseline.threshold == 0.65

    def test_updates_baseline_on_better_session(self, tmp_path):
        store = LearningStore(tmp_path / "store.json")
        store.save_session(_make_record(avg_score=0.70, final_threshold=0.60))
        store.save_session(_make_record(avg_score=0.85, final_threshold=0.70))
        baseline = store.load_baseline()
        assert baseline is not None
        assert baseline.avg_score == 0.85
        assert baseline.threshold == 0.70

    def test_does_not_downgrade_baseline(self, tmp_path):
        store = LearningStore(tmp_path / "store.json")
        store.save_session(_make_record(avg_score=0.85, final_threshold=0.70))
        store.save_session(_make_record(avg_score=0.60, final_threshold=0.55))
        baseline = store.load_baseline()
        assert baseline is not None
        assert baseline.avg_score == 0.85

    def test_retains_max_sessions(self, tmp_path):
        store = LearningStore(tmp_path / "store.json", max_sessions=3)
        for i in range(5):
            store.save_session(
                _make_record(session_id=f"s-{i}", avg_score=0.5 + i * 0.05)
            )
        assert store.get_session_count() == 3

    def test_aggregates_violations(self, tmp_path):
        store = LearningStore(tmp_path / "store.json")
        store.save_session(_make_record())
        store.save_session(_make_record())
        violations = store.get_aggregated_violations()
        assert violations["no_collision"] == 6
        assert violations["ttc_safety"] == 2

    def test_handles_corrupt_file(self, tmp_path):
        store_path = tmp_path / "store.json"
        store_path.write_text("not valid json", encoding="utf-8")
        store = LearningStore(store_path)
        assert store.load_baseline() is None
        # Should still be able to save after corruption
        store.save_session(_make_record())
        assert store.load_baseline() is not None

    def test_handles_non_dict_json(self, tmp_path):
        store_path = tmp_path / "store.json"
        store_path.write_text("[1, 2, 3]", encoding="utf-8")
        store = LearningStore(store_path)
        assert store.load_baseline() is None

    def test_creates_parent_dirs(self, tmp_path):
        store_path = tmp_path / "deep" / "nested" / "store.json"
        store = LearningStore(store_path)
        store.save_session(_make_record())
        assert store_path.exists()
