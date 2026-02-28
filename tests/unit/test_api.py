"""API 模块测试 — ConstitutionSession + (可选) FastAPI 端点。"""

from __future__ import annotations

from aylm.api.session import ConstitutionSession, FrameRecord

# ── ConstitutionSession 基础 ─────────────────────────────


class TestConstitutionSession:
    """ConstitutionSession 核心功能测试。"""

    def test_init_default(self):
        session = ConstitutionSession(ego_speed=10.0)
        assert session.is_available
        assert session.ego_speed == 10.0
        assert session.ego_heading == 0.0
        assert session.frame_count == 0

    def test_init_with_heading(self):
        session = ConstitutionSession(ego_speed=5.0, ego_heading=0.5)
        assert session.ego_speed == 5.0
        assert session.ego_heading == 0.5

    def test_update_ego_speed(self):
        session = ConstitutionSession(ego_speed=10.0)
        session.update_ego(speed=20.0)
        assert session.ego_speed == 20.0
        assert session.ego_heading == 0.0  # 未改变

    def test_update_ego_heading(self):
        session = ConstitutionSession(ego_speed=10.0)
        session.update_ego(heading=1.5)
        assert session.ego_heading == 1.5
        assert session.ego_speed == 10.0  # 未改变

    def test_update_ego_both(self):
        session = ConstitutionSession(ego_speed=10.0)
        session.update_ego(speed=15.0, heading=0.3)
        assert session.ego_speed == 15.0
        assert session.ego_heading == 0.3


class TestConstitutionSessionEvaluate:
    """评估功能测试。"""

    def setup_method(self):
        self.session = ConstitutionSession(ego_speed=10.0)

    def test_evaluate_no_obstacles(self):
        result = self.session.evaluate(obstacles=[])
        assert "approved" in result
        assert "safety_score" in result
        assert "recommended_action" in result
        assert "frame_id" in result
        assert result["frame_id"] == 0
        assert result["approved"] is True

    def test_evaluate_increments_frame(self):
        self.session.evaluate(obstacles=[])
        self.session.evaluate(obstacles=[])
        assert self.session.frame_count == 2

    def test_evaluate_with_obstacles_safe(self):
        """远处障碍物应为安全。"""
        result = self.session.evaluate(
            obstacles=[
                {
                    "center_robot": [50.0, 5.0, 0.0],
                    "dimensions_robot": [2.0, 1.0, 1.5],
                    "_label": "VEHICLE",
                    "confidence": 0.9,
                }
            ],
        )
        assert result["approved"] is True
        assert result["safety_score"] > 0.5

    def test_evaluate_with_command_string(self):
        """自然语言指令评估。"""
        result = self.session.evaluate(
            command="保持当前速度",
            obstacles=[],
        )
        assert "approved" in result

    def test_evaluate_with_command_dict(self):
        """JSON 指令评估。"""
        result = self.session.evaluate(
            command={
                "type": "trajectory",
                "points": [[10.0, 0.0, 0.0, 1.0]],
            },
            obstacles=[],
        )
        assert "approved" in result

    def test_evaluate_collision_rejected(self):
        """近处障碍物应被否决或警告。"""
        result = self.session.evaluate(
            obstacles=[
                {
                    "center_robot": [1.0, 0.0, 0.0],
                    "dimensions_robot": [0.5, 0.5, 0.5],
                    "_label": "PERSON",
                    "confidence": 0.95,
                }
            ],
        )
        # 碰撞距离很近，不应该被批准
        assert result["approved"] is False or result["safety_score"] < 0.8

    def test_evaluate_unavailable(self):
        """评估器不可用时返回错误。"""
        session = ConstitutionSession(ego_speed=10.0)
        session._validator = None  # 模拟不可用
        result = session.evaluate(obstacles=[])
        assert result["approved"] is False
        assert result["recommended_action"] == "error"


class TestConstitutionSessionHistory:
    """历史记录与趋势测试。"""

    def setup_method(self):
        self.session = ConstitutionSession(ego_speed=10.0)

    def test_history_recorded(self):
        self.session.evaluate(obstacles=[])
        assert len(self.session.history) == 1
        record = self.session.history[0]
        assert isinstance(record, FrameRecord)
        assert record.frame_id == 0

    def test_trend_unknown_few_frames(self):
        self.session.evaluate(obstacles=[])
        assert self.session.trend == "unknown"

    def test_trend_stable(self):
        """多帧相同结果应为 stable。"""
        for _ in range(5):
            self.session.evaluate(obstacles=[])
        assert self.session.trend in ("stable", "unknown")

    def test_summary(self):
        self.session.evaluate(obstacles=[])
        s = self.session.summary
        assert "frame_count" in s
        assert "trend" in s
        assert "avg_score" in s
        assert "ego_speed" in s
        assert s["frame_count"] == 1

    def test_summary_empty(self):
        s = self.session.summary
        assert s["frame_count"] == 0
        assert s["trend"] == "unknown"


class TestConstitutionSessionBatch:
    """批量评估测试。"""

    def test_batch_basic(self):
        session = ConstitutionSession(ego_speed=10.0)
        frames = [
            {"obstacles": [], "timestamp": 0.0},
            {"obstacles": [], "timestamp": 1.0},
            {"obstacles": [], "timestamp": 2.0},
        ]
        results = session.evaluate_batch(frames)
        assert len(results) == 3
        assert session.frame_count == 3

    def test_batch_dynamic_speed(self):
        session = ConstitutionSession(ego_speed=10.0)
        frames = [
            {"obstacles": [], "ego_speed": 10.0, "timestamp": 0.0},
            {"obstacles": [], "ego_speed": 20.0, "timestamp": 1.0},
        ]
        results = session.evaluate_batch(frames)
        assert len(results) == 2
        # 最后的 ego_speed 应该是 20.0
        assert session.ego_speed == 20.0


class TestConstitutionSessionReset:
    """重置测试。"""

    def test_reset(self):
        session = ConstitutionSession(ego_speed=10.0)
        session.evaluate(obstacles=[])
        session.evaluate(obstacles=[])
        assert session.frame_count == 2

        session.reset()
        assert session.frame_count == 0
        assert len(session.history) == 0
        # 配置应保留
        assert session.ego_speed == 10.0
        assert session.is_available


class TestConstitutionSessionJsonSerializable:
    """确保结果可 JSON 序列化。"""

    def test_evaluate_result_serializable(self):
        import json

        session = ConstitutionSession(ego_speed=10.0)
        result = session.evaluate(
            obstacles=[
                {
                    "center_robot": [5.0, 0.0, 0.0],
                    "dimensions_robot": [1.0, 1.0, 1.5],
                    "_label": "VEHICLE",
                    "confidence": 0.9,
                }
            ],
        )
        # 不应抛出 TypeError
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)

    def test_summary_serializable(self):
        import json

        session = ConstitutionSession(ego_speed=10.0)
        session.evaluate(obstacles=[])
        serialized = json.dumps(session.summary, default=str)
        assert isinstance(serialized, str)
