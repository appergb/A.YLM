"""宪法模块全面单元测试。"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aylm.constitution.adapter import ConstitutionObstacle, ObstacleMotion
from aylm.constitution.base import ConstitutionPrinciple, Severity, ViolationResult
from aylm.constitution.command_parser import (
    JSONCommandParser,
    NaturalLanguageParser,
)
from aylm.constitution.config import ConstitutionConfig, PrincipleConfig
from aylm.constitution.default_generator import DefaultTrainingSignalGenerator
from aylm.constitution.evaluator import ConstitutionEvaluator, EvaluationResult
from aylm.constitution.principles.collision import NoCollisionPrinciple
from aylm.constitution.principles.following import SafeFollowingPrinciple
from aylm.constitution.principles.lane import LaneCompliancePrinciple
from aylm.constitution.principles.speed import SpeedLimitPrinciple
from aylm.constitution.principles.ttc import TTCSafetyPrinciple
from aylm.constitution.registry import ConstitutionRegistry
from aylm.constitution.scorer import RecommendedAction, SafetyScore
from aylm.constitution.training import SignalType, TrainingSignal
from aylm.constitution.types import (
    AIDecision,
    EgoState,
    SceneState,
    TrajectoryPoint,
)
from aylm.constitution.validator import CommandValidator, ValidationResult
from aylm.constitution.weighted_scorer import WeightedSafetyScorer
from aylm.tools.json_utils import numpy_safe_dump, numpy_safe_dumps

# ──────────────────────────── helpers ────────────────────────────


def _ego(
    speed: float = 10.0,
    position: tuple = (0, 0, 0),
    heading: float = 0.0,
) -> EgoState:
    return EgoState(
        position=np.array(position, dtype=np.float32),
        velocity=np.array([speed, 0, 0], dtype=np.float32),
        heading=heading,
        speed=speed,
    )


def _obstacle(
    center: tuple = (10, 0, 0),
    dims: tuple = (2, 1, 1),
    speed: float = 0.0,
    stationary: bool = True,
    label: str = "VEHICLE",
):
    """创建一个用于宪法评估的简单障碍物。"""
    motion = None
    if not stationary:
        motion = ObstacleMotion(
            velocity_robot=np.array([speed, 0, 0], dtype=np.float32),
            speed=abs(speed),
            heading=0.0,
            is_stationary=False,
        )
    return ConstitutionObstacle(
        center_robot=np.array(center, dtype=np.float32),
        dimensions=np.array(dims, dtype=np.float32),
        label=label,
        confidence=0.9,
        track_id=1,
        motion=motion,
    )


def _scene(
    obstacles=None,
    ego_speed: float = 10.0,
    lane_boundaries=None,
) -> SceneState:
    return SceneState(
        frame_id=0,
        timestamp=1.0,
        ego_state=_ego(speed=ego_speed),
        obstacles=obstacles or [],
        lane_boundaries=lane_boundaries,
    )


def _decision(
    trajectory_points=None,
    target_speed: float | None = None,
) -> AIDecision:
    if trajectory_points is None:
        trajectory_points = [
            TrajectoryPoint(
                position=np.array([i * 2.0, 0, 0], dtype=np.float32),
                timestamp=float(i) * 0.5,
            )
            for i in range(5)
        ]
    return AIDecision(
        decision_type="trajectory",
        trajectory=trajectory_points,
        target_speed=target_speed,
    )


# ──────────────────────────── Tests ────────────────────────────


class TestConstitutionObstacle:
    """适配器层测试。"""

    def test_from_obstacle_dict_basic(self):
        data = {
            "center_robot": [5.0, -1.0, 0.5],
            "dimensions_robot": [2.0, 1.0, 1.5],
            "_label": "PERSON",
            "confidence": 0.85,
            "track_id": 3,
        }
        obs = ConstitutionObstacle.from_obstacle_dict(data)
        np.testing.assert_allclose(obs.center_robot, [5.0, -1.0, 0.5], atol=1e-5)
        assert obs.label == "PERSON"
        assert obs.confidence == 0.85
        assert obs.track_id == 3
        assert obs.motion is None

    def test_from_obstacle_dict_with_motion(self):
        data = {
            "center_robot": [10.0, 0, 0],
            "dimensions_robot": [4, 2, 1.5],
            "category": "车辆",
            "confidence": 0.9,
            "motion": {
                "velocity_robot": [5.0, 0, 0],
                "speed": 5.0,
                "heading": 0.0,
                "is_stationary": False,
            },
        }
        obs = ConstitutionObstacle.from_obstacle_dict(data)
        assert obs.motion is not None
        assert obs.motion.speed == 5.0
        assert not obs.motion.is_stationary

    def test_to_dict_roundtrip(self):
        obs = _obstacle(center=(5, 1, 0), speed=3.0, stationary=False)
        d = obs.to_dict()
        assert "center_robot" in d
        assert "motion" in d
        assert d["motion"]["speed"] == 3.0

    def test_no_motion_in_dict(self):
        obs = _obstacle(center=(5, 0, 0), stationary=True)
        d = obs.to_dict()
        assert "motion" not in d


class TestConstitutionRegistry:
    """注册/查找/清除测试。"""

    def setup_method(self):
        self._saved_p = dict(ConstitutionRegistry._principles)
        self._saved_s = dict(ConstitutionRegistry._scorers)
        self._saved_g = dict(ConstitutionRegistry._generators)
        self._saved_cp = dict(ConstitutionRegistry._command_parsers)

    def teardown_method(self):
        ConstitutionRegistry._principles = self._saved_p
        ConstitutionRegistry._scorers = self._saved_s
        ConstitutionRegistry._generators = self._saved_g
        ConstitutionRegistry._command_parsers = self._saved_cp

    def test_register_and_get_principle(self):
        @ConstitutionRegistry.register_principle("test_principle")
        class TestPrinciple(ConstitutionPrinciple):
            @property
            def name(self):
                return "test_principle"

            @property
            def severity(self):
                return Severity.LOW

            def evaluate(self, state, decision):
                return ViolationResult(
                    violated=False,
                    severity=self.severity,
                    confidence=1.0,
                    description="test",
                )

        assert ConstitutionRegistry.get_principle("test_principle") is TestPrinciple
        assert "test_principle" in ConstitutionRegistry.list_principles()

    def test_create_principle(self):
        p = ConstitutionRegistry.create_principle("no_collision")
        assert isinstance(p, NoCollisionPrinciple)

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError):
            ConstitutionRegistry.create_principle("nonexistent")

    def test_clear(self):
        ConstitutionRegistry.clear()
        assert len(ConstitutionRegistry.list_principles()) == 0
        assert len(ConstitutionRegistry.list_scorers()) == 0
        assert len(ConstitutionRegistry.list_generators()) == 0


class TestAutoRegistration:
    """导入即注册验证。"""

    def test_builtin_principles_registered(self):
        import aylm.constitution  # noqa: F401

        names = ConstitutionRegistry.list_principles()
        assert "no_collision" in names
        assert "ttc_safety" in names
        assert "safe_following" in names
        assert "lane_compliance" in names
        assert "speed_limit" in names

    def test_builtin_scorer_registered(self):
        import aylm.constitution  # noqa: F401

        assert "weighted" in ConstitutionRegistry.list_scorers()

    def test_builtin_generator_registered(self):
        import aylm.constitution  # noqa: F401

        assert "default" in ConstitutionRegistry.list_generators()


class TestNoCollisionPrinciple:
    """碰撞检测原则测试。"""

    def test_no_obstacles_safe(self):
        p = NoCollisionPrinciple()
        result = p.evaluate(_scene(), _decision())
        assert not result.violated

    def test_no_trajectory_safe(self):
        p = NoCollisionPrinciple()
        result = p.evaluate(
            _scene(obstacles=[_obstacle()]),
            AIDecision(decision_type="trajectory"),
        )
        assert not result.violated

    def test_collision_detected(self):
        p = NoCollisionPrinciple(safety_margin=1.0)
        obs = _obstacle(center=(1.0, 0, 0), dims=(0.5, 0.5, 0.5))
        traj = [
            TrajectoryPoint(
                position=np.array([1.0, 0, 0], dtype=np.float32),
                timestamp=0.5,
            )
        ]
        result = p.evaluate(
            _scene(obstacles=[obs]),
            _decision(trajectory_points=traj),
        )
        assert result.violated
        assert result.severity == Severity.CRITICAL

    def test_safe_distance(self):
        p = NoCollisionPrinciple(safety_margin=0.5)
        obs = _obstacle(center=(20.0, 0, 0), dims=(1, 1, 1))
        result = p.evaluate(_scene(obstacles=[obs]), _decision())
        assert not result.violated


class TestTTCSafetyPrinciple:
    """TTC 原则测试。"""

    def test_no_obstacles_safe(self):
        p = TTCSafetyPrinciple()
        result = p.evaluate(_scene(), _decision())
        assert not result.violated

    def test_stationary_obstacles_ignored(self):
        p = TTCSafetyPrinciple()
        obs = _obstacle(center=(5, 0, 0), stationary=True)
        result = p.evaluate(_scene(obstacles=[obs]), _decision())
        assert not result.violated

    def test_approaching_critical(self):
        p = TTCSafetyPrinciple(
            warning_threshold=3.0,
            critical_threshold=1.5,
            min_safe_distance=1.0,
        )
        obs = _obstacle(center=(5, 0, 0), speed=-8.0, stationary=False)
        result = p.evaluate(
            _scene(obstacles=[obs], ego_speed=10.0),
            _decision(),
        )
        # 高速接近时 TTC 应该很小
        assert result.violated


class TestSafeFollowingPrinciple:
    """跟车距离原则测试。"""

    def test_no_obstacles_safe(self):
        p = SafeFollowingPrinciple()
        result = p.evaluate(_scene(), _decision())
        assert not result.violated

    def test_sufficient_distance(self):
        p = SafeFollowingPrinciple(time_gap=2.0, min_gap=2.0)
        obs = _obstacle(center=(50, 0, 0), speed=5.0, stationary=False)
        result = p.evaluate(_scene(obstacles=[obs], ego_speed=10.0), _decision())
        assert not result.violated

    def test_insufficient_distance(self):
        p = SafeFollowingPrinciple(time_gap=2.0, min_gap=2.0)
        # required = 2.0 * 10 + 2.0 = 22m, obstacle at 5m
        obs = _obstacle(center=(5, 0, 0), speed=5.0, stationary=False)
        result = p.evaluate(_scene(obstacles=[obs], ego_speed=10.0), _decision())
        assert result.violated
        assert result.severity == Severity.HIGH

    def test_obstacle_behind_ignored(self):
        p = SafeFollowingPrinciple()
        obs = _obstacle(center=(-5, 0, 0), speed=5.0, stationary=False)
        result = p.evaluate(_scene(obstacles=[obs], ego_speed=10.0), _decision())
        assert not result.violated


class TestLaneCompliancePrinciple:
    """车道合规原则测试。"""

    def test_no_lane_data_returns_low_confidence(self):
        p = LaneCompliancePrinciple()
        result = p.evaluate(
            _scene(lane_boundaries=None),
            _decision(),
        )
        assert not result.violated
        assert result.confidence == 0.3

    def test_within_lane(self):
        p = LaneCompliancePrinciple(lane_width=3.5, max_offset_ratio=0.9)
        traj = [
            TrajectoryPoint(
                position=np.array([i * 2.0, 0.5, 0], dtype=np.float32),
                timestamp=float(i) * 0.5,
            )
            for i in range(5)
        ]
        result = p.evaluate(
            _scene(lane_boundaries=[[0, 0], [100, 0]]),
            _decision(trajectory_points=traj),
        )
        assert not result.violated

    def test_outside_lane(self):
        p = LaneCompliancePrinciple(lane_width=3.5, max_offset_ratio=0.9)
        # max_offset = 3.5/2 * 0.9 = 1.575
        traj = [
            TrajectoryPoint(
                position=np.array([5.0, 2.0, 0], dtype=np.float32),
                timestamp=0.5,
            )
        ]
        result = p.evaluate(
            _scene(lane_boundaries=[[0, 0], [100, 0]]),
            _decision(trajectory_points=traj),
        )
        assert result.violated

    def test_no_trajectory(self):
        p = LaneCompliancePrinciple()
        result = p.evaluate(
            _scene(lane_boundaries=[[0, 0]]),
            AIDecision(decision_type="trajectory"),
        )
        assert not result.violated


class TestSpeedLimitPrinciple:
    """速度限制原则测试。"""

    def test_within_limit_safe(self):
        p = SpeedLimitPrinciple(speed_limit=15.0)
        result = p.evaluate(_scene(ego_speed=10.0), _decision())
        assert not result.violated

    def test_over_limit_critical(self):
        p = SpeedLimitPrinciple(speed_limit=10.0)
        result = p.evaluate(_scene(ego_speed=15.0), _decision())
        assert result.violated
        assert result.severity == Severity.CRITICAL

    def test_target_speed_over_limit(self):
        p = SpeedLimitPrinciple(speed_limit=10.0)
        result = p.evaluate(
            _scene(ego_speed=8.0),
            _decision(target_speed=15.0),
        )
        assert result.violated
        assert result.severity == Severity.HIGH

    def test_approaching_limit_warning(self):
        p = SpeedLimitPrinciple(speed_limit=10.0, warning_ratio=0.9)
        # 9.5 > 10 * 0.9 = 9.0
        result = p.evaluate(_scene(ego_speed=9.5), _decision())
        assert result.violated
        assert result.severity == Severity.MEDIUM


class TestWeightedSafetyScorer:
    """打分器测试。"""

    def test_no_violations_full_score(self):
        scorer = WeightedSafetyScorer()
        violations = [
            (
                "no_collision",
                ViolationResult(
                    violated=False,
                    severity=Severity.CRITICAL,
                    confidence=1.0,
                    description="safe",
                ),
            ),
        ]
        score = scorer.score_violations(violations)
        assert score.overall == 1.0
        assert score.recommended_action == RecommendedAction.SAFE

    def test_critical_violation_emergency_stop(self):
        scorer = WeightedSafetyScorer()
        violations = [
            (
                "no_collision",
                ViolationResult(
                    violated=True,
                    severity=Severity.CRITICAL,
                    confidence=0.9,
                    description="collision!",
                ),
            ),
        ]
        score = scorer.score_violations(violations)
        assert score.overall < 1.0
        assert score.recommended_action == RecommendedAction.EMERGENCY_STOP
        assert "no_collision" in score.violations

    def test_high_violation_intervention(self):
        scorer = WeightedSafetyScorer()
        violations = [
            (
                "ttc_safety",
                ViolationResult(
                    violated=True,
                    severity=Severity.HIGH,
                    confidence=0.85,
                    description="ttc warning",
                ),
            ),
        ]
        score = scorer.score_violations(violations)
        assert score.recommended_action == RecommendedAction.INTERVENTION

    def test_category_mapping(self):
        scorer = WeightedSafetyScorer()
        violations = [
            (
                "lane_compliance",
                ViolationResult(
                    violated=True,
                    severity=Severity.MEDIUM,
                    confidence=0.8,
                    description="lane violation",
                ),
            ),
        ]
        score = scorer.score_violations(violations)
        assert score.boundary_score < 1.0
        assert score.collision_score == 1.0
        assert score.ttc_score == 1.0

    def test_empty_violations(self):
        scorer = WeightedSafetyScorer()
        score = scorer.score_violations([])
        assert score.overall == 1.0
        assert score.confidence == 0.5


class TestDefaultTrainingSignalGenerator:
    """训练信号生成器测试。"""

    def test_no_violation_no_signal(self):
        gen = DefaultTrainingSignalGenerator(generate_positive=False)
        score = SafetyScore(overall=1.0, violations=[], violation_details=[])
        signal = gen.generate(score, _scene(), _decision())
        assert signal is None

    def test_no_violation_positive_signal(self):
        gen = DefaultTrainingSignalGenerator(generate_positive=True)
        score = SafetyScore(overall=1.0, violations=[], violation_details=[])
        signal = gen.generate(score, _scene(), _decision())
        assert signal is not None
        assert signal.signal_type == SignalType.POSITIVE

    def test_violation_negative_signal(self):
        gen = DefaultTrainingSignalGenerator()
        score = SafetyScore(
            overall=0.5,
            violations=["no_collision"],
            violation_details=[
                {
                    "violated": True,
                    "severity": "CRITICAL",
                    "confidence": 0.9,
                    "description": "collision",
                    "correction_hint": None,
                }
            ],
        )
        signal = gen.generate(score, _scene(), _decision())
        assert signal is not None
        assert signal.signal_type == SignalType.NEGATIVE

    def test_violation_with_correction(self):
        gen = DefaultTrainingSignalGenerator()
        score = SafetyScore(
            overall=0.3,
            violations=["lane_compliance"],
            violation_details=[
                {
                    "violated": True,
                    "severity": "MEDIUM",
                    "confidence": 0.8,
                    "description": "lane deviation",
                    "correction_hint": {"action": "return_to_center"},
                }
            ],
        )
        signal = gen.generate(score, _scene(), _decision())
        assert signal is not None
        assert signal.signal_type == SignalType.CORRECTION
        assert signal.correction_target is not None

    def test_export_json_lines(self):
        gen = DefaultTrainingSignalGenerator()
        signals = [
            TrainingSignal(
                timestamp=1.0,
                frame_id=0,
                signal_type=SignalType.NEGATIVE,
                safety_score=0.5,
                violations=["test"],
            ),
            TrainingSignal(
                timestamp=2.0,
                frame_id=1,
                signal_type=SignalType.POSITIVE,
                safety_score=1.0,
            ),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        gen.export(signals, path)

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["signal_type"] == "negative"
        Path(path).unlink()


class TestConstitutionEvaluator:
    """端到端编排测试。"""

    def test_default_config_evaluation(self):
        evaluator = ConstitutionEvaluator()
        scene = _scene(
            obstacles=[_obstacle(center=(50, 0, 0), stationary=True)],
            ego_speed=5.0,
        )
        result = evaluator.evaluate(scene, _decision())
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.safety_score, SafetyScore)
        assert result.safety_score.overall > 0

    def test_collision_scenario(self):
        evaluator = ConstitutionEvaluator()
        obs = _obstacle(center=(1.0, 0, 0), dims=(0.5, 0.5, 0.5))
        traj = [
            TrajectoryPoint(
                position=np.array([1.0, 0, 0], dtype=np.float32),
                timestamp=0.5,
            )
        ]
        scene = _scene(obstacles=[obs], ego_speed=10.0)
        result = evaluator.evaluate(scene, _decision(trajectory_points=traj))
        assert result.safety_score.overall < 1.0
        assert len(result.violations) > 0

    def test_safe_scenario(self):
        evaluator = ConstitutionEvaluator()
        scene = _scene(ego_speed=5.0)
        result = evaluator.evaluate(scene, _decision())
        assert result.safety_score.overall == 1.0

    def test_active_principles(self):
        evaluator = ConstitutionEvaluator()
        names = evaluator.active_principles
        assert "no_collision" in names
        assert "ttc_safety" in names

    def test_custom_config(self):
        config = ConstitutionConfig(
            principles=[
                PrincipleConfig(name="no_collision", severity="critical"),
                PrincipleConfig(name="speed_limit", severity="medium"),
            ],
        )
        evaluator = ConstitutionEvaluator(config=config)
        assert len(evaluator.active_principles) == 2
        assert "no_collision" in evaluator.active_principles
        assert "speed_limit" in evaluator.active_principles

    def test_disabled_principle_skipped(self):
        config = ConstitutionConfig(
            principles=[
                PrincipleConfig(
                    name="no_collision", severity="critical", enabled=False
                ),
                PrincipleConfig(name="speed_limit", severity="medium"),
            ],
        )
        evaluator = ConstitutionEvaluator(config=config)
        assert "no_collision" not in evaluator.active_principles

    def test_to_dict(self):
        evaluator = ConstitutionEvaluator()
        scene = _scene(ego_speed=5.0)
        result = evaluator.evaluate(scene, _decision())
        d = result.to_dict()
        assert "safety_score" in d
        assert "violations" in d
        assert "principle_results" in d

    def test_training_signal_on_violation(self):
        config = ConstitutionConfig(generate_positive_signals=True)
        evaluator = ConstitutionEvaluator(config=config)
        scene = _scene(ego_speed=5.0)
        result = evaluator.evaluate(scene, _decision())
        # 安全场景 + generate_positive=True → 应生成正训练信号
        assert result.training_signal is not None
        assert result.training_signal.signal_type == SignalType.POSITIVE


class TestConstitutionConfig:
    """配置测试。"""

    def test_default_principles(self):
        config = ConstitutionConfig()
        assert len(config.principles) == 5
        names = [p.name for p in config.principles]
        assert "no_collision" in names
        assert "safe_following" in names
        assert "speed_limit" in names

    def test_from_dict(self):
        data = {
            "principles": [
                {"name": "no_collision", "severity": "critical"},
                {
                    "name": "ttc_safety",
                    "severity": "high",
                    "params": {"warning_threshold": 4.0},
                },
            ],
            "thresholds": {"min_safe_distance": 3.0},
            "weights": {"collision": 1.0, "ttc": 0.9, "boundary": 0.6},
        }
        config = ConstitutionConfig.from_dict(data)
        assert len(config.principles) == 2
        assert config.min_safe_distance == 3.0
        assert config.ttc_weight == 0.9

    def test_to_dict_roundtrip(self):
        config = ConstitutionConfig()
        d = config.to_dict()
        config2 = ConstitutionConfig.from_dict(d)
        assert len(config.principles) == len(config2.principles)
        assert config.collision_weight == config2.collision_weight

    def test_save_load_json(self):
        config = ConstitutionConfig()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        config.save_json(path)
        config2 = ConstitutionConfig.from_json(path)
        assert len(config2.principles) == 5
        Path(path).unlink()

    def test_get_principle(self):
        config = ConstitutionConfig()
        p = config.get_principle("no_collision")
        assert p is not None
        assert p.severity == "critical"

    def test_is_principle_enabled(self):
        config = ConstitutionConfig()
        assert config.is_principle_enabled("no_collision")
        assert not config.is_principle_enabled("nonexistent")


# ──────────────────── numpy JSON 安全序列化测试 ────────────────────


class TestNumpyJsonSafety:
    """numpy 类型 JSON 序列化安全测试。"""

    def test_numpy_safe_dumps_bool(self):
        data = {"flag": np.bool_(True)}
        result = numpy_safe_dumps(data)
        parsed = json.loads(result)
        assert parsed["flag"] is True
        assert isinstance(parsed["flag"], bool)

    def test_numpy_safe_dumps_float(self):
        data = {"value": np.float64(3.14)}
        result = numpy_safe_dumps(data)
        parsed = json.loads(result)
        assert abs(parsed["value"] - 3.14) < 1e-10

    def test_numpy_safe_dumps_int(self):
        data = {"count": np.int64(42)}
        result = numpy_safe_dumps(data)
        parsed = json.loads(result)
        assert parsed["count"] == 42

    def test_numpy_safe_dumps_array(self):
        data = {"pos": np.array([1.0, 2.0, 3.0])}
        result = numpy_safe_dumps(data)
        parsed = json.loads(result)
        assert parsed["pos"] == [1.0, 2.0, 3.0]

    def test_numpy_safe_dump_to_file(self):
        data = {"flag": np.bool_(False), "val": np.float64(1.5)}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
            numpy_safe_dump(data, f)
        with open(path, encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed["flag"] is False
        assert parsed["val"] == 1.5
        Path(path).unlink()

    def test_violation_result_to_dict_json_serializable(self):
        """确保 ViolationResult.to_dict() 的结果可以被 json.dumps 序列化。"""
        result = ViolationResult(
            violated=np.bool_(True),
            severity=Severity.CRITICAL,
            confidence=np.float64(0.95),
            description="test",
            metrics={"distance": 1.5},
        )
        d = result.to_dict()
        # 应该不会抛出 TypeError
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["violated"] is True

    def test_ego_state_to_dict_json_serializable(self):
        """确保 EgoState.to_dict() 的结果可以被 json.dumps 序列化。"""
        ego = _ego(speed=10.0, heading=0.5)
        d = ego.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert isinstance(parsed["speed"], float)
        assert isinstance(parsed["heading"], float)

    def test_constitution_obstacle_to_dict_json_serializable(self):
        """确保 ConstitutionObstacle.to_dict() 的结果可以被 json.dumps 序列化。"""
        obs = _obstacle(center=(5, 1, 0), speed=3.0, stationary=False)
        d = obs.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert isinstance(parsed["motion"]["speed"], float)
        assert isinstance(parsed["motion"]["is_stationary"], bool)

    def test_collision_principle_result_json_serializable(self):
        """碰撞原则结果可以被 json.dumps 序列化（回归测试）。"""
        p = NoCollisionPrinciple(safety_margin=1.0)
        obs = _obstacle(center=(1.0, 0, 0), dims=(0.5, 0.5, 0.5))
        traj = [
            TrajectoryPoint(
                position=np.array([1.0, 0, 0], dtype=np.float32),
                timestamp=0.5,
            )
        ]
        result = p.evaluate(
            _scene(obstacles=[obs]),
            _decision(trajectory_points=traj),
        )
        d = result.to_dict()
        # 这里之前会抛出 TypeError: Object of type bool_ is not JSON serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert isinstance(parsed["violated"], bool)

    def test_full_evaluation_json_serializable(self):
        """完整评估结果 JSON 序列化回归测试。"""
        evaluator = ConstitutionEvaluator()
        obs = _obstacle(center=(1.0, 0, 0), dims=(0.5, 0.5, 0.5))
        traj = [
            TrajectoryPoint(
                position=np.array([1.0, 0, 0], dtype=np.float32),
                timestamp=0.5,
            )
        ]
        scene = _scene(obstacles=[obs], ego_speed=10.0)
        result = evaluator.evaluate(scene, _decision(trajectory_points=traj))
        d = result.to_dict()
        # 必须不抛出异常
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ──────────────────── TrajectoryPoint.from_dict 测试 ────────────────────


class TestTrajectoryPointFromDict:
    """TrajectoryPoint.from_dict 测试。"""

    def test_basic(self):
        d = {"position": [1.0, 2.0, 3.0], "timestamp": 0.5}
        pt = TrajectoryPoint.from_dict(d)
        np.testing.assert_allclose(pt.position, [1.0, 2.0, 3.0])
        assert pt.timestamp == 0.5
        assert pt.velocity is None

    def test_with_velocity(self):
        d = {
            "position": [1, 0, 0],
            "velocity": [5, 0, 0],
            "timestamp": 1.0,
        }
        pt = TrajectoryPoint.from_dict(d)
        np.testing.assert_allclose(pt.velocity, [5, 0, 0])

    def test_default_timestamp(self):
        d = {"position": [0, 0, 0]}
        pt = TrajectoryPoint.from_dict(d)
        assert pt.timestamp == 0.0


# ──────────────────── AIDecision.from_dict 测试 ────────────────────


class TestAIDecisionFromDict:
    """AIDecision.from_dict 测试。"""

    def test_basic(self):
        d = {
            "decision_type": "trajectory",
            "trajectory": [
                {"position": [1, 0, 0], "timestamp": 0.5},
                {"position": [2, 0, 0], "timestamp": 1.0},
            ],
        }
        decision = AIDecision.from_dict(d)
        assert decision.decision_type == "trajectory"
        assert len(decision.trajectory) == 2
        np.testing.assert_allclose(decision.trajectory[0].position, [1, 0, 0])

    def test_roundtrip(self):
        original = _decision()
        d = original.to_dict()
        restored = AIDecision.from_dict(d)
        assert restored.decision_type == original.decision_type
        assert len(restored.trajectory) == len(original.trajectory)

    def test_with_control(self):
        d = {
            "decision_type": "control",
            "control": {"steering": 0.1, "throttle": 0.5, "brake": 0.0},
            "target_speed": 10.0,
        }
        decision = AIDecision.from_dict(d)
        assert decision.control["steering"] == 0.1
        assert decision.target_speed == 10.0


# ──────────────────── CommandParser 注册测试 ────────────────────


class TestCommandParserRegistry:
    """指令解析器注册测试。"""

    def setup_method(self):
        self._saved = dict(ConstitutionRegistry._command_parsers)

    def teardown_method(self):
        ConstitutionRegistry._command_parsers = self._saved

    def test_builtin_parsers_registered(self):
        import aylm.constitution  # noqa: F401

        names = ConstitutionRegistry.list_command_parsers()
        assert "json" in names
        assert "natural_language" in names

    def test_get_command_parser(self):
        cls = ConstitutionRegistry.get_command_parser("json")
        assert cls is JSONCommandParser

    def test_get_unknown_returns_none(self):
        assert ConstitutionRegistry.get_command_parser("nonexistent") is None


# ──────────────────── JSONCommandParser 测试 ────────────────────


class TestJSONCommandParser:
    """JSON 指令解析器测试。"""

    def test_can_parse_dict(self):
        parser = JSONCommandParser()
        assert parser.can_parse({"type": "trajectory", "points": []})
        assert not parser.can_parse("some string")
        assert not parser.can_parse({})

    def test_parse_trajectory(self):
        parser = JSONCommandParser()
        cmd = {"type": "trajectory", "points": [[5, 0, 0, 0.5], [10, 0, 0, 1.0]]}
        decision = parser.parse(cmd)
        assert decision.decision_type == "trajectory"
        assert len(decision.trajectory) == 2
        np.testing.assert_allclose(decision.trajectory[0].position, [5, 0, 0])
        assert decision.trajectory[0].timestamp == 0.5

    def test_parse_control(self):
        parser = JSONCommandParser()
        cmd = {"type": "control", "steering": 0.1, "throttle": 0.5, "brake": 0.0}
        decision = parser.parse(cmd, ego_speed=10.0)
        assert decision.decision_type == "control"
        assert decision.control["steering"] == 0.1
        assert len(decision.trajectory) > 0

    def test_parse_waypoint(self):
        parser = JSONCommandParser()
        cmd = {"type": "waypoint", "target": [20, 0, 0], "speed": 5.0}
        decision = parser.parse(cmd, ego_speed=5.0)
        assert decision.decision_type == "waypoint"
        assert decision.target_speed == 5.0
        assert len(decision.trajectory) >= 2

    def test_parse_unknown_type_raises(self):
        parser = JSONCommandParser()
        with pytest.raises(ValueError, match="不支持"):
            parser.parse({"type": "unknown"})


# ──────────────────── NaturalLanguageParser 测试 ────────────────────


class TestNaturalLanguageParser:
    """自然语言指令解析器测试。"""

    def test_can_parse_chinese(self):
        parser = NaturalLanguageParser()
        assert parser.can_parse("向左转弯30度")
        assert parser.can_parse("紧急刹车")
        assert parser.can_parse("加速到60")
        assert parser.can_parse("停车")
        assert parser.can_parse("变道到右侧")
        assert not parser.can_parse("")
        assert not parser.can_parse({"type": "json"})

    def test_can_parse_english(self):
        parser = NaturalLanguageParser()
        assert parser.can_parse("turn left 30")
        assert parser.can_parse("emergency brake")
        assert parser.can_parse("stop")
        assert parser.can_parse("change lane right")

    def test_parse_turn_left(self):
        parser = NaturalLanguageParser()
        decision = parser.parse("向左转弯30度", ego_speed=10.0)
        assert decision.decision_type == "trajectory"
        assert len(decision.trajectory) > 0
        assert decision.metadata["parsed_action"] == "turn_left"

    def test_parse_emergency_brake(self):
        parser = NaturalLanguageParser()
        decision = parser.parse("紧急刹车", ego_speed=10.0)
        assert decision.decision_type == "control"
        assert decision.control["brake"] == 1.0
        assert decision.target_speed == 0.0

    def test_parse_accelerate(self):
        parser = NaturalLanguageParser()
        decision = parser.parse("加速到80", ego_speed=10.0)
        assert decision.target_speed == pytest.approx(80 / 3.6, rel=0.01)

    def test_parse_stop(self):
        parser = NaturalLanguageParser()
        decision = parser.parse("停车", ego_speed=10.0)
        assert decision.target_speed == 0.0

    def test_parse_lane_change(self):
        parser = NaturalLanguageParser()
        decision = parser.parse("变道到右侧", ego_speed=10.0)
        assert decision.metadata["parsed_action"] == "lane_change_right"
        # 轨迹应该有横向偏移
        last_y = decision.trajectory[-1].position[1]
        assert last_y > 0  # 右变道 Y > 0

    def test_parse_reverse(self):
        parser = NaturalLanguageParser()
        decision = parser.parse("后退")
        assert decision.trajectory[-1].position[0] < 0  # X 负方向

    def test_parse_unknown_raises(self):
        parser = NaturalLanguageParser()
        with pytest.raises(ValueError, match="无法解析"):
            parser.parse("随便说点什么完全无关的话")


# ──────────────────── CommandValidator 测试 ────────────────────


class TestCommandValidator:
    """指令验证器测试。"""

    def test_safe_trajectory(self):
        validator = CommandValidator()
        result = validator.validate(
            command={"type": "trajectory", "points": [[50, 0, 0, 1.0]]},
            ego_speed=5.0,
        )
        assert isinstance(result, ValidationResult)
        assert result.approved is True
        assert result.safety_score > 0

    def test_collision_trajectory_rejected(self):
        validator = CommandValidator()
        result = validator.validate(
            command={"type": "trajectory", "points": [[1, 0, 0, 0.5]]},
            ego_speed=10.0,
            obstacles=[
                {
                    "center_robot": [1, 0, 0],
                    "dimensions_robot": [0.5, 0.5, 0.5],
                    "_label": "PERSON",
                    "confidence": 0.9,
                }
            ],
        )
        assert result.approved is False
        assert result.safety_score < 1.0
        assert len(result.violations) > 0
        assert result.alternative_decision is not None

    def test_natural_language_validation(self):
        validator = CommandValidator()
        result = validator.validate(
            command="向左转弯",
            ego_speed=10.0,
            obstacles=[
                {
                    "center_robot": [5, 2, 0],
                    "dimensions_robot": [1, 1, 1],
                    "_label": "VEHICLE",
                    "confidence": 0.9,
                }
            ],
        )
        assert isinstance(result, ValidationResult)
        assert isinstance(result.safety_score, float)

    def test_unparseable_command(self):
        validator = CommandValidator()
        result = validator.validate(command=12345)
        assert result.approved is False
        assert "解析失败" in result.reason

    def test_result_to_dict(self):
        validator = CommandValidator()
        result = validator.validate(
            command={"type": "trajectory", "points": [[50, 0, 0, 1.0]]},
            ego_speed=5.0,
        )
        d = result.to_dict()
        assert "approved" in d
        assert "safety_score" in d
        assert "reason" in d

    def test_result_json_serializable(self):
        """验证结果必须可以被 json.dumps 序列化。"""
        validator = CommandValidator()
        result = validator.validate(
            command={"type": "trajectory", "points": [[1, 0, 0, 0.5]]},
            ego_speed=10.0,
            obstacles=[
                {
                    "center_robot": [1, 0, 0],
                    "dimensions_robot": [0.5, 0.5, 0.5],
                    "_label": "PERSON",
                    "confidence": 0.9,
                }
            ],
        )
        d = result.to_dict()
        # 必须不抛出 TypeError
        json_str = numpy_safe_dumps(d)
        assert len(json_str) > 0

    def test_custom_threshold(self):
        validator = CommandValidator(approval_threshold=0.99)
        result = validator.validate(
            command={"type": "trajectory", "points": [[50, 0, 0, 1.0]]},
            ego_speed=5.0,
        )
        # 高阈值可能导致不通过
        assert isinstance(result.approved, bool)

    def test_with_scene_state(self):
        validator = CommandValidator()
        scene = _scene(ego_speed=5.0)
        result = validator.validate(
            command={"type": "trajectory", "points": [[50, 0, 0, 1.0]]},
            scene=scene,
        )
        assert result.approved is True


# ──────────────────── ConstitutionIntegration 扩展测试 ────────────────────


class TestConstitutionIntegrationExtended:
    """宪法集成层扩展测试（新增 decision 参数）。"""

    def test_evaluate_frame_default_decision(self):
        from aylm.tools.constitution_integration import ConstitutionIntegration

        integration = ConstitutionIntegration(ego_speed=5.0)
        if not integration.is_available:
            pytest.skip("宪法评估器不可用")
        result = integration.evaluate_frame([], frame_id=0)
        assert result is not None
        assert "safety_score" in result

    def test_evaluate_frame_with_custom_decision(self):
        from aylm.tools.constitution_integration import ConstitutionIntegration

        integration = ConstitutionIntegration(ego_speed=10.0)
        if not integration.is_available:
            pytest.skip("宪法评估器不可用")

        decision = AIDecision(
            decision_type="trajectory",
            trajectory=[
                TrajectoryPoint(
                    position=np.array([1, 0, 0], dtype=np.float32),
                    timestamp=0.5,
                )
            ],
            target_speed=10.0,
        )
        obstacles = [
            {
                "center_robot": [1, 0, 0],
                "dimensions_robot": [0.5, 0.5, 0.5],
                "_label": "PERSON",
                "confidence": 0.9,
            }
        ]
        result = integration.evaluate_frame(
            obstacles_data=obstacles,
            frame_id=0,
            decision=decision,
        )
        assert result is not None
        # 碰撞场景应该有违规
        score = result.get("safety_score", {}).get("overall", 1.0)
        assert score < 1.0
