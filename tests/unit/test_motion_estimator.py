"""运动矢量估计器模块测试。"""

import numpy as np
import pytest

from aylm.tools.motion_estimator import (
    KalmanConfig,
    MotionEstimator,
    MotionVector,
    TrackedObject3D,
    create_tracked_object,
)
from aylm.tools.semantic_types import SemanticLabel


class TestMotionVector:
    """MotionVector 数据类测试。"""

    def test_creation(self):
        """测试创建 MotionVector。"""
        velocity_cv = np.array([1.0, 0.0, 2.0], dtype=np.float32)
        velocity_robot = np.array([2.0, -1.0, 0.0], dtype=np.float32)

        mv = MotionVector(
            velocity_cv=velocity_cv,
            velocity_robot=velocity_robot,
            speed=2.236,
            heading=0.5,
            is_stationary=False,
        )

        np.testing.assert_array_equal(mv.velocity_cv, velocity_cv)
        np.testing.assert_array_equal(mv.velocity_robot, velocity_robot)
        assert mv.speed == pytest.approx(2.236)
        assert mv.heading == pytest.approx(0.5)
        assert mv.is_stationary is False

    def test_stationary_flag(self):
        """测试静止标志。"""
        mv = MotionVector(
            velocity_cv=np.array([0.01, 0.01, 0.01], dtype=np.float32),
            velocity_robot=np.array([0.01, -0.01, -0.01], dtype=np.float32),
            speed=0.017,
            heading=0.0,
            is_stationary=True,
        )
        assert mv.is_stationary is True


class TestKalmanConfig:
    """KalmanConfig 数据类测试。"""

    def test_default_values(self):
        """测试默认配置值。"""
        config = KalmanConfig()
        assert config.process_noise == 0.1
        assert config.measurement_noise == 0.5
        assert config.initial_covariance == 1.0

    def test_custom_values(self):
        """测试自定义配置值。"""
        config = KalmanConfig(
            process_noise=0.2,
            measurement_noise=0.3,
            initial_covariance=2.0,
        )
        assert config.process_noise == 0.2
        assert config.measurement_noise == 0.3
        assert config.initial_covariance == 2.0


class TestMotionEstimator:
    """MotionEstimator 类测试。"""

    def test_init_default(self):
        """测试默认初始化。"""
        estimator = MotionEstimator()
        assert estimator.fps == 30.0
        assert estimator.stationary_threshold == 0.1
        assert len(estimator.active_tracks) == 0

    def test_init_custom(self):
        """测试自定义初始化。"""
        config = KalmanConfig(process_noise=0.2)
        estimator = MotionEstimator(
            fps=60.0,
            stationary_threshold=0.2,
            kalman_config=config,
        )
        assert estimator.fps == 60.0
        assert estimator.stationary_threshold == 0.2
        assert estimator.config.process_noise == 0.2

    def test_first_update_returns_none(self):
        """测试首次更新返回 None（无速度信息）。"""
        estimator = MotionEstimator()
        position = np.array([1.0, 2.0, 3.0])

        result = estimator.update(track_id=1, position_cv=position, frame_id=0)

        assert result is None
        assert 1 in estimator.active_tracks

    def test_second_update_returns_motion(self):
        """测试第二次更新返回运动矢量。"""
        estimator = MotionEstimator(fps=30.0)

        # 第一帧
        pos1 = np.array([0.0, 0.0, 0.0])
        estimator.update(track_id=1, position_cv=pos1, frame_id=0)

        # 第二帧，移动了 1 米（在 Z 方向）
        pos2 = np.array([0.0, 0.0, 1.0])
        result = estimator.update(track_id=1, position_cv=pos2, frame_id=1)

        assert result is not None
        assert isinstance(result, MotionVector)
        # 速度应该大于 0
        assert result.speed > 0

    def test_velocity_calculation(self):
        """测试速度计算。"""
        estimator = MotionEstimator(fps=1.0)  # 1 FPS 便于计算

        # 第一帧
        pos1 = np.array([0.0, 0.0, 0.0])
        estimator.update(track_id=1, position_cv=pos1, frame_id=0, timestamp=0.0)

        # 第二帧，1 秒后移动 1 米
        pos2 = np.array([1.0, 0.0, 0.0])
        result = estimator.update(
            track_id=1, position_cv=pos2, frame_id=1, timestamp=1.0
        )

        assert result is not None
        # Kalman 滤波会平滑速度，但应该接近 1 m/s
        assert result.speed > 0.5

    def test_stationary_detection(self):
        """测试静止检测。"""
        estimator = MotionEstimator(fps=30.0, stationary_threshold=0.1)

        # 第一帧
        pos1 = np.array([0.0, 0.0, 5.0])
        estimator.update(track_id=1, position_cv=pos1, frame_id=0)

        # 第二帧，几乎不动
        pos2 = np.array([0.001, 0.001, 5.001])
        result = estimator.update(track_id=1, position_cv=pos2, frame_id=1)

        assert result is not None
        # 速度很小，应该被判定为静止
        assert result.is_stationary is True

    def test_moving_detection(self):
        """测试运动检测。"""
        estimator = MotionEstimator(fps=1.0, stationary_threshold=0.1)

        # 第一帧
        pos1 = np.array([0.0, 0.0, 0.0])
        estimator.update(track_id=1, position_cv=pos1, frame_id=0, timestamp=0.0)

        # 第二帧，明显移动
        pos2 = np.array([2.0, 0.0, 0.0])
        result = estimator.update(
            track_id=1, position_cv=pos2, frame_id=1, timestamp=1.0
        )

        assert result is not None
        assert result.is_stationary is False

    def test_predict_future_position(self):
        """测试位置预测。"""
        estimator = MotionEstimator(fps=1.0)

        # 建立轨迹
        estimator.update(
            track_id=1, position_cv=np.array([0.0, 0.0, 0.0]), frame_id=0, timestamp=0.0
        )
        estimator.update(
            track_id=1, position_cv=np.array([1.0, 0.0, 0.0]), frame_id=1, timestamp=1.0
        )

        # 预测 1 秒后的位置
        predicted = estimator.predict(track_id=1, dt=1.0)

        assert predicted is not None
        # 应该在 X 方向继续移动
        assert predicted[0] > 1.0

    def test_predict_nonexistent_track(self):
        """测试预测不存在的轨迹。"""
        estimator = MotionEstimator()
        result = estimator.predict(track_id=999, dt=1.0)
        assert result is None

    def test_get_velocity(self):
        """测试获取速度。"""
        estimator = MotionEstimator(fps=1.0)

        # 建立轨迹
        estimator.update(
            track_id=1, position_cv=np.array([0.0, 0.0, 0.0]), frame_id=0, timestamp=0.0
        )
        estimator.update(
            track_id=1, position_cv=np.array([1.0, 0.0, 0.0]), frame_id=1, timestamp=1.0
        )

        velocity = estimator.get_velocity(track_id=1)

        assert velocity is not None
        assert len(velocity) == 3

    def test_get_velocity_nonexistent_track(self):
        """测试获取不存在轨迹的速度。"""
        estimator = MotionEstimator()
        result = estimator.get_velocity(track_id=999)
        assert result is None

    def test_multiple_tracks(self):
        """测试多目标同时跟踪。"""
        estimator = MotionEstimator()

        # 跟踪两个目标
        estimator.update(track_id=1, position_cv=np.array([0.0, 0.0, 0.0]), frame_id=0)
        estimator.update(
            track_id=2, position_cv=np.array([10.0, 10.0, 10.0]), frame_id=0
        )

        assert len(estimator.active_tracks) == 2
        assert 1 in estimator.active_tracks
        assert 2 in estimator.active_tracks

    def test_remove_track(self):
        """测试移除轨迹。"""
        estimator = MotionEstimator()

        estimator.update(track_id=1, position_cv=np.array([0.0, 0.0, 0.0]), frame_id=0)
        assert 1 in estimator.active_tracks

        result = estimator.remove_track(track_id=1)
        assert result is True
        assert 1 not in estimator.active_tracks

    def test_remove_nonexistent_track(self):
        """测试移除不存在的轨迹。"""
        estimator = MotionEstimator()
        result = estimator.remove_track(track_id=999)
        assert result is False

    def test_clear(self):
        """测试清除所有轨迹。"""
        estimator = MotionEstimator()

        # 添加多个轨迹
        for i in range(5):
            estimator.update(
                track_id=i, position_cv=np.array([float(i), 0.0, 0.0]), frame_id=0
            )

        assert len(estimator.active_tracks) == 5

        estimator.clear()
        assert len(estimator.active_tracks) == 0

    def test_heading_calculation(self):
        """测试航向角计算。"""
        estimator = MotionEstimator(fps=1.0)

        # 第一帧
        estimator.update(
            track_id=1, position_cv=np.array([0.0, 0.0, 0.0]), frame_id=0, timestamp=0.0
        )

        # 第二帧，沿 Z 轴正方向移动（OpenCV 坐标系）
        # 在机器人坐标系中，这对应 X 轴正方向（前进）
        result = estimator.update(
            track_id=1, position_cv=np.array([0.0, 0.0, 5.0]), frame_id=1, timestamp=1.0
        )

        assert result is not None
        # 航向角应该接近 0（正前方）
        assert abs(result.heading) < 1.0  # 允许一定误差

    def test_kalman_smoothing(self):
        """测试 Kalman 滤波平滑效果。"""
        estimator = MotionEstimator(fps=10.0)

        # 模拟带噪声的轨迹
        np.random.seed(42)
        base_positions = [np.array([float(i), 0.0, 0.0]) for i in range(10)]
        noisy_positions = [p + np.random.normal(0, 0.1, 3) for p in base_positions]

        results = []
        for i, pos in enumerate(noisy_positions):
            result = estimator.update(track_id=1, position_cv=pos, frame_id=i)
            if result is not None:
                results.append(result)

        # 速度应该相对稳定（Kalman 滤波平滑了噪声）
        speeds = [r.speed for r in results]
        speed_std = np.std(speeds)

        # 标准差应该较小（平滑效果）
        assert speed_std < 5.0  # 合理的阈值


class TestTrackedObject3D:
    """TrackedObject3D 数据类测试。"""

    def test_creation(self):
        """测试创建 TrackedObject3D。"""
        obj = TrackedObject3D(
            track_id=1,
            center_cv=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            center_robot=np.array([3.0, -1.0, -2.0], dtype=np.float32),
            dimensions=np.array([0.5, 1.7, 0.4], dtype=np.float32),
            motion=None,
            semantic_label=SemanticLabel.PERSON,
            confidence=0.95,
            frame_id=10,
            timestamp=0.333,
        )

        assert obj.track_id == 1
        assert obj.semantic_label == SemanticLabel.PERSON
        assert obj.confidence == 0.95
        assert obj.frame_id == 10
        assert obj.timestamp == pytest.approx(0.333)
        assert obj.motion is None


class TestCreateTrackedObject:
    """create_tracked_object 辅助函数测试。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        obj = create_tracked_object(
            track_id=1,
            center_cv=np.array([0.0, 0.0, 5.0]),
            dimensions=np.array([0.6, 1.7, 0.4]),
            semantic_label=SemanticLabel.PERSON,
            confidence=0.9,
            frame_id=0,
            timestamp=0.0,
        )

        assert obj.track_id == 1
        assert obj.semantic_label == SemanticLabel.PERSON
        assert obj.confidence == 0.9
        # 验证坐标系转换
        assert obj.center_cv is not None
        assert obj.center_robot is not None

    def test_with_motion(self):
        """测试带运动信息的创建。"""
        motion = MotionVector(
            velocity_cv=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            velocity_robot=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            speed=1.0,
            heading=0.0,
            is_stationary=False,
        )

        obj = create_tracked_object(
            track_id=1,
            center_cv=np.array([0.0, 0.0, 5.0]),
            dimensions=np.array([0.6, 1.7, 0.4]),
            semantic_label=SemanticLabel.VEHICLE,
            confidence=0.85,
            frame_id=10,
            timestamp=0.333,
            motion=motion,
        )

        assert obj.motion is not None
        assert obj.motion.speed == 1.0
        assert obj.motion.is_stationary is False
