"""运动矢量估计器模块。

基于连续帧的 3D 位置变化计算速度，使用 Kalman 滤波器平滑轨迹。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .coordinate_utils import opencv_to_robot
from .semantic_types import SemanticLabel


@dataclass
class MotionVector:
    """运动矢量。"""

    velocity_cv: NDArray[np.float32]  # OpenCV 坐标系速度 [vx, vy, vz] m/s
    velocity_robot: NDArray[np.float32]  # 机器人坐标系速度 [vx, vy, vz] m/s
    speed: float  # 标量速度 m/s
    heading: float  # 航向角（弧度，相对于机器人前进方向）
    is_stationary: bool  # 是否静止（速度 < 阈值）


@dataclass
class TrackedObject3D:
    """3D 跟踪对象。"""

    track_id: int
    center_cv: NDArray[np.float32]  # OpenCV 坐标系中心
    center_robot: NDArray[np.float32]  # 机器人坐标系中心
    dimensions: NDArray[np.float32]  # [width, height, depth]
    motion: Optional[MotionVector]  # 运动信息
    semantic_label: SemanticLabel
    confidence: float
    frame_id: int
    timestamp: float  # 秒


@dataclass
class KalmanConfig:
    """Kalman 滤波器配置。"""

    process_noise: float = 0.1  # 过程噪声
    measurement_noise: float = 0.5  # 观测噪声
    initial_covariance: float = 1.0  # 初始协方差


@dataclass
class _TrackState:
    """单个轨迹的 Kalman 滤波器状态。"""

    # 状态向量: [x, y, z, vx, vy, vz]
    state: NDArray[np.float64]
    # 协方差矩阵 6x6
    covariance: NDArray[np.float64]
    # 上一帧 ID
    last_frame_id: int
    # 上一帧时间戳
    last_timestamp: float
    # 初始化标志
    initialized: bool = False


class MotionEstimator:
    """运动矢量估计器。

    使用 Kalman 滤波器平滑轨迹并估计速度。

    状态向量: [x, y, z, vx, vy, vz]
    观测向量: [x, y, z]
    """

    def __init__(
        self,
        fps: float = 30.0,
        stationary_threshold: float = 0.1,
        kalman_config: Optional[KalmanConfig] = None,
    ):
        """初始化运动估计器。

        Args:
            fps: 帧率，用于计算默认时间间隔
            stationary_threshold: 静止判定阈值（m/s）
            kalman_config: Kalman 滤波器配置
        """
        self.fps = fps
        self.dt_default = 1.0 / fps
        self.stationary_threshold = stationary_threshold
        self.config = kalman_config or KalmanConfig()

        # 轨迹状态字典 {track_id: _TrackState}
        self._tracks: dict[int, _TrackState] = {}

        # 预计算 Kalman 矩阵
        self._init_kalman_matrices()

    def _init_kalman_matrices(self) -> None:
        """初始化 Kalman 滤波器矩阵。"""
        # 观测矩阵 H: 3x6，只观测位置
        self._H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )

        # 观测噪声协方差 R: 3x3
        self._R = np.eye(3, dtype=np.float64) * self.config.measurement_noise

    def _get_transition_matrix(self, dt: float) -> NDArray[np.float64]:
        """获取状态转移矩阵 F。

        Args:
            dt: 时间间隔（秒）

        Returns:
            6x6 状态转移矩阵
        """
        return np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

    def _get_process_noise(self, dt: float) -> NDArray[np.float64]:
        """获取过程噪声协方差矩阵 Q。

        使用离散白噪声加速度模型。

        Args:
            dt: 时间间隔（秒）

        Returns:
            6x6 过程噪声协方差矩阵
        """
        q = self.config.process_noise
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        # 离散白噪声加速度模型
        return (
            np.array(
                [
                    [dt4 / 4, 0, 0, dt3 / 2, 0, 0],
                    [0, dt4 / 4, 0, 0, dt3 / 2, 0],
                    [0, 0, dt4 / 4, 0, 0, dt3 / 2],
                    [dt3 / 2, 0, 0, dt2, 0, 0],
                    [0, dt3 / 2, 0, 0, dt2, 0],
                    [0, 0, dt3 / 2, 0, 0, dt2],
                ],
                dtype=np.float64,
            )
            * q
        )

    def _init_track(
        self, position: NDArray, frame_id: int, timestamp: float
    ) -> _TrackState:
        """初始化新轨迹。

        Args:
            position: 初始位置 [x, y, z]
            frame_id: 帧 ID
            timestamp: 时间戳

        Returns:
            新的轨迹状态
        """
        state = np.zeros(6, dtype=np.float64)
        state[:3] = position

        covariance = np.eye(6, dtype=np.float64) * self.config.initial_covariance
        # 速度初始不确定性更大
        covariance[3:, 3:] *= 10.0

        return _TrackState(
            state=state,
            covariance=covariance,
            last_frame_id=frame_id,
            last_timestamp=timestamp,
            initialized=True,
        )

    def _predict(self, track: _TrackState, dt: float) -> None:
        """Kalman 预测步骤。

        Args:
            track: 轨迹状态
            dt: 时间间隔
        """
        F = self._get_transition_matrix(dt)
        Q = self._get_process_noise(dt)

        # 状态预测: x' = F * x
        track.state = F @ track.state

        # 协方差预测: P' = F * P * F^T + Q
        track.covariance = F @ track.covariance @ F.T + Q

    def _update_kalman(
        self, track: _TrackState, measurement: NDArray[np.float64]
    ) -> None:
        """Kalman 更新步骤。

        Args:
            track: 轨迹状态
            measurement: 观测值 [x, y, z]
        """
        H = self._H
        R = self._R

        # 创新（残差）: y = z - H * x
        y = measurement - H @ track.state

        # 创新协方差: S = H * P * H^T + R
        S = H @ track.covariance @ H.T + R

        # Kalman 增益: K = P * H^T * S^(-1)
        K = track.covariance @ H.T @ np.linalg.inv(S)

        # 状态更新: x = x + K * y
        track.state = track.state + K @ y

        # 协方差更新: P = (I - K * H) * P
        identity = np.eye(6, dtype=np.float64)
        track.covariance = (identity - K @ H) @ track.covariance

    def update(
        self,
        track_id: int,
        position_cv: NDArray,
        frame_id: int,
        timestamp: Optional[float] = None,
    ) -> Optional[MotionVector]:
        """更新轨迹，返回运动矢量。

        Args:
            track_id: 轨迹 ID
            position_cv: OpenCV 坐标系位置 [x, y, z]
            frame_id: 帧 ID
            timestamp: 时间戳（秒），如果为 None 则根据帧率计算

        Returns:
            运动矢量，如果是第一帧则返回 None
        """
        position = np.asarray(position_cv, dtype=np.float64)

        # 计算时间戳
        if timestamp is None:
            timestamp = frame_id / self.fps

        # 检查是否是新轨迹
        if track_id not in self._tracks:
            self._tracks[track_id] = self._init_track(position, frame_id, timestamp)
            return None

        track = self._tracks[track_id]

        # 计算时间间隔
        dt = timestamp - track.last_timestamp
        if dt <= 0:
            dt = self.dt_default

        # Kalman 预测
        self._predict(track, dt)

        # Kalman 更新
        self._update_kalman(track, position)

        # 更新时间信息
        track.last_frame_id = frame_id
        track.last_timestamp = timestamp

        # 提取速度
        velocity_cv = track.state[3:6].astype(np.float32)

        # 转换到机器人坐标系
        velocity_robot = opencv_to_robot(velocity_cv).astype(np.float32)

        # 计算标量速度
        speed = float(np.linalg.norm(velocity_cv))

        # 计算航向角（在机器人坐标系中，相对于 X 轴/前进方向）
        # heading = atan2(vy, vx)，范围 [-pi, pi]
        heading = float(np.arctan2(velocity_robot[1], velocity_robot[0]))

        # 判断是否静止
        is_stationary = speed < self.stationary_threshold

        return MotionVector(
            velocity_cv=velocity_cv,
            velocity_robot=velocity_robot,
            speed=speed,
            heading=heading,
            is_stationary=is_stationary,
        )

    def predict(self, track_id: int, dt: float) -> Optional[NDArray[np.float64]]:
        """预测未来位置。

        Args:
            track_id: 轨迹 ID
            dt: 预测时间间隔（秒）

        Returns:
            预测的 OpenCV 坐标系位置 [x, y, z]，如果轨迹不存在则返回 None
        """
        if track_id not in self._tracks:
            return None

        track = self._tracks[track_id]
        F = self._get_transition_matrix(dt)

        # 预测状态
        predicted_state = F @ track.state

        return predicted_state[:3]

    def get_velocity(self, track_id: int) -> Optional[NDArray[np.float64]]:
        """获取当前速度估计。

        Args:
            track_id: 轨迹 ID

        Returns:
            OpenCV 坐标系速度 [vx, vy, vz]，如果轨迹不存在则返回 None
        """
        if track_id not in self._tracks:
            return None

        return self._tracks[track_id].state[3:6].copy()

    def remove_track(self, track_id: int) -> bool:
        """移除轨迹。

        Args:
            track_id: 轨迹 ID

        Returns:
            是否成功移除
        """
        if track_id in self._tracks:
            del self._tracks[track_id]
            return True
        return False

    def clear(self) -> None:
        """清除所有轨迹。"""
        self._tracks.clear()

    @property
    def active_tracks(self) -> list[int]:
        """获取活跃轨迹 ID 列表。"""
        return list(self._tracks.keys())


def create_tracked_object(
    track_id: int,
    center_cv: NDArray,
    dimensions: NDArray,
    semantic_label: SemanticLabel,
    confidence: float,
    frame_id: int,
    timestamp: float,
    motion: Optional[MotionVector] = None,
) -> TrackedObject3D:
    """创建 3D 跟踪对象。

    Args:
        track_id: 轨迹 ID
        center_cv: OpenCV 坐标系中心
        dimensions: 尺寸 [width, height, depth]
        semantic_label: 语义标签
        confidence: 置信度
        frame_id: 帧 ID
        timestamp: 时间戳
        motion: 运动信息

    Returns:
        TrackedObject3D 实例
    """
    center_cv = np.asarray(center_cv, dtype=np.float32)
    center_robot = opencv_to_robot(center_cv).astype(np.float32)
    dimensions = np.asarray(dimensions, dtype=np.float32)

    return TrackedObject3D(
        track_id=track_id,
        center_cv=center_cv,
        center_robot=center_robot,
        dimensions=dimensions,
        motion=motion,
        semantic_label=semantic_label,
        confidence=confidence,
        frame_id=frame_id,
        timestamp=timestamp,
    )
