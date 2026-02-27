"""多目标跟踪模块。

基于 IoU 匹配的多目标跟踪器，使用匈牙利算法进行检测-轨迹匹配。
支持轨迹的创建、更新和删除。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from .semantic_types import Detection2D

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """跟踪目标。"""

    track_id: int  # 唯一跟踪 ID
    bbox: NDArray[np.float32]  # [x1, y1, x2, y2]
    class_id: int  # COCO 类别 ID
    confidence: float  # 置信度
    age: int = 1  # 跟踪帧数
    hits: int = 1  # 成功匹配次数
    time_since_update: int = 0  # 距上次更新的帧数

    def __post_init__(self) -> None:
        """确保 bbox 是正确的类型。"""
        if not isinstance(self.bbox, np.ndarray):
            self.bbox = np.array(self.bbox, dtype=np.float32)
        elif self.bbox.dtype != np.float32:
            self.bbox = self.bbox.astype(np.float32)


@dataclass
class TrackerConfig:
    """跟踪器配置参数。"""

    max_age: int = 30  # 轨迹最大存活帧数（无匹配时）
    min_hits: int = 3  # 轨迹确认所需最小匹配次数
    iou_threshold: float = 0.3  # IoU 匹配阈值


@dataclass
class _Track:
    """内部轨迹状态。"""

    track_id: int
    bbox: NDArray[np.float32]
    class_id: int
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    velocity: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )

    def predict(self) -> NDArray[np.float32]:
        """预测下一帧位置（简单线性预测）。"""
        predicted_bbox = self.bbox + self.velocity
        # 确保边界框有效
        predicted_bbox[2] = max(predicted_bbox[2], predicted_bbox[0] + 1)
        predicted_bbox[3] = max(predicted_bbox[3], predicted_bbox[1] + 1)
        return predicted_bbox

    def update(self, detection: Detection2D) -> None:
        """使用新检测更新轨迹。"""
        new_bbox = detection.bbox.astype(np.float32)
        # 更新速度（指数移动平均）
        self.velocity = 0.7 * self.velocity + 0.3 * (new_bbox - self.bbox)
        self.bbox = new_bbox
        self.class_id = detection.class_id
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        """标记为未匹配。"""
        self.time_since_update += 1
        # 使用预测位置
        self.bbox = self.predict()

    def to_tracked_object(self) -> TrackedObject:
        """转换为 TrackedObject。"""
        return TrackedObject(
            track_id=self.track_id,
            bbox=self.bbox.copy(),
            class_id=self.class_id,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
        )


def _compute_iou(bbox1: NDArray[np.float32], bbox2: NDArray[np.float32]) -> float:
    """计算两个边界框的 IoU。

    Args:
        bbox1: 边界框 1，[x1, y1, x2, y2]
        bbox2: 边界框 2，[x1, y1, x2, y2]

    Returns:
        IoU 值
    """
    # 计算交集
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 计算并集
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def _compute_iou_matrix(
    tracks: list[_Track], detections: list[Detection2D]
) -> NDArray[np.float32]:
    """计算轨迹和检测之间的 IoU 矩阵。

    Args:
        tracks: 轨迹列表
        detections: 检测列表

    Returns:
        IoU 矩阵，shape (len(tracks), len(detections))
    """
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float32)

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou_matrix[i, j] = _compute_iou(track.bbox, det.bbox)

    return iou_matrix


class MultiObjectTracker:
    """多目标跟踪器。

    基于 IoU 匹配的简化版跟踪器，使用匈牙利算法进行检测-轨迹匹配。

    使用示例:
        >>> tracker = MultiObjectTracker()
        >>> for frame_id, detections in enumerate(all_detections):
        ...     tracked = tracker.update(detections, frame_id)
        ...     for obj in tracked:
        ...         print(f"Track {obj.track_id}: {obj.bbox}")
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        config: Optional[TrackerConfig] = None,
    ):
        """初始化跟踪器。

        Args:
            max_age: 轨迹最大存活帧数（无匹配时）
            min_hits: 轨迹确认所需最小匹配次数
            iou_threshold: IoU 匹配阈值
            config: 跟踪器配置（优先级高于单独参数）
        """
        if config is not None:
            self.config = config
        else:
            self.config = TrackerConfig(
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
            )

        self._tracks: list[_Track] = []
        self._next_id: int = 1
        self._frame_count: int = 0

        logger.info(
            f"MultiObjectTracker 初始化: max_age={self.config.max_age}, "
            f"min_hits={self.config.min_hits}, iou_threshold={self.config.iou_threshold}"
        )

    def update(
        self, detections: list[Detection2D], frame_id: Optional[int] = None
    ) -> list[TrackedObject]:
        """更新跟踪器，返回跟踪结果。

        Args:
            detections: 当前帧的检测结果
            frame_id: 帧 ID（可选，用于日志）

        Returns:
            确认的跟踪目标列表
        """
        self._frame_count += 1
        current_frame = frame_id if frame_id is not None else self._frame_count

        logger.debug(f"帧 {current_frame}: 收到 {len(detections)} 个检测")

        # 增加所有轨迹的年龄
        for track in self._tracks:
            track.age += 1

        if not detections:
            # 没有检测，标记所有轨迹为未匹配
            for track in self._tracks:
                track.mark_missed()
            self._remove_dead_tracks()
            return self._get_confirmed_tracks()

        if not self._tracks:
            # 没有轨迹，为所有检测创建新轨迹
            for det in detections:
                self._create_track(det)
            return self._get_confirmed_tracks()

        # 计算 IoU 矩阵
        iou_matrix = _compute_iou_matrix(self._tracks, detections)

        # 使用匈牙利算法进行匹配
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = (
            self._hungarian_match(iou_matrix)
        )

        logger.debug(
            f"匹配结果: {len(matched_tracks)} 匹配, "
            f"{len(unmatched_tracks)} 未匹配轨迹, "
            f"{len(unmatched_dets)} 未匹配检测"
        )

        # 更新匹配的轨迹
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            self._tracks[track_idx].update(detections[det_idx])

        # 标记未匹配的轨迹
        for track_idx in unmatched_tracks:
            self._tracks[track_idx].mark_missed()

        # 为未匹配的检测创建新轨迹
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        # 删除死亡轨迹
        self._remove_dead_tracks()

        return self._get_confirmed_tracks()

    def _hungarian_match(
        self, iou_matrix: NDArray[np.float32]
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """使用匈牙利算法进行匹配。

        Args:
            iou_matrix: IoU 矩阵

        Returns:
            (匹配的轨迹索引, 匹配的检测索引, 未匹配的轨迹索引, 未匹配的检测索引)
        """
        if iou_matrix.size == 0:
            return [], [], list(range(len(self._tracks))), []

        # 转换为代价矩阵（1 - IoU）
        cost_matrix = 1 - iou_matrix

        # 匈牙利算法
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        matched_tracks: list[int] = []
        matched_dets: list[int] = []
        unmatched_tracks: list[int] = list(range(len(self._tracks)))
        unmatched_dets: list[int] = list(range(iou_matrix.shape[1]))

        for track_idx, det_idx in zip(track_indices, det_indices):
            # 检查 IoU 是否超过阈值
            if iou_matrix[track_idx, det_idx] >= self.config.iou_threshold:
                matched_tracks.append(int(track_idx))
                matched_dets.append(int(det_idx))
                unmatched_tracks.remove(int(track_idx))
                unmatched_dets.remove(int(det_idx))

        return matched_tracks, matched_dets, unmatched_tracks, unmatched_dets

    def _create_track(self, detection: Detection2D) -> _Track:
        """创建新轨迹。"""
        track = _Track(
            track_id=self._next_id,
            bbox=detection.bbox.astype(np.float32),
            class_id=detection.class_id,
            confidence=detection.confidence,
        )
        self._tracks.append(track)
        self._next_id += 1

        logger.debug(f"创建新轨迹 ID={track.track_id}")
        return track

    def _remove_dead_tracks(self) -> None:
        """删除死亡轨迹。"""
        before_count = len(self._tracks)
        self._tracks = [
            t for t in self._tracks if t.time_since_update < self.config.max_age
        ]
        removed = before_count - len(self._tracks)
        if removed > 0:
            logger.debug(f"删除 {removed} 条死亡轨迹")

    def _get_confirmed_tracks(self) -> list[TrackedObject]:
        """获取确认的轨迹（hits >= min_hits）。"""
        confirmed = [
            t.to_tracked_object()
            for t in self._tracks
            if t.hits >= self.config.min_hits and t.time_since_update == 0
        ]
        logger.debug(f"返回 {len(confirmed)} 条确认轨迹")
        return confirmed

    def reset(self) -> None:
        """重置跟踪器状态。"""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        logger.info("跟踪器已重置")

    def get_all_tracks(self) -> list[TrackedObject]:
        """获取所有轨迹（包括未确认的）。"""
        return [t.to_tracked_object() for t in self._tracks]

    @property
    def track_count(self) -> int:
        """当前轨迹数量。"""
        return len(self._tracks)

    @property
    def confirmed_track_count(self) -> int:
        """确认轨迹数量。"""
        return len([t for t in self._tracks if t.hits >= self.config.min_hits])
