"""多目标跟踪器模块测试。"""

import numpy as np
import pytest

from aylm.tools.object_tracker import (
    MultiObjectTracker,
    TrackedObject,
    TrackerConfig,
    _compute_iou,
)
from aylm.tools.semantic_types import Detection2D, SemanticLabel


class TestComputeIoU:
    """IoU 计算测试。"""

    def test_identical_boxes(self):
        """测试完全重叠的边界框。"""
        bbox = np.array([0, 0, 100, 100], dtype=np.float32)
        iou = _compute_iou(bbox, bbox)
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self):
        """测试无重叠的边界框。"""
        bbox1 = np.array([0, 0, 50, 50], dtype=np.float32)
        bbox2 = np.array([100, 100, 150, 150], dtype=np.float32)
        iou = _compute_iou(bbox1, bbox2)
        assert iou == 0.0

    def test_partial_overlap(self):
        """测试部分重叠的边界框。"""
        bbox1 = np.array([0, 0, 100, 100], dtype=np.float32)
        bbox2 = np.array([50, 50, 150, 150], dtype=np.float32)
        # 交集: 50x50 = 2500
        # 并集: 10000 + 10000 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.143
        iou = _compute_iou(bbox1, bbox2)
        assert iou == pytest.approx(2500 / 17500, rel=0.01)

    def test_one_inside_other(self):
        """测试一个边界框完全在另一个内部。"""
        bbox1 = np.array([0, 0, 100, 100], dtype=np.float32)
        bbox2 = np.array([25, 25, 75, 75], dtype=np.float32)
        # 交集: 50x50 = 2500
        # 并集: 10000 (大框面积)
        # IoU = 2500 / 10000 = 0.25
        iou = _compute_iou(bbox1, bbox2)
        assert iou == pytest.approx(0.25, rel=0.01)


class TestTrackedObject:
    """TrackedObject 数据类测试。"""

    def test_creation(self):
        """测试创建 TrackedObject。"""
        bbox = np.array([10, 20, 110, 120], dtype=np.float32)
        obj = TrackedObject(
            track_id=1,
            bbox=bbox,
            class_id=0,
            confidence=0.9,
            age=5,
            hits=3,
            time_since_update=0,
        )
        assert obj.track_id == 1
        assert obj.class_id == 0
        assert obj.confidence == 0.9
        assert obj.age == 5
        assert obj.hits == 3
        np.testing.assert_array_equal(obj.bbox, bbox)

    def test_bbox_type_conversion(self):
        """测试 bbox 类型自动转换。"""
        # 使用列表创建
        obj = TrackedObject(
            track_id=1,
            bbox=[10, 20, 110, 120],  # type: ignore
            class_id=0,
            confidence=0.9,
        )
        assert obj.bbox.dtype == np.float32
        assert isinstance(obj.bbox, np.ndarray)


class TestMultiObjectTracker:
    """MultiObjectTracker 类测试。"""

    def _create_detection(
        self,
        bbox: list[float],
        class_id: int = 0,
        confidence: float = 0.9,
    ) -> Detection2D:
        """创建测试用检测结果。"""
        return Detection2D(
            bbox=np.array(bbox, dtype=np.float32),
            mask=None,
            class_id=class_id,
            confidence=confidence,
            semantic_label=SemanticLabel.PERSON,
        )

    def test_init_default_config(self):
        """测试默认配置初始化。"""
        tracker = MultiObjectTracker()
        assert tracker.config.max_age == 30
        assert tracker.config.min_hits == 3
        assert tracker.config.iou_threshold == 0.3

    def test_init_custom_config(self):
        """测试自定义配置初始化。"""
        config = TrackerConfig(max_age=10, min_hits=2, iou_threshold=0.5)
        tracker = MultiObjectTracker(config=config)
        assert tracker.config.max_age == 10
        assert tracker.config.min_hits == 2
        assert tracker.config.iou_threshold == 0.5

    def test_init_with_params(self):
        """测试使用参数初始化。"""
        tracker = MultiObjectTracker(max_age=20, min_hits=5, iou_threshold=0.4)
        assert tracker.config.max_age == 20
        assert tracker.config.min_hits == 5
        assert tracker.config.iou_threshold == 0.4

    def test_first_frame_creates_tracks(self):
        """测试第一帧创建轨迹。"""
        tracker = MultiObjectTracker(min_hits=1)
        detections = [
            self._create_detection([0, 0, 50, 50]),
            self._create_detection([100, 100, 150, 150]),
        ]

        tracked = tracker.update(detections, frame_id=0)

        # min_hits=1，第一帧就应该返回确认轨迹
        assert tracker.track_count == 2
        assert len(tracked) == 2

    def test_track_id_assignment(self):
        """测试 track_id 分配。"""
        tracker = MultiObjectTracker(min_hits=1)
        detections = [
            self._create_detection([0, 0, 50, 50]),
            self._create_detection([100, 100, 150, 150]),
        ]

        tracked = tracker.update(detections, frame_id=0)

        # 验证 track_id 是唯一的
        track_ids = [t.track_id for t in tracked]
        assert len(track_ids) == len(set(track_ids))
        assert 1 in track_ids
        assert 2 in track_ids

    def test_track_association(self):
        """测试跨帧轨迹关联。"""
        tracker = MultiObjectTracker(min_hits=1, iou_threshold=0.3)

        # 第一帧
        det1 = [self._create_detection([0, 0, 50, 50])]
        tracked1 = tracker.update(det1, frame_id=0)
        track_id = tracked1[0].track_id

        # 第二帧，目标稍微移动
        det2 = [self._create_detection([5, 5, 55, 55])]
        tracked2 = tracker.update(det2, frame_id=1)

        # 应该保持相同的 track_id
        assert len(tracked2) == 1
        assert tracked2[0].track_id == track_id

    def test_new_track_creation(self):
        """测试新目标出现时创建新轨迹。"""
        tracker = MultiObjectTracker(min_hits=1)

        # 第一帧：一个目标
        det1 = [self._create_detection([0, 0, 50, 50])]
        tracker.update(det1, frame_id=0)

        # 第二帧：两个目标（原有 + 新增）
        det2 = [
            self._create_detection([0, 0, 50, 50]),
            self._create_detection([200, 200, 250, 250]),
        ]
        tracked = tracker.update(det2, frame_id=1)

        assert tracker.track_count == 2
        assert len(tracked) == 2

    def test_track_deletion_after_max_age(self):
        """测试超过 max_age 后删除轨迹。"""
        tracker = MultiObjectTracker(max_age=3, min_hits=1)

        # 第一帧：创建轨迹
        det1 = [self._create_detection([0, 0, 50, 50])]
        tracker.update(det1, frame_id=0)
        assert tracker.track_count == 1

        # 后续帧：无检测
        for i in range(1, 5):
            tracker.update([], frame_id=i)

        # 轨迹应该被删除
        assert tracker.track_count == 0

    def test_min_hits_confirmation(self):
        """测试 min_hits 确认机制。"""
        tracker = MultiObjectTracker(min_hits=3)

        det = [self._create_detection([0, 0, 50, 50])]

        # 第一帧：未确认
        tracked1 = tracker.update(det, frame_id=0)
        assert len(tracked1) == 0

        # 第二帧：未确认
        tracked2 = tracker.update(det, frame_id=1)
        assert len(tracked2) == 0

        # 第三帧：确认
        tracked3 = tracker.update(det, frame_id=2)
        assert len(tracked3) == 1

    def test_reset(self):
        """测试重置跟踪器。"""
        tracker = MultiObjectTracker(min_hits=1)

        det = [self._create_detection([0, 0, 50, 50])]
        tracker.update(det, frame_id=0)
        assert tracker.track_count == 1

        tracker.reset()
        assert tracker.track_count == 0
        assert tracker.confirmed_track_count == 0

    def test_get_all_tracks(self):
        """测试获取所有轨迹（包括未确认的）。"""
        tracker = MultiObjectTracker(min_hits=3)

        det = [self._create_detection([0, 0, 50, 50])]
        tracker.update(det, frame_id=0)

        # 未确认，但应该在 all_tracks 中
        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 1

    def test_empty_detections(self):
        """测试空检测列表。"""
        tracker = MultiObjectTracker(min_hits=1)

        # 空检测不应该崩溃
        tracked = tracker.update([], frame_id=0)
        assert len(tracked) == 0
        assert tracker.track_count == 0

    def test_track_hits_increment(self):
        """测试 hits 计数递增。"""
        tracker = MultiObjectTracker(min_hits=1)

        det = [self._create_detection([0, 0, 50, 50])]

        # 连续更新
        for i in range(5):
            tracked = tracker.update(det, frame_id=i)

        # hits 应该是 5
        assert len(tracked) == 1
        assert tracked[0].hits == 5

    def test_time_since_update(self):
        """测试 time_since_update 字段。"""
        tracker = MultiObjectTracker(min_hits=1, max_age=10)

        det = [self._create_detection([0, 0, 50, 50])]

        # 第一帧
        tracker.update(det, frame_id=0)

        # 后续帧无检测
        tracker.update([], frame_id=1)
        tracker.update([], frame_id=2)

        # 获取所有轨迹（包括未匹配的）
        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 1
        assert all_tracks[0].time_since_update == 2

    def test_class_id_preserved(self):
        """测试 class_id 保持不变。"""
        tracker = MultiObjectTracker(min_hits=1)

        det = [self._create_detection([0, 0, 50, 50], class_id=2)]
        tracked = tracker.update(det, frame_id=0)

        assert tracked[0].class_id == 2

    def test_confidence_updated(self):
        """测试置信度更新。"""
        tracker = MultiObjectTracker(min_hits=1)

        # 第一帧
        det1 = [self._create_detection([0, 0, 50, 50], confidence=0.8)]
        tracker.update(det1, frame_id=0)

        # 第二帧，置信度变化
        det2 = [self._create_detection([0, 0, 50, 50], confidence=0.95)]
        tracked = tracker.update(det2, frame_id=1)

        assert tracked[0].confidence == 0.95
