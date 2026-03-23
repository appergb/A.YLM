"""A.YLM 工具模块。

注意：为了避免导入时拉起重依赖（torch/open3d/ultralytics 等），
这里采用惰性导入。访问具体符号时才会加载对应模块。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coordinate_utils import (
        opencv_to_robot,
        robot_to_opencv,
        transform_for_navigation,
        transform_obstacle_center,
    )
    from .motion_estimator import (
        KalmanConfig,
        MotionEstimator,
        MotionVector,
        TrackedObject3D,
        create_tracked_object,
    )
    from .multiframe_fusion import (
        FramePose,
        FusionResult,
        MultiframeFusion,
        RegistrationConfig,
        RegistrationResult,
    )
    from .object_detector import DetectorConfig, ObjectDetector
    from .object_tracker import MultiObjectTracker, TrackedObject, TrackerConfig
    from .obstacle_marker import (
        ObstacleBox3D,
        ObstacleMarker,
        ObstacleMarkerConfig,
        extract_obstacles,
    )
    from .pipeline_processor import (
        PipelineConfig,
        PipelineProcessor,
        PipelineStats,
        run_pipeline,
        run_pipeline_async,
    )
    from .pointcloud_slicer import (
        PointCloudSlicer,
        SlicerConfig,
        SliceResult,
        slice_pointcloud,
    )
    from .pointcloud_voxelizer import PointCloud, PointCloudVoxelizer, VoxelizerConfig
    from .semantic_fusion import CameraIntrinsics, FusionConfig, SemanticFusion
    from .semantic_types import (
        COCO_TO_SEMANTIC,
        SEMANTIC_COLORS,
        Detection2D,
        SemanticLabel,
        SemanticPointCloud,
    )

__all__ = [
    # 点云处理
    "PointCloud",
    "PointCloudVoxelizer",
    "VoxelizerConfig",
    # 切片
    "PointCloudSlicer",
    "SlicerConfig",
    "SliceResult",
    "slice_pointcloud",
    # 语义类型
    "SemanticLabel",
    "Detection2D",
    "SemanticPointCloud",
    "COCO_TO_SEMANTIC",
    "SEMANTIC_COLORS",
    # 目标检测
    "ObjectDetector",
    "DetectorConfig",
    # 语义融合
    "SemanticFusion",
    "FusionConfig",
    "CameraIntrinsics",
    # 障碍物标记
    "ObstacleMarker",
    "ObstacleMarkerConfig",
    "ObstacleBox3D",
    "extract_obstacles",
    # 流水线
    "PipelineConfig",
    "PipelineProcessor",
    "PipelineStats",
    "run_pipeline",
    "run_pipeline_async",
    # 坐标转换
    "transform_for_navigation",
    "opencv_to_robot",
    "robot_to_opencv",
    "transform_obstacle_center",
    # 多帧融合
    "MultiframeFusion",
    "RegistrationConfig",
    "RegistrationResult",
    "FramePose",
    "FusionResult",
    # 运动估计
    "MotionEstimator",
    "MotionVector",
    "TrackedObject3D",
    "KalmanConfig",
    "create_tracked_object",
    # 目标跟踪
    "MultiObjectTracker",
    "TrackedObject",
    "TrackerConfig",
]

_LAZY_ATTRS = {
    # 点云处理
    "PointCloud": "aylm.tools.pointcloud_voxelizer",
    "PointCloudVoxelizer": "aylm.tools.pointcloud_voxelizer",
    "VoxelizerConfig": "aylm.tools.pointcloud_voxelizer",
    # 切片
    "PointCloudSlicer": "aylm.tools.pointcloud_slicer",
    "SlicerConfig": "aylm.tools.pointcloud_slicer",
    "SliceResult": "aylm.tools.pointcloud_slicer",
    "slice_pointcloud": "aylm.tools.pointcloud_slicer",
    # 语义类型
    "SemanticLabel": "aylm.tools.semantic_types",
    "Detection2D": "aylm.tools.semantic_types",
    "SemanticPointCloud": "aylm.tools.semantic_types",
    "COCO_TO_SEMANTIC": "aylm.tools.semantic_types",
    "SEMANTIC_COLORS": "aylm.tools.semantic_types",
    # 目标检测
    "ObjectDetector": "aylm.tools.object_detector",
    "DetectorConfig": "aylm.tools.object_detector",
    # 语义融合
    "SemanticFusion": "aylm.tools.semantic_fusion",
    "FusionConfig": "aylm.tools.semantic_fusion",
    "CameraIntrinsics": "aylm.tools.semantic_fusion",
    # 障碍物标记
    "ObstacleMarker": "aylm.tools.obstacle_marker",
    "ObstacleMarkerConfig": "aylm.tools.obstacle_marker",
    "ObstacleBox3D": "aylm.tools.obstacle_marker",
    "extract_obstacles": "aylm.tools.obstacle_marker",
    # 流水线
    "PipelineConfig": "aylm.tools.pipeline_processor",
    "PipelineProcessor": "aylm.tools.pipeline_processor",
    "PipelineStats": "aylm.tools.pipeline_processor",
    "run_pipeline": "aylm.tools.pipeline_processor",
    "run_pipeline_async": "aylm.tools.pipeline_processor",
    # 坐标转换
    "transform_for_navigation": "aylm.tools.coordinate_utils",
    "opencv_to_robot": "aylm.tools.coordinate_utils",
    "robot_to_opencv": "aylm.tools.coordinate_utils",
    "transform_obstacle_center": "aylm.tools.coordinate_utils",
    # 多帧融合
    "MultiframeFusion": "aylm.tools.multiframe_fusion",
    "RegistrationConfig": "aylm.tools.multiframe_fusion",
    "RegistrationResult": "aylm.tools.multiframe_fusion",
    "FramePose": "aylm.tools.multiframe_fusion",
    "FusionResult": "aylm.tools.multiframe_fusion",
    # 运动估计
    "MotionEstimator": "aylm.tools.motion_estimator",
    "MotionVector": "aylm.tools.motion_estimator",
    "TrackedObject3D": "aylm.tools.motion_estimator",
    "KalmanConfig": "aylm.tools.motion_estimator",
    "create_tracked_object": "aylm.tools.motion_estimator",
    # 目标跟踪
    "MultiObjectTracker": "aylm.tools.object_tracker",
    "TrackedObject": "aylm.tools.object_tracker",
    "TrackerConfig": "aylm.tools.object_tracker",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'aylm.tools' has no attribute {name!r}")
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())))
