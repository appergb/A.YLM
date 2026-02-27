"""A.YLM 工具模块。"""

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
from .semantic_fusion import (
    CameraIntrinsics,
    FusionConfig,
    SemanticFusion,
)
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
