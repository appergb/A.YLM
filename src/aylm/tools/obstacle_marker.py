"""3D障碍物边界框标记模块。

从 2D 检测结果和 3D 点云中提取障碍物的 3D 边界框，供导航系统使用。

核心思路：
1. 使用 YOLO 的 2D 检测结果（每个检测 = 一个障碍物）
2. 将 3D 点投影到 2D 图像平面
3. 根据 2D 掩码/边界框找到对应的 3D 点
4. 计算每个检测目标的 3D 边界框
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .semantic_types import (
    CameraIntrinsics,
    Detection2D,
    SemanticLabel,
    SemanticPointCloud,
)

logger = logging.getLogger(__name__)

# 需要标记为障碍物的语义标签
OBSTACLE_LABELS = {
    SemanticLabel.PERSON,
    SemanticLabel.VEHICLE,
    SemanticLabel.BICYCLE,
    SemanticLabel.ANIMAL,
    SemanticLabel.OBSTACLE,
}

# 可运动障碍物标签（人、车辆、自行车、动物都是可运动的）
MOVABLE_LABELS = {
    SemanticLabel.PERSON,
    SemanticLabel.VEHICLE,
    SemanticLabel.BICYCLE,
    SemanticLabel.ANIMAL,
}

# 标签的中文描述
LABEL_DESCRIPTIONS = {
    SemanticLabel.PERSON: "行人",
    SemanticLabel.VEHICLE: "车辆",
    SemanticLabel.BICYCLE: "自行车/电动车",
    SemanticLabel.ANIMAL: "动物",
    SemanticLabel.OBSTACLE: "静态障碍物",
    SemanticLabel.UNKNOWN: "未知物体",
}


@dataclass
class ObstacleBox3D:
    """3D障碍物边界框。

    坐标系说明：
    - OpenCV 坐标系: X右, Y下, Z前 (相机坐标系)
    - 机器人坐标系: X前, Y左, Z上 (ROS/导航标准)

    转换关系:
        X_robot = Z_cv
        Y_robot = -X_cv
        Z_robot = -Y_cv
    """

    center: tuple[float, float, float]  # 中心点 (x, y, z) - OpenCV 坐标系
    dimensions: tuple[float, float, float]  # 尺寸 (width, height, depth)
    label: SemanticLabel  # 语义标签
    confidence: float  # 平均置信度
    point_indices: NDArray[np.int64] = field(repr=False)  # 属于该障碍物的点索引
    track_id: Optional[int] = None  # 跟踪 ID（跨帧关联）
    frame_id: Optional[int] = None  # 帧 ID（时序关联）
    timestamp: Optional[float] = None  # 时间戳（秒）
    velocity: Optional[tuple[float, float, float]] = None  # 速度向量 (vx, vy, vz) m/s
    motion_vector: Optional[tuple[float, float, float]] = None  # 运动矢量（帧间位移）

    @property
    def is_movable(self) -> bool:
        """是否为可运动障碍物。"""
        return self.label in MOVABLE_LABELS

    @property
    def min_corner(self) -> tuple[float, float, float]:
        """最小角点 (OpenCV 坐标系)。"""
        return (
            self.center[0] - self.dimensions[0] / 2,
            self.center[1] - self.dimensions[1] / 2,
            self.center[2] - self.dimensions[2] / 2,
        )

    @property
    def max_corner(self) -> tuple[float, float, float]:
        """最大角点 (OpenCV 坐标系)。"""
        return (
            self.center[0] + self.dimensions[0] / 2,
            self.center[1] + self.dimensions[1] / 2,
            self.center[2] + self.dimensions[2] / 2,
        )

    @property
    def center_robot(self) -> tuple[float, float, float]:
        """中心点 (机器人坐标系)。

        转换: OpenCV (X右,Y下,Z前) -> 机器人 (X前,Y左,Z上)
        """
        return self._cv_to_robot(self.center)

    @property
    def dimensions_robot(self) -> tuple[float, float, float]:
        """尺寸 (机器人坐标系)。

        OpenCV: (width_x, height_y, depth_z)
        机器人: (depth_x, width_y, height_z)
        """
        # dimensions 是绝对值，只需重新排列
        return (self.dimensions[2], self.dimensions[0], self.dimensions[1])

    @staticmethod
    def _cv_to_robot(
        point: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """OpenCV 坐标系转机器人坐标系。

        OpenCV: X右, Y下, Z前
        机器人: X前, Y左, Z上

        转换:
            X_robot = Z_cv
            Y_robot = -X_cv
            Z_robot = -Y_cv
        """
        x_cv, y_cv, z_cv = point
        return (z_cv, -x_cv, -y_cv)

    def to_dict(self) -> dict:
        """转换为字典格式（用于导航系统）。

        输出包含两种坐标系的中心点：
        - center_cv: OpenCV 坐标系 (X右,Y下,Z前)
        - center_robot: 机器人坐标系 (X前,Y左,Z上)
        """
        result = {
            "type": "可运动障碍物" if self.is_movable else "静态障碍物",
            "category": LABEL_DESCRIPTIONS.get(self.label, "未知"),
            "movable": self.is_movable,
            "center_cv": list(self.center),
            "center_robot": list(self.center_robot),
            "dimensions_cv": list(self.dimensions),
            "dimensions_robot": list(self.dimensions_robot),
            "confidence": float(self.confidence),
            "point_count": len(self.point_indices),
            # 保留原始标签信息供调试
            "_label": self.label.name,
            "_label_id": int(self.label.value),
        }

        # 添加跟踪和时序信息
        if self.track_id is not None:
            result["track_id"] = self.track_id
        if self.frame_id is not None:
            result["frame_id"] = self.frame_id
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp

        # 添加运动信息
        if self.velocity is not None:
            result["velocity_cv"] = list(self.velocity)
            result["velocity_robot"] = list(self._cv_to_robot(self.velocity))
        if self.motion_vector is not None:
            result["motion_vector_cv"] = list(self.motion_vector)
            result["motion_vector_robot"] = list(self._cv_to_robot(self.motion_vector))

        return result

    def get_box_vertices(self) -> NDArray[np.float64]:
        """获取边界框的8个顶点，用于可视化。"""
        min_c = self.min_corner
        max_c = self.max_corner
        return np.array(
            [
                [min_c[0], min_c[1], min_c[2]],
                [max_c[0], min_c[1], min_c[2]],
                [max_c[0], max_c[1], min_c[2]],
                [min_c[0], max_c[1], min_c[2]],
                [min_c[0], min_c[1], max_c[2]],
                [max_c[0], min_c[1], max_c[2]],
                [max_c[0], max_c[1], max_c[2]],
                [min_c[0], max_c[1], max_c[2]],
            ],
            dtype=np.float64,
        )


@dataclass
class ObstacleMarkerConfig:
    """障碍物标记配置。"""

    min_points: int = 10  # 最小点数阈值，少于此数的检测被忽略
    min_confidence: float = 0.5  # 最小置信度阈值


class ObstacleMarker:
    """3D障碍物标记器。

    基于 2D 检测结果提取 3D 边界框，每个 YOLO 检测 = 一个障碍物。
    """

    def __init__(self, config: Optional[ObstacleMarkerConfig] = None):
        self.config = config or ObstacleMarkerConfig()

    def extract_obstacles_from_detections(
        self,
        points: NDArray[np.float64],
        detections: list[Detection2D],
        image_shape: tuple[int, int],
        intrinsics: CameraIntrinsics,
    ) -> list[ObstacleBox3D]:
        """
        从 2D 检测结果提取 3D 障碍物边界框。

        每个 YOLO 检测结果对应一个障碍物（不做额外聚类）。

        Args:
            points: 3D 点云坐标，形状 (N, 3)
            detections: YOLO 2D 检测结果列表
            image_shape: 图像尺寸 (height, width)
            intrinsics: 相机内参

        Returns:
            障碍物边界框列表
        """
        obstacles: list[ObstacleBox3D] = []
        height, width = image_shape
        n_points = len(points)

        # 将 3D 点投影到 2D 图像平面
        z = points[:, 2]
        valid_z_mask = z > 0  # 只处理相机前方的点

        # 投影公式: u = fx * X/Z + cx, v = fy * Y/Z + cy
        u = np.zeros(n_points, dtype=np.float64)
        v = np.zeros(n_points, dtype=np.float64)

        u[valid_z_mask] = (
            intrinsics.fx * points[valid_z_mask, 0] / z[valid_z_mask] + intrinsics.cx
        )
        v[valid_z_mask] = (
            intrinsics.fy * points[valid_z_mask, 1] / z[valid_z_mask] + intrinsics.cy
        )

        # 检查投影点是否在图像范围内
        in_image_mask = valid_z_mask & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        # 转换为整数像素坐标
        u_int = u.astype(np.int32)
        v_int = v.astype(np.int32)

        logger.debug(
            f"投影统计: 总点数={n_points}, Z>0={valid_z_mask.sum()}, "
            f"在图像内={in_image_mask.sum()}"
        )

        # 过滤低置信度检测
        valid_detections = [
            d
            for d in detections
            if d.confidence >= self.config.min_confidence
            and d.semantic_label in OBSTACLE_LABELS
        ]

        logger.info(f"有效障碍物检测数: {len(valid_detections)}/{len(detections)}")

        # 为每个检测结果创建一个障碍物
        for det in valid_detections:
            # 找到属于该检测的 3D 点
            if det.mask is not None:
                # 使用实例分割掩码
                mask_h, mask_w = det.mask.shape

                # 如果 mask 尺寸与原图不同，缩放 mask
                if mask_h != height or mask_w != width:
                    import cv2

                    mask_float = cv2.resize(
                        det.mask.astype(np.float32),
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    mask_resized = (mask_float > 0.5).astype(bool)
                else:
                    mask_resized = det.mask

                # 找到掩码内的 3D 点
                point_mask = np.zeros(n_points, dtype=bool)
                valid_indices = np.where(in_image_mask)[0]
                if len(valid_indices) > 0:
                    v_valid = v_int[valid_indices]
                    u_valid = u_int[valid_indices]
                    mask_values = mask_resized[v_valid, u_valid]
                    point_mask[valid_indices] = mask_values
            else:
                # 使用边界框
                x1, y1, x2, y2 = det.bbox
                point_mask = in_image_mask & (u >= x1) & (u < x2) & (v >= y1) & (v < y2)

            # 获取匹配的点索引
            point_indices = np.where(point_mask)[0]

            if len(point_indices) < self.config.min_points:
                logger.debug(
                    f"跳过 {det.semantic_label.name}: 点数不足 "
                    f"({len(point_indices)} < {self.config.min_points})"
                )
                continue

            # 计算 3D 边界框
            obstacle_points = points[point_indices]
            center, dimensions = self._compute_bounding_box(obstacle_points)

            obstacle = ObstacleBox3D(
                center=center,
                dimensions=dimensions,
                label=det.semantic_label,
                confidence=det.confidence,
                point_indices=point_indices,
            )
            obstacles.append(obstacle)

            logger.info(
                f"障碍物 {len(obstacles)}: {LABEL_DESCRIPTIONS.get(det.semantic_label, '未知')} "
                f"(置信度={det.confidence:.2f}, 点数={len(point_indices)}, "
                f"中心={[f'{c:.2f}' for c in center]})"
            )

        logger.info(f"共提取 {len(obstacles)} 个障碍物边界框")
        return obstacles

    def extract_obstacle_boxes(
        self, semantic_pc: SemanticPointCloud
    ) -> list[ObstacleBox3D]:
        """
        从语义点云中提取障碍物边界框（旧方法，基于聚类）。

        注意：推荐使用 extract_obstacles_from_detections() 方法，
        它直接使用 YOLO 检测结果，障碍物数量更准确。

        Args:
            semantic_pc: 带语义标签的点云

        Returns:
            障碍物边界框列表
        """
        logger.warning(
            "使用旧的聚类方法提取障碍物，建议使用 extract_obstacles_from_detections()"
        )
        obstacles: list[ObstacleBox3D] = []

        for label in OBSTACLE_LABELS:
            # 获取该标签的点索引
            mask = semantic_pc.labels == label.value
            indices = np.where(mask)[0]

            if len(indices) < self.config.min_points:
                continue

            points = semantic_pc.points[indices]
            confidences = semantic_pc.confidences[indices]

            # 计算整体边界框（不再聚类）
            center, dimensions = self._compute_bounding_box(points)
            avg_confidence = float(confidences.mean())

            obstacle = ObstacleBox3D(
                center=center,
                dimensions=dimensions,
                label=label,
                confidence=avg_confidence,
                point_indices=indices,
            )
            obstacles.append(obstacle)

            logger.debug(
                f"检测到 {label.name}: 中心={center}, "
                f"尺寸={dimensions}, 点数={len(indices)}"
            )

        logger.info(f"共提取 {len(obstacles)} 个障碍物边界框")
        return obstacles

    def _compute_bounding_box(
        self, points: NDArray[np.float64]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        计算点集的轴对齐边界框 (AABB)。

        Args:
            points: 点坐标 N x 3

        Returns:
            (center, dimensions): 中心点和尺寸
        """
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)

        center = tuple((min_pt + max_pt) / 2)
        dimensions = tuple(max_pt - min_pt)

        return center, dimensions  # type: ignore

    def highlight_obstacles(
        self,
        semantic_pc: SemanticPointCloud,
        obstacles: list[ObstacleBox3D],
        highlight_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> NDArray[np.float64]:
        """
        生成高亮障碍物的颜色数组。

        Args:
            semantic_pc: 语义点云
            obstacles: 障碍物列表
            highlight_color: 高亮颜色 (RGB, 0-1)

        Returns:
            颜色数组 N x 3
        """
        # 使用原始颜色或语义颜色
        if semantic_pc.colors is not None:
            colors = semantic_pc.colors.copy()
        else:
            colors = semantic_pc.colorize_by_semantic()

        # 高亮障碍物点
        for obstacle in obstacles:
            colors[obstacle.point_indices] = highlight_color

        return colors

    def export_to_json(
        self,
        obstacles: list[ObstacleBox3D],
        output_path: Path,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        导出障碍物列表为 JSON 格式（用于导航系统）。

        Args:
            obstacles: 障碍物列表
            output_path: 输出文件路径
            frame_id: 帧 ID（可选，用于时序关联）
            timestamp: 时间戳（可选，秒）
        """
        # 统计可运动和静态障碍物数量
        movable_count = sum(1 for obs in obstacles if obs.is_movable)
        static_count = len(obstacles) - movable_count

        # 统计有跟踪 ID 的障碍物
        tracked_count = sum(1 for obs in obstacles if obs.track_id is not None)

        data = {
            "coordinate_systems": {
                "cv": {
                    "description": "OpenCV/相机坐标系",
                    "axes": "X右, Y下, Z前",
                },
                "robot": {
                    "description": "机器人/ROS坐标系",
                    "axes": "X前, Y左, Z上",
                },
                "transform": "X_robot=Z_cv, Y_robot=-X_cv, Z_robot=-Y_cv",
            },
            "total_count": len(obstacles),
            "movable_count": movable_count,
            "static_count": static_count,
            "tracked_count": tracked_count,
            "obstacles": [obs.to_dict() for obs in obstacles],
        }

        # 添加时序元数据
        if frame_id is not None:
            data["frame_id"] = frame_id
        if timestamp is not None:
            data["timestamp"] = timestamp

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"导出 {len(obstacles)} 个障碍物到 {output_path}")


def extract_obstacles(
    semantic_pc: SemanticPointCloud,
    min_points: int = 10,
    min_confidence: float = 0.5,
) -> list[ObstacleBox3D]:
    """
    便捷函数：从语义点云提取障碍物（旧方法）。

    注意：此函数使用旧的聚类方法，建议使用 ObstacleMarker.extract_obstacles_from_detections()

    Args:
        semantic_pc: 带语义标签的点云
        min_points: 最小点数阈值
        min_confidence: 最小置信度阈值

    Returns:
        障碍物边界框列表

    Example:
        >>> from aylm.tools.obstacle_marker import extract_obstacles
        >>> obstacles = extract_obstacles(semantic_pc)
        >>> for obs in obstacles:
        ...     print(f"{obs.label.name}: {obs.center}")
    """
    config = ObstacleMarkerConfig(min_points=min_points, min_confidence=min_confidence)
    marker = ObstacleMarker(config)
    return marker.extract_obstacle_boxes(semantic_pc)
