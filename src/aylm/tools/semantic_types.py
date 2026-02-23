"""语义类型定义模块。

定义语义标签、检测结果和语义点云的数据结构。
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class SemanticLabel(IntEnum):
    """语义标签枚举。"""

    UNKNOWN = 0
    PERSON = 1
    VEHICLE = 2  # 车辆（汽车、卡车、公交车）
    BICYCLE = 3  # 自行车/摩托车
    ANIMAL = 4  # 动物
    OBSTACLE = 5  # 其他障碍物
    GROUND = 6  # 地面
    BACKGROUND = 7  # 背景


# COCO 类别到语义标签的映射
COCO_TO_SEMANTIC: dict[int, SemanticLabel] = {
    0: SemanticLabel.PERSON,  # person
    1: SemanticLabel.BICYCLE,  # bicycle
    2: SemanticLabel.VEHICLE,  # car
    3: SemanticLabel.BICYCLE,  # motorcycle
    5: SemanticLabel.VEHICLE,  # bus
    7: SemanticLabel.VEHICLE,  # truck
    14: SemanticLabel.ANIMAL,  # bird
    15: SemanticLabel.ANIMAL,  # cat
    16: SemanticLabel.ANIMAL,  # dog
    17: SemanticLabel.ANIMAL,  # horse
}


# 语义标签对应的颜色 (RGB, 0-1)
SEMANTIC_COLORS: dict[SemanticLabel, tuple[float, float, float]] = {
    SemanticLabel.UNKNOWN: (0.5, 0.5, 0.5),  # 灰色
    SemanticLabel.PERSON: (1.0, 0.2, 0.2),  # 红色
    SemanticLabel.VEHICLE: (0.2, 0.4, 1.0),  # 蓝色
    SemanticLabel.BICYCLE: (1.0, 0.6, 0.0),  # 橙色
    SemanticLabel.ANIMAL: (1.0, 1.0, 0.0),  # 黄色
    SemanticLabel.OBSTACLE: (0.8, 0.0, 0.8),  # 紫色
    SemanticLabel.GROUND: (0.4, 0.8, 0.4),  # 浅绿
    SemanticLabel.BACKGROUND: (0.2, 0.8, 0.2),  # 绿色
}


@dataclass
class Detection2D:
    """2D 检测结果。"""

    bbox: NDArray[np.float32]  # [x1, y1, x2, y2]
    mask: Optional[NDArray[np.bool_]]  # H x W 布尔掩码
    class_id: int  # COCO 类别 ID
    confidence: float  # 置信度
    semantic_label: SemanticLabel  # 语义标签

    @property
    def area(self) -> float:
        """边界框面积。"""
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


@dataclass
class SemanticPointCloud:
    """带语义标签的点云。"""

    points: NDArray[np.float64]  # N x 3
    colors: Optional[NDArray[np.float64]]  # N x 3
    labels: NDArray[np.uint8]  # N, 语义标签
    confidences: NDArray[np.float32]  # N, 置信度

    def get_points_by_label(self, label: SemanticLabel) -> NDArray[np.float64]:
        """获取指定标签的点。"""
        mask = self.labels == label.value
        return self.points[mask]

    def colorize_by_semantic(self) -> NDArray[np.float64]:
        """根据语义标签生成颜色。"""
        colors = np.zeros((len(self.points), 3), dtype=np.float64)
        for label in SemanticLabel:
            mask = self.labels == label.value
            if mask.any():
                colors[mask] = SEMANTIC_COLORS[label]
        return colors

    def get_label_counts(self) -> dict[SemanticLabel, int]:
        """统计每个标签的点数。"""
        counts = {}
        for label in SemanticLabel:
            count = int((self.labels == label.value).sum())
            if count > 0:
                counts[label] = count
        return counts


@dataclass
class CameraIntrinsics:
    """相机内参（用于 3D→2D 投影）。

    投影公式：
        u = fx * X/Z + cx
        v = fy * Y/Z + cy
    """

    fx: float  # x 方向焦距（像素）
    fy: float  # y 方向焦距（像素）
    cx: float  # 主点 x 坐标（像素）
    cy: float  # 主点 y 坐标（像素）

    @classmethod
    def from_matrix(cls, K: NDArray[np.float64]) -> "CameraIntrinsics":
        """从 3x3 内参矩阵创建。

        Args:
            K: 3x3 相机内参矩阵
               [[fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]]

        Returns:
            CameraIntrinsics 实例
        """
        return cls(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

    @classmethod
    def from_focal_length(
        cls, focal_length: float, image_width: int, image_height: int
    ) -> "CameraIntrinsics":
        """从焦距和图像尺寸创建（假设主点在图像中心）。

        Args:
            focal_length: 焦距（像素）
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            CameraIntrinsics 实例
        """
        return cls(
            fx=focal_length,
            fy=focal_length,
            cx=image_width / 2,
            cy=image_height / 2,
        )
