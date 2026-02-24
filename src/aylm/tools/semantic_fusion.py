"""语义融合模块。

将 2D 语义检测结果融合到 3D 点云中，生成带语义标签的点云。
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .coordinate_utils import transform_opencv_to_robot
from .semantic_types import (
    SEMANTIC_COLORS,
    CameraIntrinsics,
    Detection2D,
    SemanticLabel,
    SemanticPointCloud,
)

# 一次性 plyfile 导入检查
try:
    from plyfile import PlyData, PlyElement

    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    PlyData = None
    PlyElement = None

# 模块级常量
DEFAULT_VOXEL_SIZE = 0.1  # 默认体素大小 10cm
DEFAULT_DENSITY_THRESHOLD = 3  # 默认密度阈值
CUBE_HALF_SIZE = 0.5  # 立方体半边长（归一化）

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """语义融合配置。"""

    min_confidence: float = 0.5  # 最小置信度阈值
    use_mask: bool = True  # 是否使用实例分割掩码（否则使用边界框）
    colorize_semantic: bool = True  # 是否根据语义标签着色
    blend_ratio: float = 0.5  # 语义颜色与原始颜色的混合比例 (0=原始, 1=语义)


class SemanticFusion:
    """语义融合器。

    将 2D 检测结果投影到 3D 点云，为每个点分配语义标签。
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """初始化语义融合器。

        Args:
            config: 融合配置，为 None 时使用默认配置
        """
        self.config = config or FusionConfig()
        logger.info(
            "语义融合器初始化完成，min_confidence=%.2f, use_mask=%s",
            self.config.min_confidence,
            self.config.use_mask,
        )

    def fuse(
        self,
        points: NDArray[np.float64],
        colors: Optional[NDArray[np.float64]],
        detections: list[Detection2D],
        image_shape: tuple[int, int],
        intrinsics: CameraIntrinsics,
    ) -> SemanticPointCloud:
        """将 2D 检测结果融合到 3D 点云。

        通过将 3D 点投影到 2D 图像平面，根据检测结果为每个点分配语义标签。

        投影公式：
            u = fx * X/Z + cx
            v = fy * Y/Z + cy

        Args:
            points: 3D 点云坐标，形状 (N, 3)
            colors: 点云颜色，形状 (N, 3)，范围 [0, 1]
            detections: 2D 检测结果列表
            image_shape: 图像尺寸 (height, width)
            intrinsics: 相机内参

        Returns:
            带语义标签的点云
        """
        n_points = len(points)
        height, width = image_shape

        # 初始化标签和置信度
        labels = np.full(n_points, SemanticLabel.UNKNOWN.value, dtype=np.uint8)
        confidences = np.zeros(n_points, dtype=np.float32)

        # 过滤低置信度检测
        valid_detections = [
            d for d in detections if d.confidence >= self.config.min_confidence
        ]
        logger.info(
            "有效检测数: %d/%d (min_confidence=%.2f)",
            len(valid_detections),
            len(detections),
            self.config.min_confidence,
        )

        if not valid_detections:
            logger.warning("没有有效的检测结果，所有点标记为 UNKNOWN")
            return SemanticPointCloud(
                points=points,
                colors=colors,
                labels=labels,
                confidences=confidences,
            )

        # 将 3D 点投影到 2D 图像平面
        # 过滤 Z <= 0 的点（在相机后面）
        z = points[:, 2]
        valid_z_mask = z > 0

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
            "投影统计: 总点数=%d, Z>0=%d, 在图像内=%d",
            n_points,
            valid_z_mask.sum(),
            in_image_mask.sum(),
        )

        # 按置信度降序处理检测结果（高置信度优先）
        sorted_detections = sorted(
            valid_detections, key=lambda d: d.confidence, reverse=True
        )

        for det in sorted_detections:
            if self.config.use_mask and det.mask is not None:
                # 使用实例分割掩码
                mask_h, mask_w = det.mask.shape

                # 如果 mask 尺寸与原图不同，缩放 mask 到原图尺寸
                # （正常情况下 object_detector 已经缩放过了，这里是备用方案）
                if mask_h != height or mask_w != width:
                    import cv2

                    # 使用双线性插值缩放，然后二值化（比最近邻更精确）
                    mask_float = cv2.resize(
                        det.mask.astype(np.float32),
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    mask_resized = (mask_float > 0.5).astype(bool)
                    logger.debug(
                        "Mask 缩放: %dx%d -> %dx%d", mask_h, mask_w, height, width
                    )
                else:
                    mask_resized = det.mask

                # 只对在图像范围内的点进行掩码查询
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

            # 只更新置信度更高的点
            update_mask = point_mask & (det.confidence > confidences)
            labels[update_mask] = det.semantic_label.value
            confidences[update_mask] = det.confidence

            logger.debug(
                "检测 %s: 匹配点数=%d, 更新点数=%d",
                det.semantic_label.name,
                point_mask.sum(),
                update_mask.sum(),
            )

        # 统计标签分布
        label_counts = {}
        for label in SemanticLabel:
            count = int((labels == label.value).sum())
            if count > 0:
                label_counts[label.name] = count
        logger.info("标签分布: %s", label_counts)

        # 处理颜色
        output_colors = colors
        if colors is not None and self.config.colorize_semantic:
            output_colors = self._blend_semantic_colors(colors, labels)

        return SemanticPointCloud(
            points=points,
            colors=output_colors,
            labels=labels,
            confidences=confidences,
        )

    def _blend_semantic_colors(
        self,
        original_colors: NDArray[np.float64],
        labels: NDArray[np.uint8],
    ) -> NDArray[np.float64]:
        """混合原始颜色和语义颜色。

        Args:
            original_colors: 原始颜色，形状 (N, 3)，范围 [0, 1]
            labels: 语义标签，形状 (N,)

        Returns:
            混合后的颜色，形状 (N, 3)
        """
        # 生成语义颜色
        semantic_colors = np.zeros_like(original_colors)
        for label in SemanticLabel:
            mask = labels == label.value
            if mask.any():
                semantic_colors[mask] = SEMANTIC_COLORS[label]

        # 混合颜色
        ratio = self.config.blend_ratio
        blended = (1 - ratio) * original_colors + ratio * semantic_colors

        return np.clip(blended, 0.0, 1.0)

    def save_semantic_ply(
        self,
        semantic_pc: SemanticPointCloud,
        output_path: Path | str,
        include_semantic_colors: bool = True,
        transform_to_robot: bool = False,
    ) -> None:
        """保存带语义标签的 PLY 文件。"""
        if not HAS_PLYFILE:
            raise ImportError("需要安装 plyfile 库: pip install plyfile")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_points = len(semantic_pc.points)

        # 获取点坐标，可选转换到机器人坐标系
        points = semantic_pc.points
        if transform_to_robot:
            points = transform_opencv_to_robot(points)
            logger.debug("坐标已从 OpenCV 转换到机器人坐标系")

        # 确定颜色
        if include_semantic_colors:
            colors = semantic_pc.colorize_by_semantic()
        elif semantic_pc.colors is not None:
            colors = semantic_pc.colors
        else:
            colors = np.ones((n_points, 3), dtype=np.float64) * 0.5

        # 转换颜色到 0-255 范围
        colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

        # 构建结构化数组
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("semantic_label", "u1"),
            ("semantic_confidence", "f4"),
        ]

        vertex_data = np.zeros(n_points, dtype=dtype)
        vertex_data["x"] = points[:, 0].astype(np.float32)
        vertex_data["y"] = points[:, 1].astype(np.float32)
        vertex_data["z"] = points[:, 2].astype(np.float32)
        vertex_data["red"] = colors_uint8[:, 0]
        vertex_data["green"] = colors_uint8[:, 1]
        vertex_data["blue"] = colors_uint8[:, 2]
        vertex_data["semantic_label"] = semantic_pc.labels
        vertex_data["semantic_confidence"] = semantic_pc.confidences

        # 创建 PLY 元素并保存
        vertex_element = PlyElement.describe(vertex_data, "vertex")
        ply_data = PlyData([vertex_element], text=False)
        ply_data.write(str(output_path))

        logger.info("语义点云已保存: %s (%d 点)", output_path, n_points)

    def save_navigation_ply(
        self,
        semantic_pc: SemanticPointCloud,
        output_path: Path | str,
        voxel_size: float = DEFAULT_VOXEL_SIZE,
        min_points_per_voxel: int = 1,
    ) -> None:
        """保存导航用体素网格 PLY 文件（真正的立方体几何体）。"""
        if not HAS_PLYFILE:
            raise ImportError("需要安装 plyfile 库: pip install plyfile")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换坐标到机器人坐标系
        robot_points = transform_opencv_to_robot(semantic_pc.points)

        # 获取颜色（使用语义着色）
        colors = semantic_pc.colorize_by_semantic()

        # 体素化：基于密度阈值生成实体方块信息
        voxel_centers, voxel_colors, _voxel_labels, _, _ = self._create_solid_voxels(
            robot_points,
            colors,
            semantic_pc.labels,
            semantic_pc.confidences,
            voxel_size,
            min_points_per_voxel,
        )

        n_voxels = len(voxel_centers)
        if n_voxels == 0:
            logger.warning("没有生成任何实体方块，请检查点云数据或降低密度阈值")
            return

        # 为每个体素生成立方体几何体
        vertices, faces, vertex_colors = self._generate_cube_mesh(
            voxel_centers, voxel_colors, voxel_size
        )

        # 构建顶点数据
        vertex_dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        vertex_data = np.zeros(len(vertices), dtype=vertex_dtype)
        vertex_data["x"] = vertices[:, 0].astype(np.float32)
        vertex_data["y"] = vertices[:, 1].astype(np.float32)
        vertex_data["z"] = vertices[:, 2].astype(np.float32)
        vertex_data["red"] = vertex_colors[:, 0]
        vertex_data["green"] = vertex_colors[:, 1]
        vertex_data["blue"] = vertex_colors[:, 2]

        # 构建面数据
        face_dtype = [("vertex_indices", "i4", (3,))]
        face_data = np.zeros(len(faces), dtype=face_dtype)
        face_data["vertex_indices"] = faces

        # 添加注释
        comments = [
            "Navigation voxel grid for robot",
            "Coordinate system: Robot (X forward, Y left, Z up)",
            "Each cube is a solid voxel",
        ]

        vertex_element = PlyElement.describe(vertex_data, "vertex")
        face_element = PlyElement.describe(face_data, "face")
        ply_data = PlyData(
            [vertex_element, face_element], text=False, comments=comments
        )
        ply_data.write(str(output_path))

        original_count = len(semantic_pc.points)
        logger.info(
            "导航体素网格已保存: %s (%d 立方体, %d 顶点, %d 面, 原始 %d 点)",
            output_path,
            n_voxels,
            len(vertices),
            len(faces),
            original_count,
        )

    def _generate_cube_mesh(
        self,
        centers: NDArray[np.float64],
        colors: NDArray[np.float64],
        voxel_size: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.uint8]]:
        """为每个体素中心生成立方体网格。

        每个立方体有 8 个顶点和 12 个三角面（6个面，每面2个三角形）。

        Args:
            centers: 体素中心坐标 (N, 3)
            colors: 体素颜色 (N, 3)，范围 [0, 1]
            voxel_size: 体素大小

        Returns:
            (vertices, faces, vertex_colors)
        """
        n_cubes = len(centers)
        half = voxel_size / 2

        # 单个立方体的 8 个顶点偏移（相对于中心）
        cube_vertices = np.array(
            [
                [-half, -half, -half],  # 0
                [+half, -half, -half],  # 1
                [+half, +half, -half],  # 2
                [-half, +half, -half],  # 3
                [-half, -half, +half],  # 4
                [+half, -half, +half],  # 5
                [+half, +half, +half],  # 6
                [-half, +half, +half],  # 7
            ],
            dtype=np.float64,
        )

        # 单个立方体的 12 个三角面（顶点索引）
        cube_faces = np.array(
            [
                # 底面 (z-)
                [0, 2, 1],
                [0, 3, 2],
                # 顶面 (z+)
                [4, 5, 6],
                [4, 6, 7],
                # 前面 (x+)
                [1, 2, 6],
                [1, 6, 5],
                # 后面 (x-)
                [0, 4, 7],
                [0, 7, 3],
                # 右面 (y+)
                [2, 3, 7],
                [2, 7, 6],
                # 左面 (y-)
                [0, 1, 5],
                [0, 5, 4],
            ],
            dtype=np.int32,
        )

        # 生成所有顶点
        all_vertices = np.zeros((n_cubes * 8, 3), dtype=np.float64)
        for i, center in enumerate(centers):
            all_vertices[i * 8 : (i + 1) * 8] = center + cube_vertices

        # 生成所有面（调整顶点索引）
        all_faces = np.zeros((n_cubes * 12, 3), dtype=np.int32)
        for i in range(n_cubes):
            all_faces[i * 12 : (i + 1) * 12] = cube_faces + i * 8

        # 生成顶点颜色（每个立方体的 8 个顶点颜色相同）
        colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        all_colors = np.repeat(colors_uint8, 8, axis=0)

        return all_vertices, all_faces, all_colors

    def _create_solid_voxels(
        self,
        points: NDArray[np.float64],
        colors: NDArray[np.float64],
        labels: NDArray[np.uint8],
        confidences: NDArray[np.float32],
        voxel_size: float,
        min_points: int,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.uint8],
        NDArray[np.float32],
        NDArray[np.uint16],
    ]:
        """将点云转换为实体方块网格。"""
        # 计算每个点所属的体素索引
        voxel_indices = np.floor(points / voxel_size).astype(np.int64)

        # 使用 defaultdict 聚合每个体素的点索引
        voxel_dict: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, idx in enumerate(voxel_indices):
            voxel_dict[tuple(int(x) for x in idx)].append(i)

        # 过滤：只保留点数达到阈值的体素
        solid_voxels = {k: v for k, v in voxel_dict.items() if len(v) >= min_points}

        n_solid = len(solid_voxels)
        if n_solid == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.uint8),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.uint16),
            )

        # 为每个实体方块计算属性
        voxel_centers = np.zeros((n_solid, 3), dtype=np.float64)
        voxel_colors = np.zeros((n_solid, 3), dtype=np.float64)
        voxel_labels = np.zeros(n_solid, dtype=np.uint8)
        voxel_confidences = np.zeros(n_solid, dtype=np.float32)
        voxel_densities = np.zeros(n_solid, dtype=np.uint16)

        for i, (voxel_idx, point_indices) in enumerate(solid_voxels.items()):
            point_indices_arr = np.array(point_indices)

            # 体素中心坐标（网格对齐）
            voxel_centers[i] = (
                np.array(voxel_idx, dtype=np.float64) + 0.5
            ) * voxel_size

            # 平均颜色
            voxel_colors[i] = colors[point_indices_arr].mean(axis=0)

            # 投票选择主要标签
            voxel_point_labels = labels[point_indices_arr]
            unique_labels, counts = np.unique(voxel_point_labels, return_counts=True)
            dominant_label = unique_labels[counts.argmax()]
            voxel_labels[i] = dominant_label

            # 该标签的平均置信度
            label_mask = voxel_point_labels == dominant_label
            voxel_confidences[i] = confidences[point_indices_arr][label_mask].mean()

            # 点密度
            voxel_densities[i] = min(len(point_indices), 65535)

        return (
            voxel_centers,
            voxel_colors,
            voxel_labels,
            voxel_confidences,
            voxel_densities,
        )

    @staticmethod
    def load_semantic_ply(filepath: Path | str) -> SemanticPointCloud:
        """加载带语义标签的 PLY 文件。"""
        if not HAS_PLYFILE:
            raise ImportError("需要安装 plyfile 库: pip install plyfile")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        ply_data = PlyData.read(str(filepath))
        vertex = ply_data["vertex"]

        # 读取坐标
        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(
            np.float64
        )

        # 读取颜色（如果存在）
        colors = None
        if "red" in vertex.data.dtype.names:
            colors = (
                np.column_stack(
                    [vertex["red"], vertex["green"], vertex["blue"]]
                ).astype(np.float64)
                / 255.0
            )

        # 读取语义标签（如果存在）
        if "semantic_label" in vertex.data.dtype.names:
            labels = np.array(vertex["semantic_label"], dtype=np.uint8)
        else:
            labels = np.zeros(len(points), dtype=np.uint8)
            logger.warning("PLY 文件不包含 semantic_label 字段，使用默认值 UNKNOWN")

        # 读取置信度（如果存在）
        if "semantic_confidence" in vertex.data.dtype.names:
            confidences = np.array(vertex["semantic_confidence"], dtype=np.float32)
        else:
            confidences = np.zeros(len(points), dtype=np.float32)
            logger.warning("PLY 文件不包含 semantic_confidence 字段，使用默认值 0")

        logger.info("语义点云已加载: %s (%d 点)", filepath, len(points))

        return SemanticPointCloud(
            points=points,
            colors=colors,
            labels=labels,
            confidences=confidences,
        )
