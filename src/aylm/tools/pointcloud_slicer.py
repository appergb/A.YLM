"""点云切片模块。

以摄像机位置为圆心，按半径裁切点云，减少体素化计算量。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


@dataclass
class SlicerConfig:
    """切片配置。"""

    radius: float = 10.0  # 切片半径（米）
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 圆心位置（摄像机位置）
    use_2d_distance: bool = True  # 是否只计算 X-Y 平面距离（忽略 Z）


@dataclass
class SliceResult:
    """切片结果。"""

    points: NDArray[np.float64]  # 切片后的点 N x 3
    colors: Optional[NDArray[np.float64]]  # 切片后的颜色 N x 3
    mask: NDArray[np.bool_]  # 保留点的掩码
    original_count: int  # 原始点数
    sliced_count: int  # 切片后点数


class PointCloudSlicer:
    """点云切片处理器。"""

    def __init__(self, config: Optional[SlicerConfig] = None):
        self.config = config or SlicerConfig()

    def slice_by_radius(
        self,
        points: NDArray[np.float64],
        colors: Optional[NDArray[np.float64]] = None,
        radius: Optional[float] = None,
        center: Optional[tuple[float, float, float]] = None,
    ) -> SliceResult:
        """
        按半径切片点云。

        Args:
            points: 点云坐标 N x 3
            colors: 点云颜色 N x 3（可选）
            radius: 切片半径（米），默认使用配置值
            center: 圆心位置，默认使用配置值

        Returns:
            SliceResult: 切片结果
        """
        radius = radius or self.config.radius
        center = center or self.config.center
        center_arr = np.array(center, dtype=np.float64)

        original_count = len(points)
        logger.info(
            f"开始切片: 原始点数={original_count}, 半径={radius}m, 圆心={center}"
        )

        # 计算距离
        if self.config.use_2d_distance:
            # 只计算 X-Y 平面距离（水平距离）
            diff = points[:, :2] - center_arr[:2]
            distances = np.sqrt((diff**2).sum(axis=1))
        else:
            # 计算 3D 距离
            diff = points - center_arr
            distances = np.sqrt((diff**2).sum(axis=1))

        # 创建掩码：保留距离 <= radius 的点
        mask = distances <= radius

        # 应用掩码
        sliced_points = points[mask]
        sliced_colors = colors[mask] if colors is not None else None
        sliced_count = len(sliced_points)

        reduction = (
            (1 - sliced_count / original_count) * 100 if original_count > 0 else 0
        )
        logger.info(
            f"切片完成: {original_count} → {sliced_count} 点 (减少 {reduction:.1f}%)"
        )

        return SliceResult(
            points=sliced_points,
            colors=sliced_colors,
            mask=mask,
            original_count=original_count,
            sliced_count=sliced_count,
        )

    def slice_ply(
        self,
        input_path: Path,
        output_path: Path,
        radius: Optional[float] = None,
        center: Optional[tuple[float, float, float]] = None,
    ) -> SliceResult:
        """
        切片 PLY 文件。

        Args:
            input_path: 输入 PLY 文件路径
            output_path: 输出 PLY 文件路径
            radius: 切片半径（米）
            center: 圆心位置

        Returns:
            SliceResult: 切片结果
        """
        logger.info(f"加载点云: {input_path}")
        plydata = PlyData.read(str(input_path))
        vertex = plydata["vertex"]

        # 提取点坐标
        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])

        # 提取颜色（如果有）
        colors = None
        if all(p in vertex.data.dtype.names for p in ["red", "green", "blue"]):
            colors = (
                np.column_stack(
                    [vertex["red"], vertex["green"], vertex["blue"]]
                ).astype(np.float64)
                / 255.0
            )

        # 执行切片
        result = self.slice_by_radius(points, colors, radius, center)

        # 保存切片后的点云
        self._save_ply(result.points, result.colors, output_path)

        return result

    def _save_ply(
        self,
        points: NDArray[np.float64],
        colors: Optional[NDArray[np.float64]],
        filepath: Path,
    ) -> None:
        """保存点云到 PLY 文件。"""
        logger.info(f"保存切片点云: {filepath}")

        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        if colors is not None:
            dtype.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

        data = np.zeros(len(points), dtype=dtype)
        data["x"], data["y"], data["z"] = points.T

        if colors is not None:
            colors_uint8 = (colors * 255).astype(np.uint8)
            data["red"], data["green"], data["blue"] = colors_uint8.T

        vertex = PlyElement.describe(data, "vertex")
        PlyData([vertex], text=True).write(str(filepath))
        logger.info(f"保存完成: {len(points)} 点")


def slice_pointcloud(
    input_path: str,
    output_path: str,
    radius: float = 10.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SliceResult:
    """
    便捷函数：切片点云。

    Args:
        input_path: 输入 PLY 文件路径
        output_path: 输出 PLY 文件路径
        radius: 切片半径（米），默认 10 米
        center: 圆心位置，默认原点（摄像机位置）

    Returns:
        SliceResult: 切片结果

    Example:
        >>> from aylm.tools.pointcloud_slicer import slice_pointcloud
        >>> result = slice_pointcloud("input.ply", "sliced.ply", radius=10.0)
        >>> print(f"切片后点数: {result.sliced_count}")
    """
    config = SlicerConfig(radius=radius, center=center)
    slicer = PointCloudSlicer(config)
    return slicer.slice_ply(Path(input_path), Path(output_path))
