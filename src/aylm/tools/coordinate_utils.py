"""坐标系转换工具模块。

提供OpenCV坐标系与机器人坐标系、ENU地理坐标系之间的转换功能。

坐标系定义：
- OpenCV: X右, Y下, Z前 (右手系)
- Robot: X前, Y左, Z上 (右手系)
- ENU: X东, Y北, Z上 (右手系)
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# OpenCV到机器人坐标系的旋转矩阵
# OpenCV: X右, Y下, Z前 -> Robot: X前, Y左, Z上
_OPENCV_TO_ROBOT = np.array(
    [
        [0, 0, 1],  # Robot X = OpenCV Z
        [-1, 0, 0],  # Robot Y = -OpenCV X
        [0, -1, 0],  # Robot Z = -OpenCV Y
    ],
    dtype=np.float64,
)

# 机器人到OpenCV坐标系的旋转矩阵（逆变换）
# Robot: X前, Y左, Z上 -> OpenCV: X右, Y下, Z前
_ROBOT_TO_OPENCV = np.array(
    [
        [0, -1, 0],  # OpenCV X = -Robot Y
        [0, 0, -1],  # OpenCV Y = -Robot Z
        [1, 0, 0],  # OpenCV Z = Robot X
    ],
    dtype=np.float64,
)

# OpenCV到ENU坐标系的旋转矩阵
# OpenCV: X右, Y下, Z前 -> ENU: X东, Y北, Z上
_OPENCV_TO_ENU = np.array(
    [
        [1, 0, 0],  # ENU X(东) = OpenCV X
        [0, 0, 1],  # ENU Y(北) = OpenCV Z
        [0, -1, 0],  # ENU Z(上) = -OpenCV Y
    ],
    dtype=np.float64,
)


def transform_opencv_to_robot(
    points: NDArray[np.floating],
) -> NDArray[np.float64]:
    """将OpenCV坐标系的点转换到机器人坐标系。

    Args:
        points: 形状为 (N, 3) 或 (3,) 的点坐标数组

    Returns:
        转换后的点坐标数组，形状与输入相同

    Raises:
        ValueError: 输入数组形状不正确
    """
    points = _validate_points(points)
    return _apply_transform(points, _OPENCV_TO_ROBOT)


def opencv_to_robot(points: NDArray) -> NDArray[np.float64]:
    """OpenCV坐标系转机器人坐标系（简化别名）。

    OpenCV (X右,Y下,Z前) -> 机器人 (X前,Y左,Z上)

    Args:
        points: 形状为 (N, 3) 或 (3,) 的点坐标数组

    Returns:
        转换后的点坐标数组
    """
    return transform_opencv_to_robot(points)


def robot_to_opencv(points: NDArray) -> NDArray[np.float64]:
    """机器人坐标系转OpenCV坐标系（逆变换）。

    机器人 (X前,Y左,Z上) -> OpenCV (X右,Y下,Z前)

    Args:
        points: 形状为 (N, 3) 或 (3,) 的点坐标数组

    Returns:
        转换后的点坐标数组

    Raises:
        ValueError: 输入数组形状不正确
    """
    points = _validate_points(points)
    return _apply_transform(points, _ROBOT_TO_OPENCV)


def transform_obstacle_center(
    center: NDArray | tuple[float, float, float],
    to_robot: bool = True,
) -> NDArray[np.float64]:
    """转换障碍物中心点坐标。

    Args:
        center: 障碍物中心点坐标 (x, y, z)
        to_robot: True 表示 OpenCV->Robot，False 表示 Robot->OpenCV

    Returns:
        转换后的中心点坐标
    """
    center_arr = np.asarray(center, dtype=np.float64)
    if to_robot:
        return opencv_to_robot(center_arr)
    return robot_to_opencv(center_arr)


def transform_opencv_to_enu(
    points: NDArray[np.floating],
) -> NDArray[np.float64]:
    """将OpenCV坐标系的点转换到ENU地理坐标系。

    Args:
        points: 形状为 (N, 3) 或 (3,) 的点坐标数组

    Returns:
        转换后的点坐标数组，形状与输入相同

    Raises:
        ValueError: 输入数组形状不正确
    """
    points = _validate_points(points)
    return _apply_transform(points, _OPENCV_TO_ENU)


def transform_for_navigation(input_path: str, output_path: str) -> int:
    """读取PLY文件并将坐标从OpenCV转换到ENU坐标系。

    Args:
        input_path: 输入PLY文件路径
        output_path: 输出PLY文件路径

    Returns:
        转换的点数量

    Raises:
        FileNotFoundError: 输入文件不存在
        ValueError: PLY文件格式错误
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 读取PLY文件
    header_lines, points, has_colors, colors = _read_ply(input_path)

    # 转换坐标
    transformed_points = transform_opencv_to_enu(points)

    # 写入PLY文件
    _write_ply(output_path, header_lines, transformed_points, has_colors, colors)

    return len(transformed_points)


def _validate_points(points: NDArray) -> NDArray[np.float64]:
    """验证并转换点坐标数组。"""
    points = np.asarray(points, dtype=np.float64)

    if points.ndim == 1:
        if points.shape[0] != 3:
            raise ValueError(f"单点必须有3个坐标，得到 {points.shape[0]}")
    elif points.ndim == 2:
        if points.shape[1] != 3:
            raise ValueError(f"点数组必须是 (N, 3) 形状，得到 {points.shape}")
    else:
        raise ValueError(f"点数组维度必须是1或2，得到 {points.ndim}")

    return points


def _apply_transform(
    points: NDArray[np.float64],
    rotation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """应用旋转变换到点坐标。"""
    if points.ndim == 1:
        return rotation @ points
    return (rotation @ points.T).T


def _read_ply(
    filepath: str,
) -> tuple[list[str], NDArray[np.float64], bool, list[list[int]] | None]:
    """读取PLY文件，返回头部、点坐标和颜色信息。"""
    header_lines: list[str] = []
    vertex_count = 0
    has_colors = False

    with open(filepath, encoding="utf-8") as f:
        # 解析头部
        for line in f:
            header_lines.append(line.rstrip("\n"))
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.startswith("property") and any(
                c in line for c in ["red", "green", "blue"]
            ):
                has_colors = True
            if line.strip() == "end_header":
                break

        if vertex_count == 0:
            raise ValueError("PLY文件中未找到顶点数据")

        # 读取顶点数据
        points: list[list[float]] = []
        colors: list[list[int]] | None = [] if has_colors else None

        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                raise ValueError("PLY文件数据不完整")
            values = line.split()
            points.append([float(values[0]), float(values[1]), float(values[2])])
            if has_colors and colors is not None and len(values) >= 6:
                colors.append([int(values[3]), int(values[4]), int(values[5])])

    return header_lines, np.array(points), has_colors, colors


def _write_ply(
    filepath: str,
    header_lines: list[str],
    points: NDArray[np.float64],
    has_colors: bool,
    colors: list[list[int]] | None,
) -> None:
    """写入PLY文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        # 写入头部
        for line in header_lines:
            f.write(line + "\n")

        # 写入顶点数据
        for i, point in enumerate(points):
            if has_colors and colors:
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{colors[i][0]} {colors[i][1]} {colors[i][2]}\n"
                )
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
