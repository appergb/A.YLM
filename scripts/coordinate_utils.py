#!/usr/bin/env python3
"""SHARP 3D高斯模型坐标转换工具.

用于智能设备导航应用的坐标系对齐

SHARP默认坐标系: OpenCV标准 (x右, y下, z前)
智能设备导航通常需要: x前, y左, z上 (或 x右, y前, z上)

作者: TRIP(appergb)
项目参与者: closer, true
个人研发项目
"""

from pathlib import Path

import numpy as np
import plyfile


def load_ply_gaussians(ply_path):
    """加载PLY格式的高斯点云数据."""
    plydata = plyfile.PlyData.read(ply_path)

    # 提取顶点数据
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    scales = np.vstack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T
    rotations = np.vstack(
        [vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]
    ).T
    colors = np.vstack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).T
    opacities = vertices["opacity"]

    return {
        "positions": positions,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities,
    }


def save_ply_gaussians(ply_path, gaussians_data):
    """保存PLY格式的高斯点云数据."""
    positions = gaussians_data["positions"]
    scales = gaussians_data["scales"]
    rotations = gaussians_data["rotations"]
    colors = gaussians_data["colors"]
    opacities = gaussians_data["opacities"]

    # 创建PLY数据结构
    vertex_data = np.zeros(
        len(positions),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("f_rest_0", "f4"),
            ("f_rest_1", "f4"),
            ("f_rest_2", "f4"),
            ("f_rest_3", "f4"),
            ("f_rest_4", "f4"),
            ("f_rest_5", "f4"),
            ("f_rest_6", "f4"),
            ("f_rest_7", "f4"),
            ("f_rest_8", "f4"),
            ("f_rest_9", "f4"),
            ("f_rest_10", "f4"),
            ("f_rest_11", "f4"),
            ("f_rest_12", "f4"),
            ("f_rest_13", "f4"),
            ("f_rest_14", "f4"),
            ("f_rest_15", "f4"),
            ("f_rest_16", "f4"),
            ("f_rest_17", "f4"),
            ("f_rest_18", "f4"),
            ("f_rest_19", "f4"),
            ("f_rest_20", "f4"),
            ("f_rest_21", "f4"),
            ("f_rest_22", "f4"),
            ("f_rest_23", "f4"),
            ("f_rest_24", "f4"),
            ("f_rest_25", "f4"),
            ("f_rest_26", "f4"),
            ("f_rest_27", "f4"),
            ("f_rest_28", "f4"),
            ("f_rest_29", "f4"),
            ("f_rest_30", "f4"),
            ("f_rest_31", "f4"),
            ("f_rest_32", "f4"),
            ("f_rest_33", "f4"),
            ("f_rest_34", "f4"),
            ("f_rest_35", "f4"),
            ("f_rest_36", "f4"),
            ("f_rest_37", "f4"),
            ("f_rest_38", "f4"),
            ("f_rest_39", "f4"),
            ("f_rest_40", "f4"),
            ("f_rest_41", "f4"),
            ("f_rest_42", "f4"),
            ("f_rest_43", "f4"),
            ("f_rest_44", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ],
    )

    # 填充数据
    vertex_data["x"] = positions[:, 0]
    vertex_data["y"] = positions[:, 1]
    vertex_data["z"] = positions[:, 2]
    vertex_data["nx"] = 0  # 法线设为0
    vertex_data["ny"] = 0
    vertex_data["nz"] = 0
    vertex_data["f_dc_0"] = colors[:, 0]
    vertex_data["f_dc_1"] = colors[:, 1]
    vertex_data["f_dc_2"] = colors[:, 2]
    # 其他f_rest字段保持为0
    vertex_data["opacity"] = opacities
    vertex_data["scale_0"] = scales[:, 0]
    vertex_data["scale_1"] = scales[:, 1]
    vertex_data["scale_2"] = scales[:, 2]
    vertex_data["rot_0"] = rotations[:, 0]
    vertex_data["rot_1"] = rotations[:, 1]
    vertex_data["rot_2"] = rotations[:, 2]
    vertex_data["rot_3"] = rotations[:, 3]

    # 创建PLY文件
    el = plyfile.PlyElement.describe(vertex_data, "vertex")
    plyfile.PlyData([el]).write(ply_path)


def transform_for_navigation(gaussians_data, transform_type="opencv_to_robot"):
    """将SHARP坐标系转换为智能设备导航坐标系.

    Args:
        gaussians_data: 高斯数据字典
        transform_type: 转换类型
            'opencv_to_robot': OpenCV (x右,y下,z前) -> 机器人 (x前,y左,z上)
            'opencv_to_enu': OpenCV (x右,y下,z前) -> ENU (x东,y北,z上)
    """
    positions = gaussians_data["positions"].copy()

    if transform_type == "opencv_to_robot":
        # OpenCV (x右, y下, z前) -> 机器人标准 (x前, y左, z上)
        # x' = z, y' = -x, z' = y
        new_positions = np.zeros_like(positions)
        new_positions[:, 0] = positions[:, 2]  # x = z
        new_positions[:, 1] = -positions[:, 0]  # y = -x
        new_positions[:, 2] = positions[:, 1]  # z = y

    elif transform_type == "opencv_to_enu":
        # OpenCV (x右, y下, z前) -> ENU (x东, y北, z上)
        # x' = x, y' = z, z' = y
        new_positions = np.zeros_like(positions)
        new_positions[:, 0] = positions[:, 0]  # x = x
        new_positions[:, 1] = positions[:, 2]  # y = z
        new_positions[:, 2] = positions[:, 1]  # z = y

    else:
        raise ValueError(f"不支持的转换类型: {transform_type}")

    gaussians_data["positions"] = new_positions
    return gaussians_data


def process_sharp_output(input_dir, output_dir, transform_type="opencv_to_robot"):
    """批量处理SHARP输出文件，进行坐标系转换.

    Args:
        input_dir: SHARP输出目录
        output_dir: 处理后输出目录
        transform_type: 坐标转换类型
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ply_files = list(input_path.glob("*.ply"))

    if not ply_files:
        print(f"警告: 在 {input_dir} 中没有找到PLY文件")
        return

    print(f"找到 {len(ply_files)} 个PLY文件待处理")

    for ply_file in ply_files:
        print(f"处理: {ply_file.name}")

        # 加载数据
        gaussians_data = load_ply_gaussians(ply_file)

        # 坐标转换
        transformed_data = transform_for_navigation(gaussians_data, transform_type)

        # 保存转换后的数据
        output_file = output_path / f"{ply_file.stem}_robot.ply"
        save_ply_gaussians(output_file, transformed_data)

        print(f"  ✓ 保存到: {output_file}")

    print("坐标转换完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHARP坐标转换工具 - 智能设备导航优化")
    parser.add_argument("--input", "-i", required=True, help="SHARP输出目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument(
        "--transform",
        "-t",
        choices=["opencv_to_robot", "opencv_to_enu"],
        default="opencv_to_robot",
        help="坐标转换类型",
    )

    args = parser.parse_args()

    print("SHARP 3D高斯模型坐标转换工具")
    print("=" * 40)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"转换类型: {args.transform}")
    print()

    process_sharp_output(args.input, args.output, args.transform)
