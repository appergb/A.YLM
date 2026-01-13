#!/usr/bin/env python3
"""iPhone照片去畸变工具.

专门用于处理iPhone超广角镜头的畸变，优化SHARP模型生成质量.

使用方法:
python undistort_iphone.py -i inputs/input_images -o outputs/undistorted_images

作者: TRIP(appergb)
项目参与者: closer, true
个人研发项目
"""

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

# iPhone 16 Pro Max超广角镜头参数 (示例值，需要根据实际设备校准)
IPHONE_ULTRA_WIDE_PARAMS = {
    "fx": 1200,  # 焦距x
    "fy": 1200,  # 焦距y
    "cx": 960,  # 光心x
    "cy": 540,  # 光心y
    "k1": -0.3,  # 径向畸变系数
    "k2": 0.1,
    "p1": 0.0,  # 切向畸变系数
    "p2": 0.0,
    "width": 1920,
    "height": 1080,
}


def load_camera_params(json_path=None):
    """加载相机参数."""
    if json_path and Path(json_path).exists():
        with open(json_path, "r") as f:
            return json.load(f)
    else:
        print("使用默认iPhone超广角参数，如需精确校准请提供相机参数文件")
        return IPHONE_ULTRA_WIDE_PARAMS


def undistort_image(image_path, camera_params, output_path):
    """去畸变单张图像."""
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图像: {image_path}")
        return False

    h, w = img.shape[:2]

    # 相机矩阵
    K = np.array(
        [
            [camera_params["fx"], 0, camera_params["cx"]],
            [0, camera_params["fy"], camera_params["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # 畸变系数 [k1, k2, p1, p2]
    dist_coeffs = np.array(
        [
            camera_params["k1"],
            camera_params["k2"],
            camera_params["p1"],
            camera_params["p2"],
        ],
        dtype=np.float32,
    )

    # 计算去畸变映射
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1
    )

    # 应用去畸变
    undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # 裁剪到有效区域
    x, y, w_crop, h_crop = roi
    undistorted = undistorted[y : y + h_crop, x : x + w_crop]

    # 保存结果
    cv2.imwrite(str(output_path), undistorted)
    return True


def process_directory(input_dir, output_dir, camera_params):
    """处理目录中的所有图像."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    image_files: List[Path] = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"**/*{ext}"))
        image_files.extend(input_path.glob(f"**/*{ext.upper()}"))

    if not image_files:
        print(f"在 {input_dir} 中没有找到支持的图像文件")
        return

    print(f"找到 {len(image_files)} 张图像待处理")

    success_count = 0
    for image_file in image_files:
        # 保持相对路径结构
        relative_path = image_file.relative_to(input_path)
        output_file = output_path / relative_path

        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"处理: {relative_path}")

        if undistort_image(image_file, camera_params, output_file):
            success_count += 1
            print(f"  ✓ 保存到: {output_file}")
        else:
            print(f"  ✗ 处理失败: {relative_path}")

    print(f"\n处理完成: {success_count}/{len(image_files)} 张图像成功去畸变")


def calibrate_from_images(image_dir, pattern_size=(9, 6), square_size=1.0):
    """从图像序列进行相机校准.

    Args:
        image_dir: 包含校准图像的目录.
        pattern_size: 棋盘格角点数量 (width, height).
        square_size: 棋盘格方块大小 (cm).

    Returns:
        校准参数字典或None.
    """
    print("开始相机校准...")

    # 准备对象点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储所有图像的点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点

    # 查找图像文件
    image_path = Path(image_dir)
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))

    print(f"找到 {len(image_files)} 张校准图像")

    for image_file in image_files:
        img = cv2.imread(str(image_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            # 提高角点精度
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
            imgpoints.append(corners2)

            # 显示角点 (可选)
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        print("错误: 没有找到有效的棋盘格图像")
        return None

    # 相机校准
    h, w = gray.shape[:2]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    if ret:
        print("相机校准成功!")
        print(f"相机矩阵 K:\n{K}")
        print(f"畸变系数: {dist.ravel()}")

        # 保存参数
        params = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "k1": float(dist[0, 0]),
            "k2": float(dist[0, 1]),
            "p1": float(dist[0, 2]),
            "p2": float(dist[0, 3]),
            "width": w,
            "height": h,
        }

        return params
    else:
        print("相机校准失败")
        return None


def main():
    """主函数入口."""
    parser = argparse.ArgumentParser(description="iPhone照片去畸变工具")
    parser.add_argument("--input", "-i", required=True, help="输入图像目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--params", "-p", help="相机参数JSON文件路径")
    parser.add_argument("--calibrate", "-c", action="store_true", help="进行相机校准")
    parser.add_argument(
        "--pattern-size",
        nargs=2,
        type=int,
        default=[9, 6],
        help="棋盘格角点数量 (默认: 9 6)",
    )
    parser.add_argument(
        "--square-size", type=float, default=1.0, help="棋盘格方块大小 (默认: 1.0)"
    )

    args = parser.parse_args()

    print("iPhone照片去畸变工具")
    print("=" * 30)

    if args.calibrate:
        print("执行相机校准模式...")
        params = calibrate_from_images(
            args.input, tuple(args.pattern_size), args.square_size
        )
        if params:
            # 保存校准结果
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            params_file = output_path / "camera_params.json"
            with open(params_file, "w") as f:
                json.dump(params, f, indent=2)
            print(f"相机参数已保存到: {params_file}")
    else:
        # 去畸变模式
        params = load_camera_params(args.params)
        print(
            f"相机参数: fx={params['fx']:.1f}, fy={params['fy']:.1f}, "
            f"cx={params['cx']:.1f}, cy={params['cy']:.1f}"
        )
        print(f"畸变系数: k1={params['k1']:.3f}, k2={params['k2']:.3f}")

        process_directory(args.input, args.output, params)


if __name__ == "__main__":
    main()
