#!/usr/bin/env python3
"""点云自动体素化工具.

用于将3D高斯点云转换为体素表示，用于路径规划和环境建模.

功能特性：
- 自动读取PLY/PCD/XYZ格式点云
- 体素下采样和网格化
- 地面自动识别和水平化（RANSAC + PCA）
- 去除孤立噪声点
- 批量处理支持
- 多种格式导出

作者: TRIP(appergb)
项目参与者: closer, true
个人研发项目
"""

import argparse
import logging
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class PointCloudVoxelizer:
    """点云体素化处理器."""

    # 体素化参数
    DEFAULT_VOXEL_SIZE = 0.005  # 5mm
    DEFAULT_MAX_DISTANCE = 10.0  # 10米
    DEFAULT_MIN_INLIER_RATIO = 0.08  # 8%
    DEFAULT_MAX_GROUND_ANGLE = 60.0  # 60度

    # 地面平整参数
    DEFAULT_WINDOW_SIZE = 2.0  # 2米
    DEFAULT_DEPRESSION_THRESHOLD = 0.05  # 5cm

    def __init__(self, voxel_size: float = DEFAULT_VOXEL_SIZE, debug: bool = False):
        """初始化体素化器.

        Args:
            voxel_size: 体素尺寸（米），默认2cm
            debug: 是否启用调试模式
        """
        self.voxel_size = voxel_size
        self.debug = debug

        # 检测平台和硬件
        self.is_macos = platform.system() == "Darwin"
        self.is_apple_silicon = self._detect_apple_silicon()

        # 设置日志
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        if self.is_apple_silicon:
            self.logger.warning("检测到Apple Silicon (M系列芯片)，将使用兼容模式")
            self.logger.warning("某些几何操作可能会被跳过以避免兼容性问题")

    def _detect_apple_silicon(self) -> bool:
        """检测是否为Apple Silicon (M系列芯片)."""
        try:
            if not self.is_macos:
                return False

            # 方法1: 检查处理器型号
            try:
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    cpu_brand = result.stdout.strip()
                    if "Apple" in cpu_brand and (
                        "M1" in cpu_brand or "M2" in cpu_brand or "M3" in cpu_brand
                    ):
                        return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # 方法2: 检查架构
            machine = platform.machine().lower()
            if machine in ["arm64", "aarch64"]:
                return True

            # 方法3: 环境变量检查
            if os.environ.get("APPLE_SILICON_FORCE_COMPAT", "").lower() in [
                "1",
                "true",
                "yes",
            ]:
                return True

            return False

        except Exception:
            # 如果检测失败，默认认为不是Apple Silicon
            return False

    def load_pointcloud(self, file_path: str) -> Optional[o3d.geometry.PointCloud]:
        """加载点云文件.

        Args:
            file_path: 点云文件路径

        Returns:
            Open3D点云对象或None（如果加载失败）
        """
        # 防御性编程：检查文件是否存在
        if not Path(file_path).exists():
            self.logger.error(f"文件不存在: {file_path}")
            return None

        # 检查文件是否为空
        if os.path.getsize(file_path) == 0:
            self.logger.error(f"文件为空: {file_path}")
            return None

        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".ply":
                pcd = o3d.io.read_point_cloud(file_path)
            elif file_ext == ".pcd":
                pcd = o3d.io.read_point_cloud(file_path)
            elif file_ext == ".xyz":
                # 读取XYZ格式（假设每行是x y z）
                points = np.loadtxt(file_path)
                if points.ndim == 1:
                    points = points.reshape(1, -1)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            else:
                self.logger.error(f"不支持的文件格式: {file_ext}")
                return None

            if len(pcd.points) == 0:
                self.logger.error(f"点云文件为空: {file_path}")
                return None

            self.logger.info(f"成功加载点云: {file_path}, 点数: {len(pcd.points)}")
            return pcd

        except Exception as e:
            self.logger.error(f"加载点云失败 {file_path}: {str(e)}")
            return None

    def crop_to_local_region(
        self,
        pcd: o3d.geometry.PointCloud,
        max_distance: float = 10.0,
        front_axis: str = "auto",
        keep_front_only: bool = True,
        adaptive_distance: bool = True,
    ) -> o3d.geometry.PointCloud:
        """裁剪点云到局部区域，用于路径规划.

        Args:
            pcd: 输入点云
            max_distance: 最大距离（米），默认10.0m（1000cm）
            front_axis: 前向轴 ('x', 'y', 'z', 'auto')，默认自动检测
            keep_front_only: 是否只保留前半空间

        Returns:
            裁剪后的点云
        """
        try:
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return pcd

            self.logger.info(
                f"开始局部区域裁剪... 距离阈值: {max_distance}m, 前向轴: {front_axis}"
            )

            # 1. 距离约束：计算到原点的距离（摄像头位置）
            distances_squared = np.sum(points**2, axis=1)

            if adaptive_distance:
                # 自适应距离：基于设定的距离阈值进行调整，确保至少保留足够的点
                sorted_distances = np.sort(distances_squared)
                # 计算设定距离对应的点数
                target_distance_squared = max_distance**2
                points_within_range = np.sum(
                    sorted_distances <= target_distance_squared
                )

                if points_within_range < 10000:  # 如果设定距离内点数太少
                    # 扩大距离以确保至少有10000个点
                    min_required_points = min(10000, len(points))
                    if min_required_points < len(sorted_distances):
                        adaptive_threshold = sorted_distances[min_required_points - 1]
                        max_distance = np.sqrt(adaptive_threshold)
                        self.logger.info(
                            f"自适应扩大距离阈值: {max_distance:.3f}m "
                            f"(确保至少{points_within_range}个点)"
                        )
                else:
                    self.logger.info(
                        f"使用设定距离阈值: {max_distance:.3f}m "
                        f"(范围内有{points_within_range}个点)"
                    )

            distance_mask = distances_squared <= (max_distance**2)

            self.logger.info(
                f"距离约束: {np.sum(distance_mask)}/{len(points)} 点保留 "
                f"(阈值: {max_distance:.3f}m)"
            )

            # 2. 前向约束：只保留前半空间 (z >= 0, 假设摄像头朝向+z方向)
            if keep_front_only:
                front_mask = points[:, 2] >= 0  # z >= 0
                self.logger.info(
                    f"前向约束 (z >= 0): "
                    f"{np.sum(front_mask & distance_mask)}/"
                    f"{np.sum(distance_mask)} 点保留"
                )
            else:
                front_mask = np.ones(len(points), dtype=bool)

            # 3. 组合mask
            final_mask = distance_mask & front_mask
            cropped_points = points[final_mask]

            self.logger.info(f"裁剪完成: {len(points)} -> {len(cropped_points)} 点")

            # 创建新的点云
            pcd_cropped = o3d.geometry.PointCloud()
            pcd_cropped.points = o3d.utility.Vector3dVector(cropped_points)

            # 保留颜色信息（如果有）
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                pcd_cropped.colors = o3d.utility.Vector3dVector(colors[final_mask])

            return pcd_cropped

        except Exception as e:
            self.logger.warning(f"区域裁剪失败，使用原始点云: {str(e)}")
            return pcd

    def remove_noise(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    ) -> o3d.geometry.PointCloud:
        """去除噪声点（统计离群点去除）.

        Args:
            pcd: 输入点云
            nb_neighbors: 邻域点数阈值
            std_ratio: 标准差倍数阈值

        Returns:
            去噪后的点云
        """
        try:
            self.logger.info("开始去噪处理...")
            pcd_clean, ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
            self.logger.info(
                f"去噪完成: {len(pcd.points)} -> {len(pcd_clean.points)} 点"
            )
            return pcd_clean
        except Exception as e:
            self.logger.warning(f"去噪失败，使用原始点云: {str(e)}")
            return pcd

    def detect_and_correct_ground_plane_enhanced(
        self,
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float = 0.01,
        ransac_n: int = 3,
        num_iterations: int = 1000,
        min_inlier_ratio: float = 0.08,
        max_ground_angle: float = 60.0,
    ) -> Tuple[o3d.geometry.PointCloud, dict]:
        """增强版地面检测和校正算法.

        特性：
        - 多重RANSAC检测，提高鲁棒性
        - 地面质量验证
        - 自适应阈值调整
        - 更好的旋转矩阵计算
        - 地面倾斜度限制

        Args:
            pcd: 输入点云
            distance_threshold: 平面距离阈值
            ransac_n: RANSAC采样点数
            num_iterations: 最大迭代次数
            min_inlier_ratio: 最小内点比例阈值
            max_ground_angle: 最大地面倾斜角度（度），默认45.0

        Returns:
            (校正后的点云, 检测结果信息)
        """
        try:
            self.logger.info("开始增强版地面检测和校正...")

            # 1. 多重RANSAC检测 - 运行多次取最佳结果
            best_plane_model, best_inliers = self._perform_multi_ransac(
                pcd, distance_threshold, ransac_n, num_iterations
            )

            if best_plane_model is None:
                self.logger.error("所有RANSAC检测均失败")
                return pcd, {"success": False, "error": "RANSAC检测失败"}

            [a, b, c, d] = best_plane_model
            inlier_ratio = (
                len(best_inliers) / len(pcd.points) if best_inliers is not None else 0.0
            )

            # 2. 地面质量评估
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)

            # 计算地面倾斜角度（与水平面的夹角）
            ground_angle = np.degrees(np.arccos(np.abs(np.dot(normal, [0, 0, 1]))))

            self.logger.info(
                f"地面法向量: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]"
            )
            self.logger.info(f"内点比例: {inlier_ratio:.2%}")
            self.logger.info(f"地面倾斜角度: {ground_angle:.1f}°")

            # 3. 质量验证
            quality_info = {
                "inlier_ratio": inlier_ratio,
                "ground_angle": ground_angle,
                "plane_normal": normal,
                "plane_distance": d,
                "inlier_count": len(best_inliers) if best_inliers is not None else 0,
            }

            # 检查地面质量
            quality_valid, quality_error = self._validate_ground_quality(
                inlier_ratio, ground_angle, min_inlier_ratio, max_ground_angle
            )
            if not quality_valid:
                return pcd, {**quality_info, "success": False, "error": quality_error}

            # 4. 计算旋转矩阵 - 增强版
            z_axis = np.array([0, 0, 1])  # 目标Z轴方向

            # 计算旋转轴和角度
            rot_axis = np.cross(normal, z_axis)
            rot_angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))

            # 处理平行情况
            if np.linalg.norm(rot_axis) < 1e-6:
                if np.dot(normal, z_axis) > 0:
                    self.logger.info("地面已经基本水平，无需校正")
                    return pcd, {**quality_info, "success": True, "corrected": False}
                else:
                    # 完全相反的情况（罕见）
                    rot_axis = np.array([1, 0, 0])
                    rot_angle = np.pi
                    self.logger.info("地面需要180度翻转校正")

            rot_axis = rot_axis / np.linalg.norm(rot_axis)

            # 5. 增强版旋转矩阵计算 - 跨平台兼容
            try:
                R_tensor = self._compute_rotation_matrix(rot_axis, rot_angle)
                # 转换为numpy数组用于几何操作
                R = np.asarray(R_tensor)

                # 计算旋转中心（使用内点的质心）
                inlier_points = np.asarray(pcd.points)[best_inliers]
                center = np.mean(inlier_points, axis=0)

                self.logger.info(
                    f"地面校正旋转中心: "
                    f"[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]"
                )

                # Apple Silicon兼容性处理
                if self.is_apple_silicon:
                    self.logger.warning(
                        "Apple Silicon检测到，为避免兼容性问题跳过地面旋转校正"
                    )
                    return pcd, {
                        **quality_info,
                        "success": True,
                        "corrected": False,
                        "apple_silicon_compat": True,
                    }

                # 应用旋转
                pcd_corrected = pcd.rotate(R, center=center)

                # 6. 验证校正结果
                verification_result = self._verify_ground_correction(
                    pcd_corrected, quality_info
                )
                if not verification_result["valid"]:
                    self.logger.warning(
                        f"校正结果验证失败: {verification_result['reason']}"
                    )
                    return pcd, {
                        **quality_info,
                        "success": False,
                        "error": verification_result["reason"],
                    }

                self.logger.info("地面校正完成并验证通过")
                return pcd_corrected, {
                    **quality_info,
                    "success": True,
                    "corrected": True,
                }

            except Exception as rotation_error:
                self.logger.warning(
                    f"旋转操作失败，使用原始点云: {str(rotation_error)}"
                )
                return pcd, {
                    **quality_info,
                    "success": False,
                    "error": f"旋转失败: {str(rotation_error)}",
                }

        except Exception as e:
            self.logger.error(f"地面检测失败: {str(e)}")
            import traceback

            self.logger.debug(f"详细错误: {traceback.format_exc()}")
            return pcd, {"success": False, "error": str(e)}

    def _perform_multi_ransac(
        self,
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """执行多重RANSAC检测."""
        # Apple Silicon兼容性：跳过RANSAC检测
        if self.is_apple_silicon:
            self.logger.warning(
                "Apple Silicon检测到，为避免兼容性问题跳过RANSAC地面检测"
            )
            # 返回默认水平地面
            return np.array([0, 0, 1, 0]), np.array([])

        # 运行多次RANSAC取最佳结果
        best_plane_model = None
        best_inliers = None
        best_score = 0.0

        self.logger.info("正在进行多重RANSAC地面检测...")

        for attempt in range(3):  # 尝试3次
            try:
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=ransac_n,
                    num_iterations=num_iterations,
                )

                inlier_ratio = len(inliers) / len(pcd.points)
                score = inlier_ratio * len(inliers)  # 综合评分

                if score > best_score:
                    best_plane_model = plane_model
                    best_inliers = inliers
                    best_score = score

            except Exception as e:
                self.logger.debug(f"RANSAC尝试 {attempt + 1} 失败: {str(e)}")
                continue

        return best_plane_model, best_inliers

    def _validate_ground_quality(
        self,
        inlier_ratio: float,
        ground_angle: float,
        min_inlier_ratio: float,
        max_ground_angle: float,
    ) -> Tuple[bool, Optional[str]]:
        """验证地面质量."""
        # 检查内点比例
        if inlier_ratio < min_inlier_ratio:
            error_msg = f"内点比例过低: {inlier_ratio:.2%} < {min_inlier_ratio:.2%}"
            self.logger.warning(error_msg)
            return False, error_msg

        # 检查地面倾斜角度
        if ground_angle > max_ground_angle:
            error_msg = (
                f"地面倾斜角度过大: {ground_angle:.1f}° > {max_ground_angle:.1f}°"
            )
            self.logger.warning(error_msg)
            return False, error_msg

        return True, None

    def _compute_rotation_matrix(
        self, rot_axis: np.ndarray, rot_angle: float
    ) -> o3d.core.Tensor:
        """计算旋转矩阵 - 使用Open3D核心Tensor API.

        Args:
            rot_axis: 旋转轴
            rot_angle: 旋转角度

        Returns:
            3x3旋转矩阵（Open3D Tensor）
        """
        try:
            # 使用Open3D标准旋转矩阵API获取numpy数组
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
            # 转换为Open3D Tensor
            return o3d.core.Tensor(R.reshape(3, 3), dtype=o3d.core.Dtype.Float32)

        except Exception as e:
            self.logger.error(f"旋转矩阵计算失败: {str(e)}")
            # 返回单位矩阵（无旋转）
            R_identity = np.eye(3)
            return o3d.core.Tensor(R_identity, dtype=o3d.core.Dtype.Float32)

    def _orthonormalize_matrix(self, R: np.ndarray) -> np.ndarray:
        """正交化矩阵，确保其为有效的旋转矩阵.

        Args:
            R: 输入矩阵

        Returns:
            正交化的旋转矩阵
        """
        try:
            # 使用Gram-Schmidt过程进行正交化
            u, _, vt = np.linalg.svd(R)
            R_ortho = u @ vt

            # 确保行列式为1（旋转矩阵特性）
            if np.linalg.det(R_ortho) < 0:
                # 如果行列式为负，反转最后一列
                R_ortho[:, -1] *= -1

            return R_ortho

        except Exception as e:
            self.logger.warning(f"矩阵正交化失败: {str(e)}")
            return np.eye(3)

    def _verify_ground_correction(
        self, pcd: o3d.geometry.PointCloud, original_info: dict
    ) -> dict:
        """验证地面校正结果.

        Args:
            pcd: 校正后的点云
            original_info: 原始检测信息

        Returns:
            验证结果
        """
        try:
            # Apple Silicon兼容性：跳过地面验证
            if self.is_apple_silicon:
                return {"valid": True, "improvement": 0}

            # 重新检测地面
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.01, ransac_n=3, num_iterations=1000
            )

            [a, b, c, d] = plane_model
            inlier_ratio = len(inliers) / len(pcd.points)
            normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
            ground_angle = np.degrees(np.arccos(np.abs(np.dot(normal, [0, 0, 1]))))

            # 验证标准
            angle_improved = (
                ground_angle < original_info["ground_angle"] * 0.8
            )  # 角度改善至少20%
            inliers_preserved = (
                inlier_ratio > original_info["inlier_ratio"] * 0.9
            )  # 内点数保持90%以上

            if angle_improved and inliers_preserved:
                return {
                    "valid": True,
                    "improvement": original_info["ground_angle"] - ground_angle,
                }
            else:
                reason = []
                if not angle_improved:
                    reason.append(
                        f"角度改善不足: {original_info['ground_angle']:.1f}° "
                        f"-> {ground_angle:.1f}°"
                    )
                if not inliers_preserved:
                    reason.append(
                        f"内点比例下降: {original_info['inlier_ratio']:.2%} "
                        f"-> {inlier_ratio:.2%}"
                    )
                return {"valid": False, "reason": "; ".join(reason)}

        except Exception as e:
            return {"valid": False, "reason": f"验证过程失败: {str(e)}"}

    def detect_and_correct_ground_plane(
        self,
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float = 0.01,
        ransac_n: int = 3,
        num_iterations: int = 1000,
    ) -> o3d.geometry.PointCloud:
        """兼容性接口 - 调用增强版算法."""
        corrected_pcd, result = self.detect_and_correct_ground_plane_enhanced(
            pcd, distance_threshold, ransac_n, num_iterations
        )

        if result["success"]:
            return corrected_pcd
        else:
            self.logger.info(f"地面校正未应用: {result.get('error', '未知原因')}")
            return pcd

    def _fix_ply_header(self, ply_path: str) -> None:
        """修复PLY文件头部，确保兼容性.

        Args:
            ply_path: PLY文件路径
        """
        try:
            with open(ply_path, "r") as f:
                lines = f.readlines()

            # 检查是否需要修复头部
            if len(lines) > 0 and "ply" in lines[0].lower():
                # PLY文件看起来正常
                return

            # 如果头部有问题，尝试重新创建
            self.logger.warning(f"PLY文件头部可能有问题: {ply_path}")

        except Exception as e:
            self.logger.debug(f"PLY头部检查失败: {str(e)}")

    def detect_ground_plane(
        self,
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float = 0.01,
        ransac_n: int = 3,
        num_iterations: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用RANSAC检测地面平面.

        Args:
            pcd: 输入点云
            distance_threshold: 平面距离阈值
            ransac_n: RANSAC采样点数
            num_iterations: 最大迭代次数

        Returns:
            (平面方程[a,b,c,d], 内点索引)
        """
        try:
            self.logger.info("开始地面平面检测...")

            # Apple Silicon兼容性：返回默认地面
            if self.is_apple_silicon:
                self.logger.warning("Apple Silicon检测到，使用默认水平地面")
                return np.array([0, 0, 1, 0]), np.array([])

            # 使用RANSAC平面分割
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )

            [a, b, c, d] = plane_model
            self.logger.info(f"平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

            # 计算内点比例
            inlier_ratio = len(inliers) / len(pcd.points)
            self.logger.info(f"内点比例: {inlier_ratio:.2%}")

            return np.array(plane_model), np.array(inliers)

        except Exception as e:
            self.logger.error(f"地面检测失败: {str(e)}")
            # 返回默认水平地面
            return np.array([0, 0, 1, 0]), np.array([])

    def align_to_ground(
        self, pcd: o3d.geometry.PointCloud, plane_model: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """将点云旋转对齐到地面水平.

        Args:
            pcd: 输入点云
            plane_model: 平面方程[a,b,c,d]

        Returns:
            对齐后的点云
        """
        try:
            self.logger.info("开始地面对齐...")

            a, b, c, d = plane_model

            # 计算旋转矩阵，将地面法向量旋转到Z轴方向
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)

            # Z轴单位向量
            z_axis = np.array([0, 0, 1])

            # 计算旋转轴和角度
            rot_axis = np.cross(normal, z_axis)
            rot_angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))

            # 处理平行情况
            if np.linalg.norm(rot_axis) < 1e-6:
                if np.dot(normal, z_axis) > 0:
                    # 已经对齐
                    self.logger.info("点云已基本水平，无需旋转")
                    return pcd
                else:
                    # 翻转180度
                    rot_axis = np.array([1, 0, 0])
                    rot_angle = np.pi
                    self.logger.info("执行180度翻转对齐")

            rot_axis = rot_axis / np.linalg.norm(rot_axis)

            # 构建旋转矩阵
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)

            # 计算质心作为旋转中心
            center = pcd.get_center()
            self.logger.info(f"旋转中心: {center}")

            # 应用旋转
            pcd_aligned = pcd.rotate(R, center=center)

            self.logger.info("地面对齐完成")
            return pcd_aligned

        except Exception as e:
            self.logger.warning(f"地面对齐失败，使用原始点云: {str(e)}")
            import traceback

            self.logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return pcd

    def convert_to_ground_coordinate_system(
        self, pcd: o3d.geometry.PointCloud, ground_plane: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """将点云转换为Y轴垂直于地面的坐标系.

        Args:
            pcd: 输入点云
            ground_plane: 地面平面方程 [a, b, c, d] (ax + by + cz + d = 0)

        Returns:
            转换后的点云（Y轴垂直于地面）
        """
        try:
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return pcd

            # 地面法向量
            normal = ground_plane[:3]
            normal = normal / np.linalg.norm(normal)  # 归一化

            # 计算旋转矩阵，将地面法向量旋转到Y轴
            # Y轴单位向量
            y_axis = np.array([0, 1, 0])

            # 计算旋转轴和角度
            cross = np.cross(normal, y_axis)
            dot = np.dot(normal, y_axis)

            if np.abs(np.linalg.norm(cross)) < 1e-6:
                # 法向量已经平行于Y轴
                rotation_matrix = np.eye(3)
            else:
                # Rodrigues公式计算旋转矩阵
                cross_norm = np.linalg.norm(cross)
                axis = cross / cross_norm
                angle = np.arctan2(cross_norm, dot)

                # Rodrigues旋转矩阵
                K = np.array(
                    [
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ]
                )
                rotation_matrix = (
                    np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                )

            # 应用旋转变换
            rotated_points = points @ rotation_matrix.T

            # 创建新的点云
            transformed_pcd = o3d.geometry.PointCloud()
            transformed_pcd.points = o3d.utility.Vector3dVector(rotated_points)

            # 复制其他属性
            if pcd.has_colors():
                transformed_pcd.colors = pcd.colors
            if pcd.has_normals():
                # 法向量也需要旋转
                rotated_normals = np.asarray(pcd.normals) @ rotation_matrix.T
                transformed_pcd.normals = o3d.utility.Vector3dVector(rotated_normals)

            self.logger.info("坐标系转换完成：Y轴垂直于地面")
            return transformed_pcd

        except Exception as e:
            self.logger.error(f"坐标系转换失败: {str(e)}")
            return pcd

    def level_ground_voxels(
        self,
        voxel_centers: np.ndarray,
        ground_height: float = 0.0,
        leveling_range: float = 0.1,
    ) -> np.ndarray:
        """智能地面平整处理：填充凹陷区域而不改变真实地形特征.

        Args:
            voxel_centers: 体素中心坐标数组
            ground_height: 地面高度（Y坐标）
            leveling_range: 平整范围（米）

        Returns:
            平整后的体素中心坐标
        """
        try:
            leveled_centers = voxel_centers.copy()

            # 找出靠近地面的体素（Y坐标在ground_height ± leveling_range范围内）
            ground_mask = np.abs(voxel_centers[:, 1] - ground_height) <= leveling_range

            if not np.any(ground_mask):
                return leveled_centers

            ground_voxels = leveled_centers[ground_mask]

            # 智能凹陷检测和填充
            filled_count = self._intelligent_ground_leveling(
                ground_voxels, ground_height
            )

            self.logger.info(
                f"智能地面平整: 检测到 {len(ground_voxels)} 个地面体素，"
                f"填充了 {filled_count} 个凹陷"
            )

            # 更新原始数组
            leveled_centers[ground_mask] = ground_voxels

            return leveled_centers

        except Exception as e:
            self.logger.error(f"地面平整失败: {str(e)}")
            return voxel_centers

    def _intelligent_ground_leveling(
        self, ground_voxels: np.ndarray, ground_height: float = 0.0
    ) -> int:
        """智能地面平整算法：检测和填充局部凹陷，保护真实地形特征.

        算法逻辑：
        1. 将地面体素分成局部区域进行分析
        2. 在每个局部区域计算地面统计特征（均值、中位数、标准差）
        3. 检测相对于局部地面的凹陷点
        4. 只填充明显凹陷，避免改变台阶、斜坡等真实地形特征
        5. 填充高度限制在5cm以内，确保不会过度修改地形

        Args:
            ground_voxels: 地面体素坐标数组（会被修改）
            ground_height: 参考地面高度

        Returns:
            填充的体素数量
        """
        if len(ground_voxels) < 10:
            # 体素太少，使用简单平整
            ground_voxels[:, 1] = ground_height
            return len(ground_voxels)

        filled_count = 0

        # 将地面体素分成小的局部区域进行分析
        # 基于X和Z坐标的空间分布
        x_coords = ground_voxels[:, 0]
        z_coords = ground_voxels[:, 2]
        y_coords = ground_voxels[:, 1]

        # 计算局部区域大小（基于体素密度）
        x_range = np.max(x_coords) - np.min(x_coords)
        z_range = np.max(z_coords) - np.min(z_coords)
        area = max(x_range * z_range, 1.0)

        # 根据区域大小确定局部窗口大小
        if area > 100:  # 大区域
            window_size = self.DEFAULT_WINDOW_SIZE  # 2米窗口
        elif area > 25:  # 中等区域
            window_size = 1.0  # 1米窗口
        else:  # 小区域
            window_size = 0.5  # 0.5米窗口

        # 使用KDTree优化距离计算，将O(n²)降低到O(n log n)
        tree = cKDTree(ground_voxels[:, [0, 2]])  # 只用X,Z坐标构建树

        # 对每个体素进行局部分析
        for i in range(len(ground_voxels)):
            voxel = ground_voxels[i]
            voxel_x, voxel_y, voxel_z = voxel

            # 查询局部邻域 - O(log n)复杂度
            indices = tree.query_ball_point(voxel[[0, 2]], window_size)

            if len(indices) < 3:
                continue  # 邻域体素太少，跳过

            local_y_coords = y_coords[indices]

            # 计算局部地面的统计特征
            local_mean = np.mean(local_y_coords)
            local_std = np.std(local_y_coords)
            local_median = np.median(local_y_coords)

            # 检测是否为凹陷
            # 条件1: 当前体素明显低于局部平均值
            # 条件2: 当前体素低于局部中位数一定阈值
            # 条件3: 局部标准差不大（避免在复杂地形中过度平整）

            depression_threshold = min(
                self.DEFAULT_DEPRESSION_THRESHOLD, local_std * 1.5
            )  # 5cm或1.5倍标准差
            is_depression = (voxel_y < local_median - depression_threshold) and (
                voxel_y < local_mean - depression_threshold * 0.5
            )

            # 额外的安全检查：确保不是真正的地形特征
            # 如果局部区域的标准差很大，说明可能是台阶或斜坡，不要平整
            if local_std > 0.03:  # 3cm以上的标准差认为是地形特征
                continue

            # 检查周围是否有高于当前体素的点（可能是台阶边缘）
            higher_neighbors = np.sum(local_y_coords > voxel_y + 0.02)  # 2cm以上
            if higher_neighbors > len(local_y_coords) * 0.3:  # 30%以上邻域更高
                continue  # 可能是台阶或斜坡的边缘

            if is_depression:
                # 计算填充高度：填充到局部中位数，但不超过一定范围
                target_height = min(local_median, ground_height + 0.02)  # 不超过2cm填充
                fill_amount = target_height - voxel_y

                # 只填充不超过5cm的凹陷（避免填充真正的坑洼）
                if 0 < fill_amount <= 0.05:
                    ground_voxels[i, 1] = target_height
                    filled_count += 1

        return filled_count

    def add_ground_texture_noise(
        self,
        voxel_centers: np.ndarray,
        ground_height: float = 0.0,
        noise_strength: float = 0.005,
        texture_scale: float = 0.1,
    ) -> np.ndarray:
        """为地面体素添加纹理噪声.

        Args:
            voxel_centers: 体素中心坐标数组
            ground_height: 地面高度
            noise_strength: 噪声强度（米）
            texture_scale: 纹理尺度

        Returns:
            添加噪声后的体素中心坐标
        """
        try:
            textured_centers = voxel_centers.copy()

            # 找出地面体素（Y坐标接近ground_height）
            ground_tolerance = 0.01  # 1cm容差
            ground_mask = (
                np.abs(voxel_centers[:, 1] - ground_height) <= ground_tolerance
            )

            if np.any(ground_mask):
                ground_voxels = textured_centers[ground_mask]

                # 生成基于位置的噪声
                # 使用X和Z坐标作为噪声输入
                x_coords = ground_voxels[:, 0]
                z_coords = ground_voxels[:, 2]

                # 创建简单的柏林噪声-like函数
                noise_x = (
                    noise_strength
                    * np.sin(x_coords / texture_scale)
                    * np.cos(z_coords / texture_scale)
                )
                noise_z = (
                    noise_strength
                    * np.cos(x_coords / texture_scale)
                    * np.sin(z_coords / texture_scale)
                )

                # 应用噪声到Y坐标（轻微的起伏）
                noise_y = 0.002 * (
                    np.sin(x_coords / texture_scale * 2)
                    + np.cos(z_coords / texture_scale * 2)
                )

                # 应用噪声
                textured_centers[ground_mask, 0] += noise_x
                textured_centers[ground_mask, 1] += noise_y
                textured_centers[ground_mask, 2] += noise_z

                self.logger.info(
                    f"地面纹理: 为 {np.sum(ground_mask)} 个体素添加了噪声纹理"
                )

            return textured_centers

        except Exception as e:
            self.logger.error(f"添加地面纹理失败: {str(e)}")
            return voxel_centers

    def recognize_ground_material(
        self, pcd: o3d.geometry.PointCloud, ground_plane: np.ndarray
    ) -> str:
        """识别地面材质类型.

        Args:
            pcd: 输入点云
            ground_plane: 地面平面方程

        Returns:
            识别出的材质类型 ('tile', 'wood', 'carpet', 'concrete', 'unknown')
        """
        try:
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return "unknown"

            # 计算点到地面的距离
            distances = np.abs(np.dot(points, ground_plane[:3]) + ground_plane[3])
            distances = distances / np.linalg.norm(ground_plane[:3])

            # 地面点（距离小于5cm）
            ground_points = points[distances < 0.05]

            if len(ground_points) < 100:
                return "unknown"

            # 分析地面点的分布特征

            # 1. 计算点的密度（每平方米点数）
            area = (np.max(ground_points[:, 0]) - np.min(ground_points[:, 0])) * (
                np.max(ground_points[:, 2]) - np.min(ground_points[:, 2])
            )
            density = len(ground_points) / max(area, 1.0)

            # 3. 计算Y坐标的变异系数
            cv_y = np.std(ground_points[:, 1]) / max(
                np.mean(ground_points[:, 1]), 0.001
            )

            # 基于特征识别材质
            if density > 1000 and cv_y < 0.01:
                # 高密度，高度均匀 - 可能是瓷砖
                return "tile"
            elif density > 500 and cv_y < 0.02:
                # 中等密度，较均匀 - 可能是木地板
                return "wood"
            elif density > 200 and cv_y > 0.05:
                # 密度较低，高度变化大 - 可能是地毯
                return "carpet"
            elif density > 100:
                # 一般密度 - 可能是混凝土
                return "concrete"
            else:
                return "unknown"

        except Exception as e:
            self.logger.error(f"地面材质识别失败: {str(e)}")
            return "unknown"

    def voxelize(
        self,
        pcd: o3d.geometry.PointCloud,
        create_regular_grid: bool = False,
        ground_leveling: bool = True,
        texture_noise: bool = True,
    ) -> o3d.geometry.PointCloud:
        """执行体素化处理.

        Args:
            pcd: 输入点云
            create_regular_grid: 是否创建规则体素网格（用于路径规划）

        Returns:
            体素化后的点云
        """
        try:
            self.logger.info(".4f")

            if create_regular_grid:
                # 创建规则体素网格（路径规划专用）
                points = np.asarray(pcd.points)

                # 检测地面平面用于坐标系转换
                ground_plane = None
                if ground_leveling or texture_noise:
                    try:
                        # Apple Silicon兼容性：跳过地面检测
                        if self.is_apple_silicon:
                            self.logger.warning("Apple Silicon检测到，跳过地面增强处理")
                        else:
                            plane_model, inliers = pcd.segment_plane(
                                distance_threshold=0.01, ransac_n=3, num_iterations=1000
                            )
                            inlier_ratio = len(inliers) / len(points)
                            if inlier_ratio > 0.1:  # 至少10%的点在地面上
                                ground_plane = plane_model
                                self.logger.info("检测到地面平面，将进行坐标系转换")
                            else:
                                self.logger.warning("地面检测失败，将跳过地面增强处理")
                    except Exception as e:
                        self.logger.warning(f"地面检测失败: {str(e)}")

                # 坐标系转换（Y轴垂直于地面）
                if ground_plane is not None:
                    pcd_transformed = self.convert_to_ground_coordinate_system(
                        pcd, ground_plane
                    )
                    points = np.asarray(pcd_transformed.points)
                else:
                    points = np.asarray(pcd.points)

                # 将点云坐标转换为体素索引
                voxel_indices = np.floor(points / self.voxel_size).astype(int)

                # 去除重复的体素索引，获得占据的体素
                unique_voxel_indices = np.unique(voxel_indices, axis=0)

                # 将体素索引转换回世界坐标（体素中心）
                voxel_centers = (unique_voxel_indices + 0.5) * self.voxel_size

                # 地面材质识别
                material_type = "unknown"
                if ground_plane is not None:
                    material_type = self.recognize_ground_material(pcd, ground_plane)
                    self.logger.info(f"识别地面材质: {material_type}")

                # 地面平整处理
                if ground_leveling and ground_plane is not None:
                    voxel_centers = self.level_ground_voxels(
                        voxel_centers, ground_height=0.0
                    )

                # 纹理噪声处理
                if texture_noise and ground_plane is not None:
                    # 根据材质调整噪声参数
                    noise_params = {
                        "tile": {"strength": 0.003, "scale": 0.05},  # 瓷砖：细腻纹理
                        "wood": {"strength": 0.005, "scale": 0.1},  # 木地板：中等纹理
                        "carpet": {"strength": 0.008, "scale": 0.15},  # 地毯：粗糙纹理
                        "concrete": {
                            "strength": 0.002,
                            "scale": 0.08,
                        },  # 混凝土：光滑纹理
                        "unknown": {"strength": 0.004, "scale": 0.1},  # 默认纹理
                    }

                    params = noise_params.get(material_type, noise_params["unknown"])
                    voxel_centers = self.add_ground_texture_noise(
                        voxel_centers,
                        ground_height=0.0,
                        noise_strength=params["strength"],
                        texture_scale=params["scale"],
                    )

                self.logger.info(f"规则体素网格: {len(voxel_centers)} 个占据体素")

                # 创建新的点云
                pcd_voxelized = o3d.geometry.PointCloud()
                pcd_voxelized.points = o3d.utility.Vector3dVector(voxel_centers)

            else:
                # 标准体素下采样
                pcd_voxelized = pcd.voxel_down_sample(voxel_size=self.voxel_size)

            self.logger.info(
                f"体素化完成: {len(pcd.points)} -> {len(pcd_voxelized.points)} 点"
            )

            return pcd_voxelized

        except Exception as e:
            self.logger.error(f"体素化失败: {str(e)}")
            return pcd

    def save_pointcloud(
        self, pcd: o3d.geometry.PointCloud, output_path: str, format_type: str = "auto"
    ) -> bool:
        """保存点云到文件.

        Args:
            pcd: 要保存的点云
            output_path: 输出文件路径
            format_type: 格式类型 ('ply', 'pcd', 'xyz', 'auto')

        Returns:
            保存是否成功
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if format_type == "auto":
                format_type = Path(output_path).suffix.lower().lstrip(".")

            if format_type == "ply":
                # 保存为ASCII格式的PLY文件，更好的兼容性
                success = o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
                if success:
                    self.logger.info(
                        f"PLY文件已保存为ASCII格式以提高兼容性: {output_path}"
                    )
            elif format_type == "pcd":
                success = o3d.io.write_point_cloud(output_path, pcd)
            elif format_type == "xyz":
                # 保存为XYZ格式
                points = np.asarray(pcd.points)
                if len(points) > 0:
                    np.savetxt(output_path, points, fmt="%.6f")
                    success = True
                else:
                    success = False
            else:
                self.logger.error(f"不支持的输出格式: {format_type}")
                return False

            if success:
                self.logger.info(f"点云已保存到: {output_path}")
                return True
            else:
                self.logger.error(f"保存失败: {output_path}")
                return False

        except Exception as e:
            self.logger.error(f"保存点云失败: {str(e)}")
            return False

    def visualize_pointcloud(
        self,
        pcd: o3d.geometry.PointCloud,
        title: str = "Point Cloud Visualization",
        show_voxel_grid: bool = True,
        voxel_size: Optional[float] = None,
    ) -> None:
        """可视化点云.

        Args:
            pcd: 要可视化的点云
            title: 窗口标题
            show_voxel_grid: 是否显示体素网格
            voxel_size: 体素尺寸（用于网格显示）
        """
        try:
            vis_list = []

            if show_voxel_grid and voxel_size is not None:
                # 创建体素网格可视化
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                    pcd, voxel_size=voxel_size
                )
                vis_list.append(voxel_grid)
                self.logger.info(
                    f"创建体素网格可视化: {len(voxel_grid.get_voxels())} 个体素"
                )
            else:
                vis_list.append(pcd)

            # 设置坐标系
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis_list.append(coord_frame)

            # 可视化
            o3d.visualization.draw_geometries(
                vis_list,
                window_name=title,
                width=1200,
                height=800,
                left=50,
                top=50,
                point_show_normal=False,
            )

        except Exception as e:
            self.logger.error(f"可视化失败: {str(e)}")

    def get_statistics(self, pcd: o3d.geometry.PointCloud) -> dict:
        """获取点云统计信息.

        Args:
            pcd: 输入点云

        Returns:
            统计信息字典
        """
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return {}

        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_min = bbox.min_bound
        bbox_max = bbox.max_bound

        stats = {
            "point_count": len(points),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_size": bbox_max - bbox_min,
            "volume": np.prod(bbox_max - bbox_min),
            "voxel_size": self.voxel_size,
            "estimated_voxels": np.prod(
                np.ceil((bbox_max - bbox_min) / self.voxel_size)
            ),
        }

        return stats

    def process_single_file(
        self,
        input_path: str,
        output_path: str,
        remove_noise_flag: bool = True,
        align_ground: bool = True,
        output_format: str = "auto",
        crop_local_region: bool = True,
        max_distance: float = 10.0,
        front_axis: str = "auto",
        keep_front_only: bool = True,
        visualize: bool = False,
        create_regular_grid: bool = False,
        save_intermediate: bool = False,
        intermediate_dir: str = os.environ.get(
            "VOXELIZER_INTERMEDIATE_DIR", "outputs/output_gaussians/cropped_pointclouds"
        ),
        ground_min_inliers: float = 0.08,
        ground_max_angle: float = 60.0,
        use_enhanced_ground: bool = False,
        ground_leveling: bool = True,
        texture_noise: bool = True,
    ) -> bool:
        """处理单个点云文件.

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            remove_noise_flag: 是否去噪
            align_ground: 是否地面对齐
            output_format: 输出格式
            ground_leveling: 是否对地面体素进行平整处理
            texture_noise: 是否为地面添加纹理噪声

        Returns:
            处理是否成功
        """
        try:
            self.logger.info(f"开始处理文件: {input_path}")

            # 1. 加载点云
            pcd = self.load_pointcloud(input_path)
            if pcd is None:
                return False

            # 2. 区域裁剪（可选）
            if crop_local_region:
                pcd = self.crop_to_local_region(
                    pcd,
                    max_distance,
                    front_axis,
                    keep_front_only,
                    adaptive_distance=True,
                )

                # 保存裁剪后的中间结果
                if save_intermediate:
                    intermediate_path = (
                        Path(intermediate_dir) / f"cropped_{Path(input_path).name}"
                    )
                    Path(intermediate_dir).mkdir(parents=True, exist_ok=True)
                    success_intermediate = self.save_pointcloud(
                        pcd, str(intermediate_path), output_format
                    )
                    if success_intermediate:
                        self.logger.info(f"裁剪后点云已保存到: {intermediate_path}")
                    else:
                        self.logger.warning(f"保存裁剪后点云失败: {intermediate_path}")

            # 3. 地面校正（路径规划专用或增强模式）
            use_ground_correction = create_regular_grid or use_enhanced_ground
            if use_ground_correction:
                if create_regular_grid:
                    self.logger.info("路径规划模式：启用增强版地面检测和校正")
                else:
                    self.logger.info("增强模式：启用增强版地面检测和校正")

                pcd, ground_result = self.detect_and_correct_ground_plane_enhanced(
                    pcd,
                    min_inlier_ratio=ground_min_inliers,
                    max_ground_angle=ground_max_angle,
                )
                if ground_result["success"]:
                    if ground_result.get("corrected", False):
                        self.logger.info("地面校正成功应用")
                        self.logger.info(".1f")
                        self.logger.info(".2f")
                    else:
                        self.logger.info("地面已水平，无需校正")
                else:
                    self.logger.warning(
                        f"地面校正失败: {ground_result.get('error', '未知错误')}"
                    )

            # 4. 去噪（可选）
            if remove_noise_flag:
                pcd = self.remove_noise(pcd)

            # 4. 地面检测和对齐（可选）
            ground_aligned = False
            if align_ground:
                try:
                    plane_model, inliers = self.detect_ground_plane(pcd)
                    if len(inliers) > len(pcd.points) * 0.1:  # 内点占比超过10%
                        # 临时跳过旋转功能，因为Open3D在macOS上有兼容性问题
                        self.logger.info(
                            "检测到地面平面，但暂时跳过旋转对齐（兼容性问题）"
                        )
                        # pcd = self.align_to_ground(pcd, plane_model)
                        # ground_aligned = True
                    else:
                        self.logger.warning("检测到的平面内点过少，跳过对齐")
                except Exception as e:
                    self.logger.warning(f"地面对齐失败，使用原始方向: {str(e)}")

            # 4. 体素化
            pcd_voxelized = self.voxelize(
                pcd, create_regular_grid, ground_leveling, texture_noise
            )

            # 5. 保存结果
            success = self.save_pointcloud(pcd_voxelized, output_path, output_format)

            # 6. 可视化（可选）
            if visualize and success:
                self.visualize_pointcloud(
                    pcd_voxelized,
                    title=f"体素化结果 - {Path(input_path).name}",
                    show_voxel_grid=True,
                    voxel_size=self.voxel_size,
                )

            # 7. 输出统计信息
            if success:
                stats = self.get_statistics(pcd_voxelized)
                self.logger.info("处理统计:")
                self.logger.info(f"  最终点数: {stats.get('point_count', 0)}")
                self.logger.info(f"  包围盒尺寸: {stats.get('bbox_size', 'N/A')}")
                self.logger.info(f"  地面对齐: {'是' if ground_aligned else '否'}")
                self.logger.info(".1f")

            return success

        except Exception as e:
            self.logger.error(f"处理文件失败 {input_path}: {str(e)}")
            import traceback

            self.logger.debug(f"详细错误: {traceback.format_exc()}")
            return False

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.ply",
        remove_noise_flag: bool = True,
        align_ground: bool = True,
        output_format: str = "ply",
        crop_local_region: bool = False,
        max_distance: float = 10.0,
        front_axis: str = "auto",
        keep_front_only: bool = True,
        visualize: bool = False,
        create_regular_grid: bool = False,
        save_intermediate: bool = False,
        intermediate_dir: str = "outputs/output_gaussians/cropped_pointclouds",
        ground_min_inliers: float = 0.08,
        ground_max_angle: float = 30.0,
        use_enhanced_ground: bool = False,
        ground_leveling: bool = True,
        texture_noise: bool = True,
    ) -> List[str]:
        """批量处理点云文件.

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_pattern: 文件匹配模式
            remove_noise_flag: 是否去噪
            align_ground: 是否地面对齐
            output_format: 输出格式

        Returns:
            成功处理的文件列表
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processed_files = []

        # 查找匹配的文件
        for file_path in input_path.glob(file_pattern):
            if file_path.is_file():
                # 构建输出文件路径
                output_file = output_path / f"voxelized_{file_path.name}"

                # 处理单个文件
                if self.process_single_file(
                    str(file_path),
                    str(output_file),
                    remove_noise_flag,
                    align_ground,
                    output_format,
                    crop_local_region,
                    max_distance,
                    front_axis,
                    keep_front_only,
                    visualize,
                    create_regular_grid,
                    save_intermediate,
                    intermediate_dir,
                    ground_min_inliers,
                    ground_max_angle,
                    use_enhanced_ground,
                    ground_leveling,
                    texture_noise,
                ):
                    processed_files.append(str(file_path))

        self.logger.info(f"批量处理完成，共处理 {len(processed_files)} 个文件")
        return processed_files


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description="点云自动体素化工具")
    parser.add_argument("input", help="输入点云文件或目录")
    parser.add_argument("-o", "--output", help="输出文件或目录")
    parser.add_argument(
        "-v",
        "--voxel-size",
        type=float,
        default=0.005,
        help="体素尺寸（米），默认0.005（5mm）",
    )
    parser.add_argument(
        "--batch", action="store_true", help="批量处理模式（输入为目录）"
    )
    parser.add_argument("--no-noise-removal", action="store_true", help="禁用噪声去除")
    parser.add_argument("--no-ground-align", action="store_true", help="禁用地面对齐")
    parser.add_argument(
        "--format",
        choices=["ply", "pcd", "xyz"],
        default="ply",
        help="输出格式，默认ply",
    )
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    # 新增的局部路径规划选项
    parser.add_argument(
        "--local-planning",
        action="store_true",
        help="启用局部路径规划模式（50cm半圆+1cm体素）",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=10.0,
        help="最大距离阈值（米），默认10.0（1000cm）",
    )
    parser.add_argument(
        "--front-axis",
        choices=["x", "y", "z", "auto"],
        default="auto",
        help="前向轴，默认auto自动检测",
    )
    parser.add_argument("--no-front-crop", action="store_true", help="禁用前半空间裁剪")
    parser.add_argument("--visualize", action="store_true", help="启用可视化窗口")

    # 增强版地面检测选项
    parser.add_argument(
        "--ground-min-inliers",
        type=float,
        default=0.08,
        help="地面检测最小内点比例，默认0.08（8%）",
    )
    parser.add_argument(
        "--ground-max-angle",
        type=float,
        default=60.0,
        help="地面最大倾斜角度（度），默认60.0",
    )
    parser.add_argument(
        "--enhanced-ground",
        action="store_true",
        help="强制使用增强版地面检测（即使不是路径规划模式）",
    )
    parser.add_argument(
        "--no-ground-leveling", action="store_true", help="禁用地面体素平整处理"
    )
    parser.add_argument(
        "--no-texture-noise", action="store_true", help="禁用地面纹理噪声"
    )

    args = parser.parse_args()

    # 局部路径规划模式：自动设置参数
    if args.local_planning:
        args.voxel_size = 0.005  # 5mm体素（更密集）
        args.max_distance = 10.0  # 10.0m距离（扩大范围）
        args.front_axis = "auto"
        args.no_front_crop = False
        args.no_noise_removal = True  # 路径规划模式下减少计算量
        args.no_ground_align = True  # 路径规划不需要地面对齐

    # 创建处理器
    processor = PointCloudVoxelizer(voxel_size=args.voxel_size, debug=args.debug)

    # 设置输出目录结构 - 基于输入文件路径
    if args.batch:
        # 批量模式：输入是目录
        input_path = Path(args.input)
        if input_path.is_dir():
            output_base = input_path.parent / "outputs" / "output_gaussians"
        else:
            output_base = input_path.parent.parent / "outputs" / "output_gaussians"
    else:
        # 单文件模式：输入是文件
        input_path = Path(args.input)
        output_base = input_path.parent

    if args.local_planning:
        cropped_dir = str(output_base / "cropped_pointclouds")
        voxelized_dir = str(output_base / "voxelized_outputs")
    else:
        cropped_dir = str(output_base)
        voxelized_dir = str(output_base)

    if args.batch:
        # 批量处理模式
        if not args.output:
            args.output = voxelized_dir

        processed = processor.process_batch(
            args.input,
            args.output,
            remove_noise_flag=not args.no_noise_removal,
            align_ground=not args.no_ground_align,
            output_format=args.format,
            crop_local_region=args.local_planning or not args.no_front_crop,
            max_distance=args.max_distance,
            front_axis=args.front_axis,
            keep_front_only=not args.no_front_crop,
            visualize=args.visualize,
            create_regular_grid=args.local_planning,
            save_intermediate=args.local_planning,
            intermediate_dir=cropped_dir,
            ground_min_inliers=args.ground_min_inliers,
            ground_max_angle=args.ground_max_angle,
            use_enhanced_ground=args.enhanced_ground,
            ground_leveling=not args.no_ground_leveling,
            texture_noise=not args.no_texture_noise,
        )

        print(f"成功处理 {len(processed)} 个文件")

    else:
        # 单文件处理
        if not args.output:
            if args.local_planning:
                args.output = (
                    f"{voxelized_dir}/voxelized_{os.path.basename(args.input)}"
                )
            else:
                input_path = Path(args.input)
                args.output = str(input_path.parent / f"voxelized_{input_path.name}")

        success = processor.process_single_file(
            args.input,
            args.output,
            remove_noise_flag=not args.no_noise_removal,
            align_ground=not args.no_ground_align,
            output_format=args.format,
            crop_local_region=args.local_planning or not args.no_front_crop,
            max_distance=args.max_distance,
            front_axis=args.front_axis,
            keep_front_only=not args.no_front_crop,
            visualize=args.visualize,
            create_regular_grid=args.local_planning,
            save_intermediate=args.local_planning,
            intermediate_dir=cropped_dir,
            ground_min_inliers=args.ground_min_inliers,
            ground_max_angle=args.ground_max_angle,
            use_enhanced_ground=args.enhanced_ground,
            ground_leveling=not args.no_ground_leveling,
            texture_noise=not args.no_texture_noise,
        )

        if success:
            print("处理完成！")
        else:
            print("处理失败！")
            sys.exit(1)


if __name__ == "__main__":
    main()
