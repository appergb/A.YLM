"""多帧点云配准融合模块。

使用 Open3D 实现 ICP 配准和位姿图优化，将多帧点云融合成全局地图。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .pointcloud_voxelizer import PointCloud

logger = logging.getLogger(__name__)

# 检查 Open3D 是否可用
try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None  # type: ignore[assignment]


@dataclass
class RegistrationConfig:
    """配准配置参数。"""

    # ICP 配准参数
    icp_max_correspondence_distance: float = 0.05  # ICP 最大对应点距离（米）
    icp_max_iteration: int = 50  # ICP 最大迭代次数
    icp_relative_fitness: float = 1e-6  # 相对适应度收敛阈值
    icp_relative_rmse: float = 1e-6  # 相对 RMSE 收敛阈值

    # 特征提取参数
    voxel_size_for_features: float = 0.05  # 特征提取用的体素大小
    fpfh_radius: float = 0.25  # FPFH 特征半径
    fpfh_max_nn: int = 100  # FPFH 最大近邻数
    normal_radius: float = 0.1  # 法向量估计半径

    # 位姿图优化参数
    pose_graph_edge_prune_threshold: float = 0.25  # 边修剪阈值
    pose_graph_preference_loop_closure: float = 0.1  # 闭环偏好

    # 全局融合参数
    fusion_voxel_size: float = 0.02  # 融合后体素下采样大小

    # 配准质量阈值
    min_fitness: float = 0.3  # 最小配准适应度
    min_points: int = 1000  # 最小点数要求


@dataclass
class FramePose:
    """单帧位姿信息。"""

    frame_index: int  # 帧索引
    transformation: NDArray[np.float64]  # 4x4 变换矩阵
    fitness: float = 0.0  # 配准适应度
    rmse: float = 0.0  # 配准 RMSE
    is_keyframe: bool = False  # 是否为关键帧

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "index": self.frame_index,
            "transformation": self.transformation.tolist(),
            "fitness": self.fitness,
            "rmse": self.rmse,
            "is_keyframe": self.is_keyframe,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FramePose:
        """从字典创建。"""
        return cls(
            frame_index=data["index"],
            transformation=np.array(data["transformation"]),
            fitness=data.get("fitness", 0.0),
            rmse=data.get("rmse", 0.0),
            is_keyframe=data.get("is_keyframe", False),
        )


@dataclass
class RegistrationResult:
    """配准结果。"""

    transformation: NDArray[np.float64]  # 4x4 变换矩阵
    fitness: float  # 适应度
    inlier_rmse: float  # 内点 RMSE
    correspondence_count: int  # 对应点数量


@dataclass
class FusionResult:
    """融合结果。"""

    fused_pointcloud: PointCloud  # 融合后的点云
    frame_poses: list[FramePose] = field(default_factory=list)  # 所有帧的位姿
    total_frames: int = 0  # 总帧数
    successful_registrations: int = 0  # 成功配准数
    fusion_time: float = 0.0  # 融合耗时


class MultiframeFusion:
    """多帧点云配准融合器。

    使用 ICP 算法进行相邻帧配准，位姿图优化进行全局一致性调整，
    最终将所有帧变换到世界坐标系并合并。

    Example:
        >>> config = RegistrationConfig(icp_max_correspondence_distance=0.05)
        >>> fusion = MultiframeFusion(config)
        >>> result = fusion.fuse_sequence(pointcloud_list)
        >>> fusion.save_fused_map(result, "fused_map.ply")
    """

    def __init__(self, config: RegistrationConfig | None = None):
        """初始化融合器。

        Args:
            config: 配准配置，为 None 时使用默认配置
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D 未安装，请运行: pip install open3d")

        self.config = config or RegistrationConfig()
        logger.info("MultiframeFusion 初始化完成")

    def fuse_sequence(
        self,
        pointclouds: list[PointCloud],
        initial_poses: list[NDArray[np.float64]] | None = None,
    ) -> FusionResult:
        """融合点云序列。

        Args:
            pointclouds: 点云列表（按时间顺序）
            initial_poses: 可选的初始位姿估计

        Returns:
            FusionResult: 融合结果
        """
        start_time = time.time()
        n_frames = len(pointclouds)

        if n_frames == 0:
            logger.warning("输入点云列表为空")
            return FusionResult(fused_pointcloud=PointCloud(np.zeros((0, 3))))

        if n_frames == 1:
            logger.info("只有一帧，直接返回")
            pose = FramePose(0, np.eye(4), 1.0, 0.0, True)
            return FusionResult(
                fused_pointcloud=pointclouds[0],
                frame_poses=[pose],
                total_frames=1,
                successful_registrations=1,
                fusion_time=time.time() - start_time,
            )

        logger.info(f"开始融合 {n_frames} 帧点云")

        # 1. 预处理所有点云
        logger.info("预处理点云...")
        processed = []
        valid_indices = []
        for i, pc in enumerate(pointclouds):
            if len(pc.points) < self.config.min_points:
                logger.warning(f"帧 {i} 点数不足 ({len(pc.points)}), 跳过")
                continue
            pcd, fpfh = self._preprocess_pointcloud(pc)
            processed.append((pcd, fpfh, pc))
            valid_indices.append(i)

        if len(processed) < 2:
            logger.error("有效帧数不足，无法配准")
            return FusionResult(fused_pointcloud=PointCloud(np.zeros((0, 3))))

        # 2. 相邻帧配准
        logger.info("执行相邻帧配准...")
        pairwise_results: list[RegistrationResult | None] = []
        for i in range(len(processed) - 1):
            source_pcd, source_fpfh, _ = processed[i + 1]
            target_pcd, target_fpfh, _ = processed[i]

            result = self._register_pair_o3d(
                source_pcd, target_pcd, source_fpfh, target_fpfh
            )

            if result.fitness < self.config.min_fitness:
                logger.warning(
                    f"帧 {valid_indices[i+1]} -> {valid_indices[i]} "
                    f"配准质量低 (fitness={result.fitness:.3f})"
                )
            pairwise_results.append(result)
            logger.debug(
                f"帧 {valid_indices[i+1]} -> {valid_indices[i]}: "
                f"fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.4f}"
            )

        # 3. 构建位姿图
        logger.info("构建位姿图...")
        pose_graph = self._build_pose_graph(processed, pairwise_results)

        # 4. 优化位姿图
        logger.info("优化位姿图...")
        pose_graph = self._optimize_pose_graph(pose_graph)

        # 5. 提取优化后的位姿
        frame_poses = []
        successful = 0
        for i, node in enumerate(pose_graph.nodes):
            original_idx = valid_indices[i]
            prev_result = pairwise_results[i - 1] if i > 0 else None
            pose = FramePose(
                frame_index=original_idx,
                transformation=np.asarray(node.pose),
                fitness=prev_result.fitness if prev_result else 1.0,
                rmse=prev_result.inlier_rmse if prev_result else 0.0,
                is_keyframe=(i == 0),
            )
            frame_poses.append(pose)
            if i == 0 or (
                prev_result and prev_result.fitness >= self.config.min_fitness
            ):
                successful += 1

        # 6. 变换并合并点云
        logger.info("合并点云...")
        original_pcs = [p[2] for p in processed]
        fused_pc = self._transform_and_merge(original_pcs, frame_poses)

        # 7. 体素下采样
        logger.info("体素下采样...")
        fused_pc = self._voxel_downsample(fused_pc, self.config.fusion_voxel_size)

        fusion_time = time.time() - start_time
        logger.info(
            f"融合完成: {len(fused_pc.points)} 点, "
            f"{successful}/{len(processed)} 帧成功, 耗时 {fusion_time:.2f}s"
        )

        return FusionResult(
            fused_pointcloud=fused_pc,
            frame_poses=frame_poses,
            total_frames=n_frames,
            successful_registrations=successful,
            fusion_time=fusion_time,
        )

    def fuse_from_directory(
        self,
        input_dir: Path,
        pattern: str = "vox_*.ply",
    ) -> FusionResult:
        """从目录加载并融合点云序列。

        Args:
            input_dir: 输入目录
            pattern: 文件匹配模式

        Returns:
            FusionResult: 融合结果
        """
        input_dir = Path(input_dir)
        ply_files = sorted(input_dir.glob(pattern))

        if not ply_files:
            logger.error(f"未找到匹配 {pattern} 的文件: {input_dir}")
            return FusionResult(fused_pointcloud=PointCloud(np.zeros((0, 3))))

        logger.info(f"加载 {len(ply_files)} 个点云文件")
        pointclouds = []
        for ply_path in ply_files:
            pc = PointCloud.from_ply(ply_path)
            pointclouds.append(pc)
            logger.debug(f"加载 {ply_path.name}: {len(pc.points)} 点")

        return self.fuse_sequence(pointclouds)

    def register_pair(
        self,
        source: PointCloud,
        target: PointCloud,
        initial_transform: NDArray[np.float64] | None = None,
    ) -> RegistrationResult:
        """配准两帧点云。

        Args:
            source: 源点云
            target: 目标点云
            initial_transform: 初始变换估计

        Returns:
            RegistrationResult: 配准结果
        """
        source_pcd, source_fpfh = self._preprocess_pointcloud(source)
        target_pcd, target_fpfh = self._preprocess_pointcloud(target)

        if initial_transform is not None:
            # 直接使用 ICP 精配准
            return self._fine_registration(source_pcd, target_pcd, initial_transform)
        else:
            # 粗配准 + 精配准
            return self._register_pair_o3d(
                source_pcd, target_pcd, source_fpfh, target_fpfh
            )

    def save_fused_map(
        self,
        result: FusionResult,
        output_path: Path,
        include_poses: bool = True,
    ) -> None:
        """保存融合后的地图。

        Args:
            result: 融合结果
            output_path: 输出 PLY 文件路径
            include_poses: 是否同时保存位姿文件
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存点云
        result.fused_pointcloud.to_ply(output_path)
        logger.info(f"融合地图已保存: {output_path}")

        # 保存位姿
        if include_poses and result.frame_poses:
            poses_path = output_path.with_suffix(".poses.json")
            self.save_poses(result.frame_poses, poses_path)

    def save_poses(
        self,
        poses: list[FramePose],
        output_path: Path,
    ) -> None:
        """保存位姿轨迹到 JSON 文件。

        Args:
            poses: 位姿列表
            output_path: 输出路径
        """
        output_path = Path(output_path)

        data = {
            "version": "1.0",
            "coordinate_system": "opencv",
            "frames": [p.to_dict() for p in poses],
            "statistics": {
                "total_frames": len(poses),
                "average_fitness": np.mean([p.fitness for p in poses]),
                "average_rmse": np.mean([p.rmse for p in poses]),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"位姿轨迹已保存: {output_path}")

    # ========== 内部方法 ==========

    def _to_o3d(self, pc: PointCloud) -> o3d.geometry.PointCloud:
        """转换为 Open3D 点云格式。"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points.astype(np.float64))
        if pc.colors is not None:
            colors = pc.colors.astype(np.float64)
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _from_o3d(self, pcd: o3d.geometry.PointCloud) -> PointCloud:
        """从 Open3D 点云格式转换。"""
        points = np.asarray(pcd.points)
        colors = None
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        return PointCloud(points=points, colors=colors)

    def _preprocess_pointcloud(
        self,
        pc: PointCloud,
    ) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """预处理点云：下采样 + 计算法向量 + 提取 FPFH 特征。"""
        pcd = self._to_o3d(pc)

        # 体素下采样
        pcd_down = pcd.voxel_down_sample(self.config.voxel_size_for_features)

        # 估计法向量
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.normal_radius, max_nn=30
            )
        )

        # 计算 FPFH 特征
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.fpfh_radius, max_nn=self.config.fpfh_max_nn
            ),
        )

        return pcd_down, fpfh

    def _register_pair_o3d(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        source_fpfh: o3d.pipelines.registration.Feature,
        target_fpfh: o3d.pipelines.registration.Feature,
    ) -> RegistrationResult:
        """配准两帧点云（粗配准 + 精配准）。"""
        # 粗配准: RANSAC
        coarse_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd,
            target_pcd,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.config.icp_max_correspondence_distance * 2,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.config.icp_max_correspondence_distance * 2
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=4000000, confidence=0.999
            ),
        )

        # 精配准: Point-to-Plane ICP
        return self._fine_registration(
            source_pcd, target_pcd, coarse_result.transformation
        )

    def _fine_registration(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        initial_transform: NDArray[np.float64],
    ) -> RegistrationResult:
        """精配准：Point-to-Plane ICP。"""
        # 确保有法向量
        if not target_pcd.has_normals():
            target_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_radius, max_nn=30
                )
            )

        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            self.config.icp_max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.config.icp_relative_fitness,
                relative_rmse=self.config.icp_relative_rmse,
                max_iteration=self.config.icp_max_iteration,
            ),
        )

        return RegistrationResult(
            transformation=np.asarray(result.transformation),
            fitness=result.fitness,
            inlier_rmse=result.inlier_rmse,
            correspondence_count=len(result.correspondence_set),
        )

    def _build_pose_graph(
        self,
        processed: list[tuple],
        pairwise_results: list[RegistrationResult | None],
    ) -> o3d.pipelines.registration.PoseGraph:
        """构建位姿图。"""
        pose_graph = o3d.pipelines.registration.PoseGraph()
        n_frames = len(processed)

        # 添加节点
        odometry = np.eye(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

        for i in range(1, n_frames):
            result = pairwise_results[i - 1]
            if result is not None:
                odometry = result.transformation @ odometry
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

        # 添加边（相邻帧）
        for i in range(n_frames - 1):
            result = pairwise_results[i]
            if result is None:
                continue

            # 信息矩阵（基于配准质量）
            information = (
                np.eye(6) * result.fitness if result.fitness > 0 else np.eye(6) * 0.1
            )

            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i,
                    i + 1,
                    result.transformation,
                    information,
                    uncertain=False,
                )
            )

        return pose_graph

    def _optimize_pose_graph(
        self,
        pose_graph: o3d.pipelines.registration.PoseGraph,
    ) -> o3d.pipelines.registration.PoseGraph:
        """优化位姿图。"""
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.config.icp_max_correspondence_distance,
            edge_prune_threshold=self.config.pose_graph_edge_prune_threshold,
            preference_loop_closure=self.config.pose_graph_preference_loop_closure,
            reference_node=0,
        )

        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        return pose_graph

    def _transform_and_merge(
        self,
        pointclouds: list[PointCloud],
        poses: list[FramePose],
    ) -> PointCloud:
        """将所有点云变换到世界坐标系并合并。"""
        all_points = []
        all_colors = []
        has_colors = pointclouds[0].colors is not None

        for pc, pose in zip(pointclouds, poses):
            # 应用变换
            points_h = np.hstack([pc.points, np.ones((len(pc.points), 1))])
            transformed = (pose.transformation @ points_h.T).T[:, :3]
            all_points.append(transformed)

            if has_colors and pc.colors is not None:
                all_colors.append(pc.colors)

        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors) if has_colors else None

        return PointCloud(points=merged_points, colors=merged_colors)

    def _voxel_downsample(
        self,
        pc: PointCloud,
        voxel_size: float,
    ) -> PointCloud:
        """体素下采样去重。"""
        pcd = self._to_o3d(pc)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        return self._from_o3d(pcd_down)
