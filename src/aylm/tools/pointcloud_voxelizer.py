"""点云体素化处理模块。

提供点云读取、滤波、地面检测、体素化下采样和坐标系转换功能。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)

# 尝试导入Open3D，如果不可用则使用numpy实现
try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.info("Open3D not available, using numpy fallback")


@dataclass
class VoxelizerConfig:
    """体素化配置参数。"""

    voxel_size: float = 0.05
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0
    ransac_distance_threshold: float = 0.02
    ransac_n_points: int = 3
    ransac_num_iterations: int = 1000


@dataclass
class PointCloud:
    """点云数据结构。"""

    points: NDArray[np.float64]
    colors: Optional[NDArray[np.float64]] = None
    normals: Optional[NDArray[np.float64]] = None


class PointCloudVoxelizer:
    """点云体素化处理器。"""

    def __init__(self, config: Optional[VoxelizerConfig] = None):
        self.config = config or VoxelizerConfig()

    def load_ply(self, filepath: Path) -> PointCloud:
        """从PLY文件加载点云。"""
        logger.info(f"Loading point cloud from {filepath}")
        plydata = PlyData.read(str(filepath))
        vertex = plydata["vertex"]

        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])
        colors = None
        if all(p in vertex.data.dtype.names for p in ["red", "green", "blue"]):
            colors = (
                np.column_stack(
                    [vertex["red"], vertex["green"], vertex["blue"]]
                ).astype(np.float64)
                / 255.0
            )

        logger.info(f"Loaded {len(points)} points")
        return PointCloud(points=points, colors=colors)

    def save_ply(self, pc: PointCloud, filepath: Path) -> None:
        """保存点云到PLY文件。"""
        logger.info(f"Saving point cloud to {filepath}")
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        if pc.colors is not None:
            dtype.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

        data = np.zeros(len(pc.points), dtype=dtype)
        data["x"], data["y"], data["z"] = pc.points.T

        if pc.colors is not None:
            colors_uint8 = (pc.colors * 255).astype(np.uint8)
            data["red"], data["green"], data["blue"] = colors_uint8.T

        vertex = PlyElement.describe(data, "vertex")
        PlyData([vertex], text=True).write(str(filepath))
        logger.info(f"Saved {len(pc.points)} points")

    def remove_statistical_outliers(self, pc: PointCloud) -> PointCloud:
        """统计离群点去除。"""
        logger.info("Removing statistical outliers")
        cfg = self.config

        if HAS_OPEN3D:
            return self._remove_outliers_o3d(pc, cfg)
        return self._remove_outliers_numpy(pc, cfg)

    def _remove_outliers_o3d(self, pc: PointCloud, cfg: VoxelizerConfig) -> PointCloud:
        """使用Open3D去除离群点。"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)
        if pc.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(pc.colors)

        pcd_clean, indices = pcd.remove_statistical_outlier(
            nb_neighbors=cfg.statistical_nb_neighbors,
            std_ratio=cfg.statistical_std_ratio,
        )
        indices = np.asarray(indices)
        logger.info(f"Removed {len(pc.points) - len(indices)} outliers")

        colors = pc.colors[indices] if pc.colors is not None else None
        return PointCloud(points=pc.points[indices], colors=colors)

    def _remove_outliers_numpy(
        self, pc: PointCloud, cfg: VoxelizerConfig
    ) -> PointCloud:
        """使用numpy去除离群点（KNN距离法）。"""
        from scipy.spatial import KDTree

        tree = KDTree(pc.points)
        distances, _ = tree.query(pc.points, k=cfg.statistical_nb_neighbors + 1)
        mean_distances = distances[:, 1:].mean(axis=1)

        threshold = (
            mean_distances.mean() + cfg.statistical_std_ratio * mean_distances.std()
        )
        mask = mean_distances < threshold

        logger.info(f"Removed {(~mask).sum()} outliers")
        colors = pc.colors[mask] if pc.colors is not None else None
        return PointCloud(points=pc.points[mask], colors=colors)

    def detect_ground_ransac(
        self, pc: PointCloud
    ) -> Tuple[PointCloud, NDArray[np.float64]]:
        """RANSAC地面检测，返回非地面点和地面平面参数。"""
        logger.info("Detecting ground plane with RANSAC")
        cfg = self.config

        if HAS_OPEN3D:
            return self._detect_ground_o3d(pc, cfg)
        return self._detect_ground_numpy(pc, cfg)

    def _detect_ground_o3d(
        self, pc: PointCloud, cfg: VoxelizerConfig
    ) -> Tuple[PointCloud, NDArray[np.float64]]:
        """使用Open3D进行RANSAC地面检测。"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=cfg.ransac_distance_threshold,
            ransac_n=cfg.ransac_n_points,
            num_iterations=cfg.ransac_num_iterations,
        )
        mask = np.ones(len(pc.points), dtype=bool)
        mask[inliers] = False

        logger.info(f"Detected ground with {len(inliers)} points")
        colors = pc.colors[mask] if pc.colors is not None else None
        return PointCloud(points=pc.points[mask], colors=colors), np.array(plane_model)

    def _detect_ground_numpy(
        self, pc: PointCloud, cfg: VoxelizerConfig
    ) -> Tuple[PointCloud, NDArray[np.float64]]:
        """使用numpy实现RANSAC地面检测。"""
        points = pc.points
        best_inliers = np.array([], dtype=int)
        best_plane = np.zeros(4)

        for _ in range(cfg.ransac_num_iterations):
            idx = np.random.choice(len(points), cfg.ransac_n_points, replace=False)
            p1, p2, p3 = points[idx]

            v1, v2 = p2 - p1, p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            d = -np.dot(normal, p1)

            distances = np.abs(points @ normal + d)
            inliers = np.where(distances < cfg.ransac_distance_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = np.append(normal, d)

        mask = np.ones(len(points), dtype=bool)
        mask[best_inliers] = False

        logger.info(f"Detected ground with {len(best_inliers)} points")
        colors = pc.colors[mask] if pc.colors is not None else None
        return PointCloud(points=points[mask], colors=colors), best_plane

    def voxel_downsample(self, pc: PointCloud) -> PointCloud:
        """体素化下采样。"""
        logger.info(f"Voxel downsampling with size {self.config.voxel_size}")

        if HAS_OPEN3D:
            return self._voxel_downsample_o3d(pc)
        return self._voxel_downsample_numpy(pc)

    def _voxel_downsample_o3d(self, pc: PointCloud) -> PointCloud:
        """使用Open3D进行体素下采样。"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)
        if pc.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(pc.colors)

        pcd_down = pcd.voxel_down_sample(self.config.voxel_size)
        points = np.asarray(pcd_down.points)
        colors = np.asarray(pcd_down.colors) if pc.colors is not None else None

        logger.info(f"Downsampled to {len(points)} points")
        return PointCloud(points=points, colors=colors)

    def _voxel_downsample_numpy(self, pc: PointCloud) -> PointCloud:
        """使用numpy进行体素下采样。"""
        voxel_size = self.config.voxel_size
        voxel_indices = np.floor(pc.points / voxel_size).astype(np.int32)

        # 使用字典聚合每个体素内的点
        voxel_dict: dict = {}
        for i, key in enumerate(map(tuple, voxel_indices)):
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        new_points = []
        new_colors = [] if pc.colors is not None else None

        for indices in voxel_dict.values():
            new_points.append(pc.points[indices].mean(axis=0))
            if pc.colors is not None:
                new_colors.append(pc.colors[indices].mean(axis=0))

        points = np.array(new_points)
        colors = np.array(new_colors) if new_colors else None

        logger.info(f"Downsampled to {len(points)} points")
        return PointCloud(points=points, colors=colors)

    @staticmethod
    def transform_opencv_to_robot(pc: PointCloud) -> PointCloud:
        """OpenCV坐标系转换到机器人坐标系。

        OpenCV: X右, Y下, Z前
        Robot:  X前, Y左, Z上
        """
        logger.info("Transforming from OpenCV to robot coordinate system")
        # 变换矩阵: [X_robot, Y_robot, Z_robot] = [Z_cv, -X_cv, -Y_cv]
        transform = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

        new_points = pc.points @ transform.T
        return PointCloud(points=new_points, colors=pc.colors, normals=pc.normals)

    def process(
        self,
        input_path: Path,
        output_path: Path,
        remove_ground: bool = True,
        transform_coords: bool = True,
    ) -> PointCloud:
        """完整的点云处理流程。"""
        logger.info(f"Processing {input_path}")

        pc = self.load_ply(input_path)
        pc = self.remove_statistical_outliers(pc)

        if remove_ground:
            pc, _ = self.detect_ground_ransac(pc)

        pc = self.voxel_downsample(pc)

        if transform_coords:
            pc = self.transform_opencv_to_robot(pc)

        self.save_ply(pc, output_path)
        logger.info("Processing complete")
        return pc
