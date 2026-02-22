"""点云体素化处理模块。

提供点云读取、滤波、地面检测、体素化下采样和坐标系转换功能。
支持GPU加速（CUDA/MPS）和CPU fallback。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

# 尝试导入PyTorch，检测GPU支持
try:
    import torch

    HAS_TORCH = True
    # 检测GPU设备
    if torch.cuda.is_available():
        GPU_DEVICE = "cuda"
        HAS_GPU = True
        logger.info(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        GPU_DEVICE = "mps"
        HAS_GPU = True
        logger.info("Apple MPS GPU available")
    else:
        GPU_DEVICE = "cpu"
        HAS_GPU = False
        logger.info("No GPU available, PyTorch will use CPU")
except ImportError:
    HAS_TORCH = False
    HAS_GPU = False
    GPU_DEVICE = "cpu"
    logger.info("PyTorch not available, GPU acceleration disabled")


@dataclass
class VoxelizerConfig:
    """体素化配置参数。"""

    voxel_size: float = 0.05
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0
    ransac_distance_threshold: float = 0.02
    ransac_n_points: int = 3
    ransac_num_iterations: int = 1000
    use_gpu: bool = True  # 是否使用GPU加速（如果可用）
    gpu_device: str = "auto"  # GPU设备：auto/cuda/mps/cpu


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
        self._device = self._get_device()

    def _get_device(self) -> str:
        """获取计算设备。"""
        if not self.config.use_gpu or not HAS_GPU:
            return "cpu"

        device_config = self.config.gpu_device.lower()
        if device_config == "auto":
            return GPU_DEVICE
        elif device_config == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device_config == "mps" and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def _should_use_gpu(self) -> bool:
        """检查是否应该使用GPU。"""
        return HAS_GPU and self.config.use_gpu and self._device != "cpu"

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

        if self._should_use_gpu():
            return self._remove_outliers_gpu(pc, cfg)
        elif HAS_OPEN3D:
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

    def _remove_outliers_gpu(self, pc: PointCloud, cfg: VoxelizerConfig) -> PointCloud:
        """使用PyTorch GPU加速去除离群点。"""
        n_points = pc.points.shape[0]

        # 对于大点云，GPU内存可能不足，自动降级到CPU
        # 估算内存需求：batch_size * n_points * 4 bytes * 3 (xyz)
        max_gpu_points = 500000  # 超过50万点时降级到CPU
        if n_points > max_gpu_points:
            logger.warning(
                f"Point cloud too large ({n_points} points) for GPU outlier removal, "
                f"falling back to CPU"
            )
            if HAS_OPEN3D:
                return self._remove_outliers_o3d(pc, cfg)
            return self._remove_outliers_numpy(pc, cfg)

        logger.info(f"Using GPU ({self._device}) for outlier removal")
        device = torch.device(self._device)

        try:
            # 转换为tensor并移到GPU
            points_tensor = torch.tensor(pc.points, dtype=torch.float32, device=device)
            k = cfg.statistical_nb_neighbors + 1

            # 动态调整batch_size以适应内存
            # 每个batch需要 batch_size * n_points * 4 bytes
            batch_size = min(5000, n_points)
            mean_distances = torch.zeros(n_points, device=device)

            for i in range(0, n_points, batch_size):
                end_idx = min(i + batch_size, n_points)
                batch_points = points_tensor[i:end_idx]

                # 计算批次点到所有点的距离
                diff = batch_points.unsqueeze(1) - points_tensor.unsqueeze(0)
                dists = torch.sqrt((diff**2).sum(dim=2))

                # 获取k个最近邻的距离（排除自身）
                knn_dists, _ = torch.topk(dists, k, largest=False, dim=1)
                mean_distances[i:end_idx] = knn_dists[:, 1:].mean(dim=1)

            # 计算阈值并过滤
            threshold = (
                mean_distances.mean() + cfg.statistical_std_ratio * mean_distances.std()
            )
            mask = mean_distances < threshold

            # 转回numpy
            mask_np = mask.cpu().numpy()
            logger.info(f"Removed {(~mask_np).sum()} outliers (GPU)")

            colors = pc.colors[mask_np] if pc.colors is not None else None
            return PointCloud(points=pc.points[mask_np], colors=colors)

        except RuntimeError as e:
            # GPU内存不足，降级到CPU
            if (
                "out of memory" in str(e).lower()
                or "invalid buffer size" in str(e).lower()
            ):
                logger.warning(f"GPU memory error: {e}, falling back to CPU")
                if HAS_OPEN3D:
                    return self._remove_outliers_o3d(pc, cfg)
                return self._remove_outliers_numpy(pc, cfg)
            raise

    def detect_ground_ransac(
        self, pc: PointCloud
    ) -> tuple[PointCloud, NDArray[np.float64]]:
        """RANSAC地面检测，返回非地面点和地面平面参数。"""
        logger.info("Detecting ground plane with RANSAC")
        cfg = self.config

        if self._should_use_gpu():
            return self._detect_ground_gpu(pc, cfg)
        elif HAS_OPEN3D:
            return self._detect_ground_o3d(pc, cfg)
        return self._detect_ground_numpy(pc, cfg)

    def _detect_ground_o3d(
        self, pc: PointCloud, cfg: VoxelizerConfig
    ) -> tuple[PointCloud, NDArray[np.float64]]:
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
    ) -> tuple[PointCloud, NDArray[np.float64]]:
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

    def _detect_ground_gpu(
        self, pc: PointCloud, cfg: VoxelizerConfig
    ) -> tuple[PointCloud, NDArray[np.float64]]:
        """使用PyTorch GPU加速进行RANSAC地面检测。"""
        logger.info(f"Using GPU ({self._device}) for ground detection")
        device = torch.device(self._device)

        points_tensor = torch.tensor(pc.points, dtype=torch.float32, device=device)
        n_points = points_tensor.shape[0]

        best_inlier_count = 0
        best_inliers_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
        best_plane = torch.zeros(4, device=device)

        # 批量RANSAC迭代
        batch_iterations = min(100, cfg.ransac_num_iterations)
        num_batches = cfg.ransac_num_iterations // batch_iterations

        for _ in range(num_batches):
            # 批量采样随机点
            indices = torch.randint(
                0, n_points, (batch_iterations, cfg.ransac_n_points), device=device
            )
            sampled_points = points_tensor[indices]  # (batch, 3, 3)

            p1 = sampled_points[:, 0, :]
            p2 = sampled_points[:, 1, :]
            p3 = sampled_points[:, 2, :]

            # 计算法向量
            v1 = p2 - p1
            v2 = p3 - p1
            normals = torch.cross(v1, v2, dim=1)
            norms = torch.norm(normals, dim=1, keepdim=True)

            # 过滤退化情况
            valid_mask = norms.squeeze() > 1e-10
            if not valid_mask.any():
                continue

            normals = normals / (norms + 1e-10)
            d = -(normals * p1).sum(dim=1, keepdim=True)

            # 计算所有点到每个平面的距离
            # points_tensor: (n_points, 3), normals: (batch, 3)
            distances = torch.abs(
                torch.mm(points_tensor, normals.T) + d.T
            )  # (n_points, batch)

            # 统计内点数
            inliers_count = (distances < cfg.ransac_distance_threshold).sum(dim=0)

            # 找最佳平面
            best_idx = inliers_count.argmax()
            if inliers_count[best_idx] > best_inlier_count and valid_mask[best_idx]:
                best_inlier_count = inliers_count[best_idx].item()
                best_inliers_mask = (
                    distances[:, best_idx] < cfg.ransac_distance_threshold
                )
                best_plane = torch.cat([normals[best_idx], d[best_idx]])

        # 转回numpy
        mask_np = ~best_inliers_mask.cpu().numpy()
        plane_np = best_plane.cpu().numpy()

        logger.info(f"Detected ground with {best_inlier_count} points (GPU)")
        colors = pc.colors[mask_np] if pc.colors is not None else None
        return PointCloud(points=pc.points[mask_np], colors=colors), plane_np

    def voxel_downsample(self, pc: PointCloud) -> PointCloud:
        """体素化下采样。"""
        logger.info(f"Voxel downsampling with size {self.config.voxel_size}")

        if self._should_use_gpu():
            return self._voxel_downsample_gpu(pc)
        elif HAS_OPEN3D:
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
        voxel_dict: dict[tuple[int, ...], list[int]] = {}
        for i, key in enumerate(map(tuple, voxel_indices)):
            voxel_dict.setdefault(key, []).append(i)

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

    def _voxel_downsample_gpu(self, pc: PointCloud) -> PointCloud:
        """使用PyTorch GPU加速进行体素下采样。"""
        logger.info(f"Using GPU ({self._device}) for voxel downsampling")
        device = torch.device(self._device)

        voxel_size = self.config.voxel_size
        points_tensor = torch.tensor(pc.points, dtype=torch.float32, device=device)

        # 计算体素索引
        voxel_indices = torch.floor(points_tensor / voxel_size).to(torch.int64)

        # 使用unique找到唯一的体素
        # 将3D索引转换为1D哈希值
        min_idx = voxel_indices.min(dim=0).values
        voxel_indices_shifted = voxel_indices - min_idx
        max_idx = voxel_indices_shifted.max(dim=0).values + 1

        # 计算1D索引
        hash_values = (
            voxel_indices_shifted[:, 0] * max_idx[1] * max_idx[2]
            + voxel_indices_shifted[:, 1] * max_idx[2]
            + voxel_indices_shifted[:, 2]
        )

        # 获取唯一体素和逆索引
        unique_hashes, inverse_indices = torch.unique(hash_values, return_inverse=True)
        n_voxels = len(unique_hashes)

        # 使用scatter_add计算每个体素的点数和坐标和
        counts = torch.zeros(n_voxels, device=device)
        counts.scatter_add_(
            0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32)
        )

        sum_points = torch.zeros(n_voxels, 3, device=device)
        for dim in range(3):
            sum_points[:, dim].scatter_add_(0, inverse_indices, points_tensor[:, dim])

        # 计算平均值
        new_points = sum_points / counts.unsqueeze(1)

        # 处理颜色
        new_colors = None
        if pc.colors is not None:
            colors_tensor = torch.tensor(pc.colors, dtype=torch.float32, device=device)
            sum_colors = torch.zeros(n_voxels, 3, device=device)
            for dim in range(3):
                sum_colors[:, dim].scatter_add_(
                    0, inverse_indices, colors_tensor[:, dim]
                )
            new_colors = (sum_colors / counts.unsqueeze(1)).cpu().numpy()

        points_np = new_points.cpu().numpy()
        logger.info(f"Downsampled to {len(points_np)} points (GPU)")
        return PointCloud(points=points_np, colors=new_colors)

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
