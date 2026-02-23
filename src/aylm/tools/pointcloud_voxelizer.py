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
    # 地面检测配置
    ground_search_height: float = 1.0  # 从 Y 最大值向下搜索地面的高度范围（米）
    ground_min_points: int = 100  # 地面区域最少点数，少于此值则跳过地面检测


@dataclass
class PointCloud:
    """点云数据结构。"""

    points: NDArray[np.float64]
    colors: Optional[NDArray[np.float64]] = None
    normals: Optional[NDArray[np.float64]] = None

    def filter_by_mask(self, mask: "NDArray[np.bool_]") -> "PointCloud":
        """根据布尔掩码过滤点云。"""
        colors = self.colors[mask] if self.colors is not None else None
        return PointCloud(points=self.points[mask], colors=colors)


class PointCloudVoxelizer:
    """点云体素化处理器。"""

    # 类常量
    _DEFAULT_GROUND_PLANE = np.array([0.0, 1.0, 0.0, 0.0])  # 默认水平面 (Y轴向下)

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
        return self._device != "cpu"

    def _to_o3d(self, pc: PointCloud) -> "o3d.geometry.PointCloud":
        """将 PointCloud 转换为 Open3D 格式。"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)
        if pc.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(pc.colors)
        return pcd

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
        pcd = self._to_o3d(pc)

        pcd_clean, indices = pcd.remove_statistical_outlier(
            nb_neighbors=cfg.statistical_nb_neighbors,
            std_ratio=cfg.statistical_std_ratio,
        )
        indices = np.asarray(indices)
        logger.info(f"Removed {len(pc.points) - len(indices)} outliers")

        return pc.filter_by_mask(np.isin(np.arange(len(pc.points)), indices))

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
        return pc.filter_by_mask(mask)

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

            return pc.filter_by_mask(mask_np)

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
        self, pc: PointCloud, ground_normal_threshold: float = 0.8
    ) -> tuple[PointCloud, NDArray[np.float64]]:
        """RANSAC地面检测，返回非地面点和地面平面参数。

        改进策略：
        1. 首先找到 Y 值最大的区域（OpenCV 坐标系中 Y 向下，Y 最大 = 最低点 = 地面）
        2. 只在这个区域进行 RANSAC 平面拟合
        3. 移除整个点云中距离地面平面较近的点

        Args:
            pc: 输入点云
            ground_normal_threshold: 地面法向量与Y轴的最小点积阈值（0-1）
                在OpenCV坐标系中，Y轴向下，地面法向量应接近[0, ±1, 0]
                默认0.8表示法向量与Y轴夹角小于约37度

        Returns:
            非地面点云和平面参数 [a, b, c, d]，其中 ax + by + cz + d = 0
        """
        logger.info("Detecting ground plane with RANSAC")
        cfg = self.config

        # 步骤1：找到地面区域（Y 值最大的区域）
        y_values = pc.points[:, 1]
        y_max = y_values.max()
        y_threshold = y_max - cfg.ground_search_height

        ground_region_mask = y_values >= y_threshold
        ground_region_count = ground_region_mask.sum()

        logger.info(
            f"Ground search region: Y >= {y_threshold:.2f}m, "
            f"found {ground_region_count} points"
        )

        # 如果地面区域点数太少，跳过地面检测
        if ground_region_count < cfg.ground_min_points:
            logger.warning(
                f"Ground region has only {ground_region_count} points "
                f"(< {cfg.ground_min_points}), skipping ground detection"
            )
            default_plane = self._DEFAULT_GROUND_PLANE.copy()
            default_plane[3] = -y_max
            return pc, default_plane

        # 步骤2：在地面区域进行 RANSAC
        ground_region_pc = pc.filter_by_mask(ground_region_mask)

        if self._should_use_gpu():
            _, plane = self._detect_ground_gpu(
                ground_region_pc, cfg, ground_normal_threshold
            )
        elif HAS_OPEN3D:
            _, plane = self._detect_ground_o3d(
                ground_region_pc, cfg, ground_normal_threshold
            )
        else:
            _, plane = self._detect_ground_numpy(
                ground_region_pc, cfg, ground_normal_threshold
            )

        # 步骤3：使用检测到的平面移除整个点云中的地面点
        a, b, c, d = plane
        distances = np.abs(
            a * pc.points[:, 0] + b * pc.points[:, 1] + c * pc.points[:, 2] + d
        )
        ground_mask = distances < cfg.ransac_distance_threshold

        # 移除地面点
        non_ground_mask = ~ground_mask
        removed_count = ground_mask.sum()

        logger.info(f"Removed {removed_count} ground points from entire point cloud")

        return pc.filter_by_mask(non_ground_mask), plane

    def _detect_ground_o3d(
        self, pc: PointCloud, cfg: VoxelizerConfig, ground_normal_threshold: float
    ) -> tuple[PointCloud, NDArray[np.float64]]:
        """使用Open3D进行RANSAC地面检测。

        添加法向量方向验证：只接受法向量与Y轴夹角小于阈值的平面。
        """
        pcd = self._to_o3d(pc)

        # 多次尝试找到符合地面法向量方向的平面
        remaining_pcd = pcd
        remaining_indices = np.arange(len(pc.points))
        best_plane = self._DEFAULT_GROUND_PLANE.copy()
        best_inliers = np.array([], dtype=int)

        for attempt in range(5):  # 最多尝试5次
            if len(remaining_pcd.points) < cfg.ransac_n_points:
                break

            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=cfg.ransac_distance_threshold,
                ransac_n=cfg.ransac_n_points,
                num_iterations=cfg.ransac_num_iterations,
            )

            # 检查法向量方向（OpenCV坐标系Y轴向下）
            normal = np.array(plane_model[:3])
            y_component = abs(normal[1])  # Y分量的绝对值

            if y_component >= ground_normal_threshold:
                # 找到地面平面
                best_plane = np.array(plane_model)
                best_inliers = remaining_indices[inliers]
                logger.info(
                    f"Ground plane found (attempt {attempt + 1}): "
                    f"normal={normal}, Y-component={y_component:.3f}"
                )
                break
            else:
                # 不是地面，从剩余点中移除这个平面继续搜索
                logger.debug(
                    f"Plane rejected (attempt {attempt + 1}): "
                    f"normal={normal}, Y-component={y_component:.3f} < {ground_normal_threshold}"
                )
                mask = np.ones(len(remaining_pcd.points), dtype=bool)
                mask[inliers] = False
                remaining_indices = remaining_indices[mask]
                remaining_pcd = remaining_pcd.select_by_index(
                    np.where(mask)[0].tolist()
                )

        # 移除地面点
        mask = np.ones(len(pc.points), dtype=bool)
        if len(best_inliers) > 0:
            mask[best_inliers] = False
            logger.info(f"Detected ground with {len(best_inliers)} points")
        else:
            logger.warning("No valid ground plane found, keeping all points")

        return pc.filter_by_mask(mask), best_plane

    def _detect_ground_numpy(
        self, pc: PointCloud, cfg: VoxelizerConfig, ground_normal_threshold: float
    ) -> tuple[PointCloud, NDArray[np.float64]]:
        """使用numpy实现RANSAC地面检测。

        添加法向量方向验证：只接受法向量与Y轴夹角小于阈值的平面。
        """
        points = pc.points
        best_inliers = np.array([], dtype=int)
        best_plane = self._DEFAULT_GROUND_PLANE.copy()

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

            # 检查法向量方向（OpenCV坐标系Y轴向下）
            y_component = abs(normal[1])
            if y_component < ground_normal_threshold:
                continue  # 跳过非地面平面

            distances = np.abs(points @ normal + d)
            inliers = np.where(distances < cfg.ransac_distance_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = np.append(normal, d)

        mask = np.ones(len(points), dtype=bool)
        if len(best_inliers) > 0:
            mask[best_inliers] = False
            logger.info(f"Detected ground with {len(best_inliers)} points")
        else:
            logger.warning("No valid ground plane found, keeping all points")

        return pc.filter_by_mask(mask), best_plane

    def _detect_ground_gpu(
        self, pc: PointCloud, cfg: VoxelizerConfig, ground_normal_threshold: float
    ) -> tuple[PointCloud, NDArray[np.float64]]:
        """使用PyTorch GPU加速进行RANSAC地面检测。

        添加法向量方向验证：只接受法向量与Y轴夹角小于阈值的平面。
        """
        logger.info(f"Using GPU ({self._device}) for ground detection")
        device = torch.device(self._device)

        points_tensor = torch.tensor(pc.points, dtype=torch.float32, device=device)
        n_points = points_tensor.shape[0]

        best_inlier_count = 0
        best_inliers_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
        best_plane = torch.tensor(
            self._DEFAULT_GROUND_PLANE, dtype=torch.float32, device=device
        )

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

            # 添加法向量方向验证（OpenCV坐标系Y轴向下）
            normals_normalized = normals / (norms + 1e-10)
            y_components = torch.abs(normals_normalized[:, 1])
            ground_mask = y_components >= ground_normal_threshold

            # 组合有效性掩码
            valid_mask = valid_mask & ground_mask

            if not valid_mask.any():
                continue

            normals = normals / (norms + 1e-10)
            d = -(normals * p1).sum(dim=1, keepdim=True)

            # 计算所有点到每个平面的距离
            # points_tensor: (n_points, 3), normals: (batch, 3)
            distances = torch.abs(
                torch.mm(points_tensor, normals.T) + d.T
            )  # (n_points, batch)

            # 统计内点数（只考虑有效的地面平面）
            inliers_count = (distances < cfg.ransac_distance_threshold).sum(dim=0)
            inliers_count = torch.where(
                valid_mask, inliers_count, torch.zeros_like(inliers_count)
            )

            # 找最佳平面
            best_idx = inliers_count.argmax()
            if inliers_count[best_idx] > best_inlier_count:
                best_inlier_count = inliers_count[best_idx].item()
                best_inliers_mask = (
                    distances[:, best_idx] < cfg.ransac_distance_threshold
                )
                best_plane = torch.cat([normals[best_idx], d[best_idx]])

        # 转回numpy
        mask_np = ~best_inliers_mask.cpu().numpy()
        plane_np = best_plane.cpu().numpy()

        if best_inlier_count > 0:
            logger.info(f"Detected ground with {best_inlier_count} points (GPU)")
        else:
            logger.warning("No valid ground plane found (GPU), keeping all points")

        return pc.filter_by_mask(mask_np), plane_np

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
        pcd = self._to_o3d(pc)

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
