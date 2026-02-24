"""视频处理流水线模块。

实现帧提取与处理的并行流水线：帧提取、SHARP推理、体素化、语义融合。
"""

import contextlib
import gc
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from .pointcloud_voxelizer import PointCloudVoxelizer, VoxelizerConfig
from .video_config import load_or_create_config
from .video_extractor import VideoExtractor
from .video_types import (
    FrameInfo,
    VideoConfig,
    VideoProcessingStats,
)

logger = logging.getLogger(__name__)

# 模块级常量
DEFAULT_NAV_VOXEL_SIZE = 0.05  # 导航体素大小 5cm
DEFAULT_FOV_DEGREES = 60.0  # 相机视场角
DEFAULT_DENSITY_THRESHOLD = 3  # 体素密度阈值


@dataclass
class VideoPipelineConfig:
    """视频流水线配置。"""

    video_config: VideoConfig | None = None
    voxel_size: float = 0.005
    remove_ground: bool = True
    transform_coords: bool = False
    use_gpu: bool = True
    frame_queue_size: int = 10
    checkpoint_path: Path | None = None
    device: str = "auto"
    verbose: bool = True
    auto_unload: bool = True
    # 语义检测配置
    enable_semantic: bool = True  # 是否启用语义检测
    semantic_model: str = "yolo11n-seg.pt"  # YOLO 模型
    semantic_confidence: float = 0.25  # 检测置信度阈值
    colorize_semantic: bool = True  # 是否根据语义标签着色
    # 点云切片配置
    enable_slice: bool = True  # 是否启用点云切片
    slice_radius: float = 10.0  # 切片半径（米）
    # 导航输出配置
    output_navigation_ply: bool = True  # 是否输出导航用点云（机器人坐标系）
    # 输入分辨率配置
    # 注意：SHARP 模型要求固定 1536，因为内部金字塔结构依赖 1536→768→384
    internal_resolution: int = 1536  # 内部处理分辨率（固定值，不可更改）


class VideoPipelineProcessor:
    """视频处理流水线。

    实现帧提取与处理的并行流水线。

    Example:
        >>> config = VideoPipelineConfig(voxel_size=0.005)
        >>> processor = VideoPipelineProcessor(config)
        >>> stats = processor.process(Path("video.mp4"), Path("output/"))
    """

    def __init__(self, config: VideoPipelineConfig | None = None):
        self.config = config or VideoPipelineConfig()
        self.stats = VideoProcessingStats()
        self._predictor = None
        self._device: torch.device | None = None
        self._model_loaded = False
        self._voxelizer: PointCloudVoxelizer | None = None
        self._detector = None  # YOLO 语义检测器
        self._frame_queue: queue.Queue[FrameInfo] = queue.Queue(
            maxsize=self.config.frame_queue_size
        )
        self._stop_event = threading.Event()
        self._extraction_done = threading.Event()

    def _detect_device(self) -> torch.device:
        """检测可用设备。"""
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self) -> bool:
        """加载SHARP模型。"""
        if self.config.verbose:
            logger.info("Loading SHARP model...")

        try:
            from sharp.models import PredictorParams, create_predictor

            self._device = self._detect_device()
            logger.info(f"Using device: {self._device}")

            # 加载检查点
            if self.config.checkpoint_path and self.config.checkpoint_path.exists():
                state_dict = torch.load(
                    self.config.checkpoint_path,
                    weights_only=True,
                    map_location=self._device,
                )
            else:
                model_url = (
                    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
                )
                logger.info("Downloading model from network...")
                state_dict = torch.hub.load_state_dict_from_url(
                    model_url, progress=True, map_location=self._device
                )

            self._predictor = create_predictor(PredictorParams())
            self._predictor.load_state_dict(state_dict)
            self._predictor.eval()
            self._predictor.to(self._device)

            self._model_loaded = True
            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _unload_model(self):
        """卸载模型释放内存。"""
        if not self._model_loaded:
            return
        logger.info("Unloading model...")
        if self._predictor is not None:
            if self._device and self._device.type != "cpu":
                with contextlib.suppress(Exception):
                    self._predictor.cpu()
            del self._predictor
            self._predictor = None
        self._device = None
        self._model_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")

    def _load_voxelizer(self):
        """加载体素化器。"""
        vox_config = VoxelizerConfig(
            voxel_size=self.config.voxel_size,
            use_gpu=self.config.use_gpu,
        )
        self._voxelizer = PointCloudVoxelizer(config=vox_config)
        logger.info(f"Voxelizer initialized (voxel_size: {self.config.voxel_size}m)")

    def _load_detector(self):
        """加载 YOLO 语义检测器。"""
        from aylm.tools.object_detector import DetectorConfig, ObjectDetector

        logger.info("Loading YOLO semantic detector...")
        detector_config = DetectorConfig(
            model_name=self.config.semantic_model,
            confidence_threshold=self.config.semantic_confidence,
            device=self.config.device,
        )
        self._detector = ObjectDetector(detector_config)
        self._detector.load()
        logger.info(f"Semantic detector loaded (model: {self.config.semantic_model})")

    def _cleanup_detector(self):
        """清理语义检测器。"""
        if self._detector is not None:
            logger.info("Unloading semantic detector...")
            self._detector.unload()
            del self._detector
            self._detector = None

    def _apply_slice(self, ply_path: Path, frame_stem: str) -> Path | None:
        """对点云执行半径切片，只保留摄像机附近的点。

        Args:
            ply_path: 原始 PLY 文件路径
            frame_stem: 帧文件名（不含扩展名）

        Returns:
            切片后的临时 PLY 文件路径，失败返回 None
        """
        import numpy as np

        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            logger.warning("plyfile not installed, skipping slice")
            return None

        logger.info(f"Applying slice (radius: {self.config.slice_radius}m)")

        try:
            # 读取点云
            ply_data = PlyData.read(str(ply_path))
            vertex = ply_data["vertex"]

            # 获取坐标
            x = np.array(vertex["x"], dtype=np.float64)
            z = np.array(vertex["z"], dtype=np.float64)

            # 计算水平距离（X-Z平面，以原点为圆心）
            horizontal_dist = np.sqrt(x**2 + z**2)

            # 筛选在半径内的点
            mask = horizontal_dist <= self.config.slice_radius
            kept_count = mask.sum()
            total_count = len(mask)

            if kept_count == 0:
                logger.warning("No points within slice radius, skipping slice")
                return None

            logger.info(
                f"  Slice: {kept_count}/{total_count} points "
                f"({100*kept_count/total_count:.1f}%)"
            )

            # 创建新的顶点数据
            new_vertex_data = vertex.data[mask]
            new_vertex = PlyElement.describe(new_vertex_data, "vertex")

            # 保存到临时文件
            sliced_path = ply_path.parent / f"sliced_{frame_stem}.ply"
            PlyData([new_vertex], text=False).write(str(sliced_path))

            return sliced_path

        except Exception as e:
            logger.warning(f"Failed to apply slice: {e}")
            return None

    def _apply_semantic_fusion(
        self,
        frame_path: Path,
        voxel_ply_path: Path,
        detections_dir: Path,
        navigation_dir: Path | None = None,
        focal_length: float | None = None,
    ) -> None:
        """对体素化后的点云应用语义融合。

        Args:
            frame_path: 原始帧图像路径
            voxel_ply_path: 体素化后的 PLY 文件路径
            detections_dir: 检测结果图片保存目录
            navigation_dir: 导航用点云输出目录（机器人坐标系）
            focal_length: SHARP 推理时使用的焦距（像素），如果为 None 则估算
        """
        import cv2
        import numpy as np

        from aylm.tools.semantic_fusion import FusionConfig, SemanticFusion
        from aylm.tools.semantic_types import CameraIntrinsics

        logger.debug(f"Applying semantic fusion: {frame_path.name}")

        try:
            # 读取原始图像
            image = cv2.imread(str(frame_path))
            if image is None:
                logger.warning(
                    f"Cannot read image, skipping semantic fusion: {frame_path}"
                )
                return

            height, width = image.shape[:2]

            # 执行目标检测
            detections = self._detector.detect(image, return_masks=True)
            logger.info(f"Detected {len(detections)} objects in {frame_path.name}")
            for det in detections:
                logger.debug(
                    f"  - {det.semantic_label.name} (class_id={det.class_id}, "
                    f"conf={det.confidence:.2f})"
                )

            # 保存检测结果可视化图片
            if detections:
                detection_image_path = voxel_ply_path.parent / f"det_{frame_path.name}"
                self._detector.save_detection_image(
                    image, detections, detection_image_path, draw_masks=True
                )
                logger.debug(f"Detection image saved: {detection_image_path.name}")

            if not detections:
                logger.debug("No detections, skipping semantic fusion")
                return

            # 读取体素化后的点云
            try:
                from plyfile import PlyData
            except ImportError:
                logger.warning("plyfile not installed, skipping semantic fusion")
                return

            ply_data = PlyData.read(str(voxel_ply_path))
            vertex = ply_data["vertex"]

            points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(
                np.float64
            )
            colors = None
            if "red" in vertex.data.dtype.names:
                colors = (
                    np.column_stack(
                        [vertex["red"], vertex["green"], vertex["blue"]]
                    ).astype(np.float64)
                    / 255.0
                )

            # 使用 SHARP 的精确焦距，如果没有则估算
            if focal_length is not None:
                f_px = focal_length
                logger.debug(f"Using SHARP focal length: {f_px:.1f}px")
            else:
                f_px = max(width, height)
                logger.debug(f"Estimated focal length: {f_px:.1f}px")

            intrinsics = CameraIntrinsics(fx=f_px, fy=f_px, cx=width / 2, cy=height / 2)

            # 执行语义融合
            fusion_config = FusionConfig(
                min_confidence=self.config.semantic_confidence,
                colorize_semantic=self.config.colorize_semantic,
            )
            fusion = SemanticFusion(fusion_config)
            semantic_pc = fusion.fuse(
                points=points,
                colors=colors,
                detections=detections,
                image_shape=(height, width),
                intrinsics=intrinsics,
            )

            # 保存带语义标签的 PLY（覆盖原文件）
            fusion.save_semantic_ply(
                semantic_pc,
                voxel_ply_path,
                include_semantic_colors=self.config.colorize_semantic,
            )
            logger.debug(f"Semantic fusion completed: {voxel_ply_path.name}")

            # 提取障碍物并导出 JSON（用于导航系统）
            from aylm.tools.obstacle_marker import ObstacleMarker, ObstacleMarkerConfig

            marker = ObstacleMarker(ObstacleMarkerConfig())
            obstacles = marker.extract_obstacles_from_detections(
                points=points,
                detections=detections,
                image_shape=(height, width),
                intrinsics=intrinsics,
            )

            if obstacles:
                json_path = (
                    voxel_ply_path.parent / f"{voxel_ply_path.stem}_obstacles.json"
                )
                marker.export_to_json(obstacles, json_path)
                logger.debug(f"Obstacles exported: {json_path.name}")

            # 输出导航用点云（机器人坐标系）
            if navigation_dir is not None and self.config.output_navigation_ply:
                self._save_navigation_ply(
                    semantic_pc, voxel_ply_path, navigation_dir, frame_path.stem, fusion
                )

        except Exception as e:
            logger.warning(f"Semantic fusion failed: {e}")
            logger.exception(f"Semantic fusion error - {frame_path.name}")

    def _save_navigation_ply(
        self,
        semantic_pc,
        voxel_ply_path: Path,
        navigation_dir: Path,
        frame_stem: str,
        fusion=None,
    ) -> None:
        """保存导航用点云（转换到机器人坐标系，体素化）。

        Args:
            semantic_pc: 语义点云
            voxel_ply_path: 原始体素化 PLY 路径
            navigation_dir: 导航输出目录
            frame_stem: 帧文件名（不含扩展名）
            fusion: SemanticFusion 实例（可选，避免重复创建）
        """
        try:
            if fusion is None:
                from aylm.tools.semantic_fusion import SemanticFusion

                fusion = SemanticFusion()

            nav_path = navigation_dir / f"nav_{frame_stem}.ply"
            fusion.save_navigation_ply(
                semantic_pc, nav_path, voxel_size=DEFAULT_NAV_VOXEL_SIZE
            )
            logger.debug(f"Navigation PLY saved: {nav_path.name}")

        except Exception as e:
            logger.warning(f"Failed to save navigation PLY: {e}")

    def _extraction_worker(
        self,
        video_path: Path,
        output_dir: Path,
        video_config: VideoConfig,
    ):
        """帧提取工作线程。"""
        extractor = VideoExtractor(config=video_config)

        try:
            # 获取视频元数据
            metadata = extractor.get_video_metadata(video_path)
            frame_indices = extractor._calculate_frame_indices(metadata)

            logger.info(
                f"Extracting {len(frame_indices)} frames from {video_path.name}"
            )

            # 逐帧提取并放入队列
            for frame_index, _timestamp in frame_indices:
                if self._stop_event.is_set():
                    break
                output_format = video_config.output_format.lower()
                output_path = output_dir / f"frame_{frame_index:06d}.{output_format}"
                frame_info = extractor.extract_frame(
                    video_path, frame_index, output_path
                )
                if frame_info:
                    self._frame_queue.put(frame_info)

        except Exception as e:
            logger.error(f"Extraction error: {e}")
        finally:
            self._extraction_done.set()

    @torch.no_grad()
    def _process_frame(
        self,
        frame_info: FrameInfo,
        ply_output_dir: Path,
        voxel_output_dir: Path,
        detections_dir: Path | None = None,
        navigation_dir: Path | None = None,
    ) -> bool:
        """处理单帧：推理 + 体素化 + 语义融合（可选）。

        Args:
            frame_info: 帧信息
            ply_output_dir: PLY 输出目录
            voxel_output_dir: 体素化输出目录
            detections_dir: 检测结果图片目录
            navigation_dir: 导航用点云输出目录（机器人坐标系）

        Returns:
            是否处理成功
        """
        from sharp.utils import io
        from sharp.utils.gaussians import save_ply, unproject_gaussians
        from torch.nn import functional as functional_nn

        frame_path = frame_info.output_path
        if frame_path is None or not frame_path.exists():
            return False

        try:
            # 加载图像
            image, _, f_px = io.load_rgb(frame_path)
            height, width = image.shape[:2]

            # 预处理（使用配置的内部分辨率）
            res = self.config.internal_resolution
            internal_shape = (res, res)
            image_pt = (
                torch.from_numpy(image.copy()).float().to(self._device).permute(2, 0, 1)
                / 255.0
            )
            disparity_factor = torch.tensor([f_px / width]).float().to(self._device)

            image_resized_pt = functional_nn.interpolate(
                image_pt[None],
                size=(internal_shape[1], internal_shape[0]),
                mode="bilinear",
                align_corners=True,
            )

            # 推理
            gaussians_ndc = self._predictor(image_resized_pt, disparity_factor)

            # 后处理
            intrinsics = (
                torch.tensor(
                    [
                        [f_px, 0, width / 2, 0],
                        [0, f_px, height / 2, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
                .float()
                .to(self._device)
            )

            intrinsics_resized = intrinsics.clone()
            intrinsics_resized[0] *= internal_shape[0] / width
            intrinsics_resized[1] *= internal_shape[1] / height

            gaussians = unproject_gaussians(
                gaussians_ndc,
                torch.eye(4).to(self._device),
                intrinsics_resized,
                internal_shape,
            )

            # 保存PLY
            ply_path = ply_output_dir / f"{frame_path.stem}.ply"
            save_ply(gaussians, f_px, (height, width), ply_path)

            # 切片（如果启用）
            input_ply_path = ply_path
            if self.config.enable_slice:
                sliced_path = self._apply_slice(ply_path, frame_path.stem)
                if sliced_path is not None:
                    input_ply_path = sliced_path

            # 体素化
            voxel_path = voxel_output_dir / f"vox_{frame_path.stem}.ply"
            self._voxelizer.process(
                input_ply_path,
                voxel_path,
                remove_ground=self.config.remove_ground,
                transform_coords=self.config.transform_coords,
            )

            # 清理临时切片文件
            if self.config.enable_slice and input_ply_path != ply_path:
                with contextlib.suppress(Exception):
                    input_ply_path.unlink()

            # 语义融合（如果启用）
            if self.config.enable_semantic and self._detector is not None:
                self._apply_semantic_fusion(
                    frame_path,
                    voxel_path,
                    detections_dir or voxel_output_dir,
                    navigation_dir=navigation_dir,
                    focal_length=f_px,  # 传递 SHARP 的精确焦距
                )

            return True

        except Exception as e:
            logger.error(f"Failed to process frame {frame_path}: {e}")
            return False

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        config_path: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> VideoProcessingStats:
        """处理视频。

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            config_path: 配置文件路径
            progress_callback: 进度回调 (processed, total)

        Returns:
            VideoProcessingStats: 处理统计

        输出目录结构:
            output_dir/
            ├── extracted_frames/     # 提取的视频帧
            ├── gaussians/            # SHARP 输出的 3D 高斯
            ├── voxelized/            # 体素化后的点云（带语义标签）
            │   ├── vox_frame_*.ply   # 体素化点云
            │   ├── det_frame_*.png   # 检测结果可视化
            │   └── vox_frame_*_obstacles.json  # 障碍物信息
            └── navigation/           # 导航用点云（机器人坐标系）
                └── nav_frame_*.ply
        """
        self.stats = VideoProcessingStats()
        self.stats.pipeline_start_time = time.time()
        self.stats.total_videos = 1
        video_config = self.config.video_config or load_or_create_config(config_path)

        # 创建输出目录
        frames_dir = output_dir / "extracted_frames"
        ply_dir = output_dir / "gaussians"
        voxel_dir = output_dir / "voxelized"
        detections_dir = output_dir / "detections"
        navigation_dir = (
            output_dir / "navigation" if self.config.output_navigation_ply else None
        )

        dirs_to_create = [frames_dir, ply_dir, voxel_dir, detections_dir]
        if navigation_dir:
            dirs_to_create.append(navigation_dir)
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)

        if self.config.verbose:
            logger.info(f"Output directories created in: {output_dir}")
            if self.config.enable_semantic:
                logger.info("Semantic detection: ENABLED")
            if navigation_dir:
                logger.info("Navigation PLY output: ENABLED (robot coordinate system)")

        if not self._load_model():
            logger.error("Failed to load model")
            self.stats.failed_videos = 1
            self.stats.pipeline_end_time = time.time()
            return self.stats

        self._load_voxelizer()

        # 加载语义检测器（如果启用）
        if self.config.enable_semantic:
            self._load_detector()

        self._stop_event.clear()
        self._extraction_done.clear()
        self._frame_queue = queue.Queue(maxsize=self.config.frame_queue_size)

        extraction_thread = threading.Thread(
            target=self._extraction_worker,
            args=(video_path, frames_dir, video_config),
            daemon=True,
        )
        extraction_thread.start()

        processed_count = 0
        extraction_start = time.time()

        while True:
            try:
                # 尝试从队列获取帧（超时1秒）
                frame_info = self._frame_queue.get(timeout=1.0)

                # 处理帧
                process_start = time.time()
                success = self._process_frame(
                    frame_info,
                    ply_dir,
                    voxel_dir,
                    detections_dir,
                    navigation_dir,
                )
                process_time = time.time() - process_start

                if success:
                    processed_count += 1
                    self.stats.total_frames_processed += 1
                    self.stats.total_processing_time += process_time

                    if self.config.verbose:
                        logger.info(
                            f"Processed frame {processed_count}: "
                            f"{frame_info.output_path.name} ({process_time:.2f}s)"
                        )

                    if progress_callback:
                        progress_callback(processed_count, -1)
            except queue.Empty:
                if self._extraction_done.is_set() and self._frame_queue.empty():
                    break

        extraction_thread.join(timeout=5.0)

        self.stats.total_frames_extracted = processed_count
        self.stats.total_extraction_time = time.time() - extraction_start
        self.stats.completed_videos = 1 if processed_count > 0 else 0
        self.stats.failed_videos = 0 if processed_count > 0 else 1
        self.stats.pipeline_end_time = time.time()

        if self.config.auto_unload:
            self._unload_model()
            self._cleanup_detector()

        if self.config.verbose:
            logger.info("=" * 50)
            logger.info("Pipeline Statistics:")
            logger.info(f"  Total frames: {self.stats.total_frames_processed}")
            logger.info(f"  Total time: {self.stats.total_time:.2f}s")
            logger.info(
                f"  Avg time per frame: {self.stats.avg_processing_time_per_frame:.2f}s"
            )
            logger.info(f"  FPS: {self.stats.frames_per_second:.2f}")
            if self.config.enable_semantic:
                logger.info("  Semantic detection: ENABLED")
            if navigation_dir:
                logger.info(f"  Navigation PLY: {navigation_dir}")
            logger.info("=" * 50)

        return self.stats

    def cleanup(self):
        """清理资源。"""
        self._stop_event.set()
        self._unload_model()
        self._cleanup_detector()
        self._voxelizer = None


def process_video(
    video_path: str,
    output_dir: str,
    config_path: str | None = None,
    voxel_size: float = 0.005,
    use_gpu: bool = True,
    verbose: bool = True,
    checkpoint_path: str | None = None,
    enable_semantic: bool = True,
    semantic_model: str = "yolo11n-seg.pt",
    semantic_confidence: float = 0.5,
    output_navigation_ply: bool = True,
) -> VideoProcessingStats:
    """便捷函数：处理视频。

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        config_path: 配置文件路径
        voxel_size: 体素尺寸
        use_gpu: 是否使用GPU
        verbose: 详细输出
        checkpoint_path: 模型检查点路径
        enable_semantic: 是否启用语义检测
        semantic_model: YOLO 模型名称
        semantic_confidence: 检测置信度阈值
        output_navigation_ply: 是否输出导航用点云（机器人坐标系）

    Returns:
        VideoProcessingStats: 处理统计

    Example:
        >>> from aylm.tools.video_pipeline import process_video
        >>> stats = process_video(
        ...     video_path="video.mp4",
        ...     output_dir="output/",
        ...     enable_semantic=True,
        ...     output_navigation_ply=True,
        ... )
        >>> print(f"Processed {stats.total_frames_processed} frames")
    """
    config = VideoPipelineConfig(
        voxel_size=voxel_size,
        use_gpu=use_gpu,
        verbose=verbose,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        enable_semantic=enable_semantic,
        semantic_model=semantic_model,
        semantic_confidence=semantic_confidence,
        output_navigation_ply=output_navigation_ply,
    )

    processor = VideoPipelineProcessor(config)
    try:
        return processor.process(
            Path(video_path),
            Path(output_dir),
            Path(config_path) if config_path else None,
        )
    finally:
        processor.cleanup()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python video_pipeline.py <video_path> <output_dir>")
        sys.exit(1)

    stats = process_video(sys.argv[1], sys.argv[2])
    sys.exit(0 if stats.failed_videos == 0 else 1)
