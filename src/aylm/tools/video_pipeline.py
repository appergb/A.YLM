"""视频处理流水线模块。

实现三阶段并行流水线：
1. 异步帧提取（边提取边处理）
2. SHARP推理
3. GPU体素化

流水线示意图:
    帧提取线程: [====提取帧1====][====提取帧2====][====提取帧3====]...
    处理线程:              [====推理+体素化帧1====][====推理+体素化帧2====]...
"""

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
        if hasattr(torch, "mps") and torch.mps.is_available():
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
                try:
                    self._predictor.cpu()
                except Exception:
                    pass
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
            for frame_index, timestamp in frame_indices:
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
    ) -> bool:
        """处理单帧：推理 + 体素化。"""
        import torch.nn.functional as F
        from sharp.utils import io
        from sharp.utils.gaussians import save_ply, unproject_gaussians

        frame_path = frame_info.output_path
        if frame_path is None or not frame_path.exists():
            return False

        try:
            # 加载图像
            image, _, f_px = io.load_rgb(frame_path)
            height, width = image.shape[:2]

            # 预处理
            internal_shape = (1536, 1536)
            image_pt = (
                torch.from_numpy(image.copy()).float().to(self._device).permute(2, 0, 1)
                / 255.0
            )
            disparity_factor = torch.tensor([f_px / width]).float().to(self._device)

            image_resized_pt = F.interpolate(
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

            # 体素化
            voxel_path = voxel_output_dir / f"vox_{frame_path.stem}.ply"
            self._voxelizer.process(
                ply_path,
                voxel_path,
                remove_ground=self.config.remove_ground,
                transform_coords=self.config.transform_coords,
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
        """
        self.stats = VideoProcessingStats()
        self.stats.pipeline_start_time = time.time()
        self.stats.total_videos = 1
        video_config = self.config.video_config or load_or_create_config(config_path)

        # 创建输出目录
        frames_dir = output_dir / "extracted_frames"
        ply_dir = output_dir / "gaussians"
        voxel_dir = output_dir / "voxelized"
        for d in [frames_dir, ply_dir, voxel_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if not self._load_model():
            logger.error("Failed to load model")
            self.stats.failed_videos = 1
            self.stats.pipeline_end_time = time.time()
            return self.stats

        self._load_voxelizer()
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
                success = self._process_frame(frame_info, ply_dir, voxel_dir)
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

        if self.config.verbose:
            logger.info("=" * 50)
            logger.info("Pipeline Statistics:")
            logger.info(f"  Total frames: {self.stats.total_frames_processed}")
            logger.info(f"  Total time: {self.stats.total_time:.2f}s")
            logger.info(
                f"  Avg time per frame: {self.stats.avg_processing_time_per_frame:.2f}s"
            )
            logger.info(f"  FPS: {self.stats.frames_per_second:.2f}")
            logger.info("=" * 50)

        return self.stats

    def cleanup(self):
        """清理资源。"""
        self._stop_event.set()
        self._unload_model()
        self._voxelizer = None


def process_video(
    video_path: str,
    output_dir: str,
    config_path: str | None = None,
    voxel_size: float = 0.005,
    use_gpu: bool = True,
    verbose: bool = True,
    checkpoint_path: str | None = None,
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

    Returns:
        VideoProcessingStats: 处理统计
    """
    config = VideoPipelineConfig(
        voxel_size=voxel_size,
        use_gpu=use_gpu,
        verbose=verbose,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
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
