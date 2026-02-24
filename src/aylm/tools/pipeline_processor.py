"""流水线处理器模块。

实现多图像流水线处理，支持推理与体素化并行、语义检测融合。
"""

from __future__ import annotations

import contextlib
import gc
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import torch
from torch.nn import functional as functional_nn

# 模块级常量
DEFAULT_VOXEL_SIZE = 0.05  # 5cm 体素
DEFAULT_SLICE_RADIUS = 20.0  # 20m 切片半径
DEFAULT_FOV_DEGREES = 60.0  # 相机视场角

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举。"""

    PENDING = "pending"  # 等待中
    PREDICTING = "predicting"  # 模型推理中
    PREDICTED = "predicted"  # 推理完成
    VOXELIZING = "voxelizing"  # 体素化中
    COMPLETED = "completed"  # 全部完成
    FAILED = "failed"  # 失败


@dataclass
class ImageTask:
    """单张图像的处理任务。"""

    image_path: Path
    index: int
    status: TaskStatus = TaskStatus.PENDING
    ply_output_path: Path | None = None
    voxel_output_path: Path | None = None
    predict_start_time: float | None = None
    predict_end_time: float | None = None
    voxel_start_time: float | None = None
    voxel_end_time: float | None = None
    error_message: str | None = None
    # 语义检测相关字段
    detections: list | None = None  # 检测结果
    semantic_ply_path: Path | None = None  # 语义 PLY 路径
    # 相机参数（用于语义融合）
    focal_length: float | None = None  # 焦距（像素）
    image_width: int | None = None  # 图像宽度
    image_height: int | None = None  # 图像高度


@dataclass
class PipelineConfig:
    """流水线配置。"""

    voxel_size: float = 0.005
    remove_ground: bool = True
    transform_coords: bool = False
    device: str = "auto"
    verbose: bool = True
    checkpoint_path: Path | None = None
    auto_unload: bool = True
    async_mode: bool = False
    # 切片配置
    enable_slice: bool = True  # 是否启用切片
    slice_radius: float = 20.0  # 切片半径（米）
    # 语义检测配置
    enable_semantic: bool = True  # 是否启用语义检测（默认开启）
    semantic_model: str = "yolo11n-seg.pt"  # YOLO 模型
    semantic_confidence: float = 0.5  # 检测置信度
    colorize_semantic: bool = True  # 语义着色
    # 导航输出配置
    output_navigation_ply: bool = True  # 是否输出导航用点云（机器人坐标系）


@dataclass
class PipelineStats:
    """流水线统计信息。"""

    total_images: int = 0
    completed_images: int = 0
    failed_images: int = 0
    total_predict_time: float = 0.0
    total_voxel_time: float = 0.0
    pipeline_start_time: float | None = None
    pipeline_end_time: float | None = None

    @property
    def total_time(self) -> float:
        if self.pipeline_start_time and self.pipeline_end_time:
            return self.pipeline_end_time - self.pipeline_start_time
        return 0.0

    @property
    def avg_predict_time(self) -> float:
        return (
            self.total_predict_time / self.completed_images
            if self.completed_images
            else 0.0
        )

    @property
    def avg_voxel_time(self) -> float:
        return (
            self.total_voxel_time / self.completed_images
            if self.completed_images
            else 0.0
        )


class PipelineLogger:
    """流水线日志记录器。"""

    LEVEL_PREFIX = {
        "INFO": "   ",
        "STAGE": ">>>",
        "OK": " ✓ ",
        "WARN": " ! ",
        "ERROR": " ✗ ",
        "PROGRESS": " → ",
    }

    STATUS_DISPLAY = {
        TaskStatus.PENDING: "⏳ 等待中",
        TaskStatus.PREDICTING: "🔄 推理中",
        TaskStatus.PREDICTED: "📦 待体素化",
        TaskStatus.VOXELIZING: "🔄 体素化中",
        TaskStatus.COMPLETED: "✅ 完成",
        TaskStatus.FAILED: "❌ 失败",
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._lock = threading.Lock()
        self._start_time = time.time()

    def _timestamp(self) -> str:
        return f"[{time.time() - self._start_time:8.2f}s]"

    def _print(self, msg: str, level: str = "INFO"):
        with self._lock:
            prefix = self.LEVEL_PREFIX.get(level, "   ")
            print(f"{self._timestamp()} {prefix} {msg}")

    def header(self, title: str):
        with self._lock:
            print("\n" + "=" * 60)
            print(f"  {title}")
            print("=" * 60)

    def section(self, title: str):
        with self._lock:
            print(f"\n{'─' * 40}\n  {title}\n{'─' * 40}")

    def stage(self, msg: str):
        self._print(msg, "STAGE")

    def info(self, msg: str):
        if self.verbose:
            self._print(msg, "INFO")

    def ok(self, msg: str):
        self._print(msg, "OK")

    def warn(self, msg: str):
        self._print(msg, "WARN")

    def error(self, msg: str):
        self._print(msg, "ERROR")

    def progress(self, msg: str):
        self._print(msg, "PROGRESS")

    def task_status(self, tasks: list[ImageTask]):
        with self._lock:
            print("\n┌─────┬────────────────────────────┬─────────────┐")
            print("│ No. │ 文件名                     │ 状态        │")
            print("├─────┼────────────────────────────┼─────────────┤")
            for task in tasks:
                name = task.image_path.name[:24]
                status = self.STATUS_DISPLAY.get(task.status, "未知")
                print(f"│ {task.index + 1:3d} │ {name:<26} │ {status:<11} │")
            print("└─────┴────────────────────────────┴────────────��┘")

    def stats(self, stats: PipelineStats):
        with self._lock:
            print("\n" + "=" * 60)
            print("  流水线执行统计")
            print("=" * 60)
            print(f"  总图像数:       {stats.total_images}")
            print(f"  成功完成:       {stats.completed_images}")
            print(f"  失败数量:       {stats.failed_images}")
            print("  ─────────────────────────────────")
            print(f"  总耗时:         {stats.total_time:.2f} 秒")
            print(f"  推理总耗时:     {stats.total_predict_time:.2f} 秒")
            print(f"  体素化总耗时:   {stats.total_voxel_time:.2f} 秒")
            print("  ─────────────────────────────────")
            print(f"  平均推理时间:   {stats.avg_predict_time:.2f} 秒/张")
            print(f"  平均体素化时间: {stats.avg_voxel_time:.2f} 秒/张")
            if stats.total_images > 1 and stats.total_time > 0:
                sequential = (
                    stats.avg_predict_time + stats.avg_voxel_time
                ) * stats.completed_images
                print(f"  流水线效率:     {sequential / stats.total_time:.1%}")
            print("=" * 60 + "\n")


class PipelineProcessor:
    """流水线处理器，实现模型推理和体素化的并行处理。"""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.log = PipelineLogger(self.config.verbose)
        self.stats = PipelineStats()

        self._predictor = None
        self._device: torch.device | None = None
        self._model_loaded = False
        self._voxelizer = None
        self._detector = None  # YOLO 语义检测器
        self._tasks: list[ImageTask] = []
        self._stop_event = threading.Event()
        self._predict_lock = threading.Lock()
        self._async_executor: ThreadPoolExecutor | None = None
        self._async_future: Future | None = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        self._unload_model()
        self._cleanup_voxelizer()
        self._cleanup_detector()
        self._cleanup_async()

    def _unload_model(self):
        if not self._model_loaded:
            return

        self.log.stage("卸载模型，释放内存...")

        try:
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
                torch.cuda.synchronize()

            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                with contextlib.suppress(Exception):
                    torch.mps.empty_cache()

            self.log.ok("模型已卸载，内存已释放")
        except Exception as e:
            self.log.warn(f"模型卸载时出现警告: {e}")

    def _cleanup_voxelizer(self):
        if self._voxelizer is not None:
            del self._voxelizer
            self._voxelizer = None

    def _load_detector(self):
        """加载 YOLO 语义检测器。"""
        from aylm.tools.object_detector import DetectorConfig, ObjectDetector

        self.log.stage("加载 YOLO 语义检测器...")
        detector_config = DetectorConfig(
            model_name=self.config.semantic_model,
            confidence_threshold=self.config.semantic_confidence,
            device=self.config.device,
        )
        self._detector = ObjectDetector(detector_config)
        self._detector.load()
        self.log.ok(f"语义检测器已加载 (模型: {self.config.semantic_model})")

    def _cleanup_detector(self):
        """清理语义检测器。"""
        if self._detector is not None:
            self.log.info("卸载语义检测器...")
            self._detector.unload()
            del self._detector
            self._detector = None

    def _cleanup_async(self):
        if self._async_executor is not None:
            self._async_executor.shutdown(wait=False)
            self._async_executor = None
        self._async_future = None

    def _detect_device(self) -> torch.device:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "mps") and torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self) -> bool:
        self.log.stage("加载SHARP模型到内存...")

        try:
            from sharp.models import PredictorParams, create_predictor

            self._device = self._detect_device()
            self.log.info(f"使用设备: {self._device}")

            if self.config.checkpoint_path and self.config.checkpoint_path.exists():
                self.log.info(f"从本地加载: {self.config.checkpoint_path}")
                state_dict = torch.load(
                    self.config.checkpoint_path,
                    weights_only=True,
                    map_location=self._device,
                )
            else:
                model_url = (
                    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
                )
                self.log.info("从网络下载模型...")
                state_dict = torch.hub.load_state_dict_from_url(
                    model_url, progress=True, map_location=self._device
                )

            self._predictor = create_predictor(PredictorParams())
            self._predictor.load_state_dict(state_dict)
            self._predictor.eval()
            self._predictor.to(self._device)

            self._model_loaded = True
            self.log.ok("模型加载完成")
            return True

        except ImportError as e:
            self.log.error("模型加载失败: 缺少依赖模块")
            self.log.error("    请确保已安装 sharp 包: pip install sharp")
            self.log.error(f"    详细信息: {e}")
            logger.exception("模型导入失败")
            return False
        except Exception as e:
            self.log.error(f"模型加载失败: {e}")
            self.log.error(f"    错误类型: {type(e).__name__}")
            logger.exception("模型加载异常")
            return False

    def _load_voxelizer(self):
        from aylm.tools.pointcloud_voxelizer import PointCloudVoxelizer, VoxelizerConfig

        self._voxelizer = PointCloudVoxelizer(
            config=VoxelizerConfig(voxel_size=self.config.voxel_size)
        )
        self.log.info(f"体素化器已初始化 (体素尺寸: {self.config.voxel_size}m)")

    @torch.no_grad()
    def _predict_single(self, task: ImageTask, output_dir: Path) -> bool:
        """对单张图像进行模型推理。"""
        from sharp.utils import io
        from sharp.utils.gaussians import save_ply, unproject_gaussians

        task.status = TaskStatus.PREDICTING
        task.predict_start_time = time.time()

        self.log.progress(f"[{task.index+1}] 开始推理: {task.image_path.name}")

        try:
            # 加载图像
            image, _, f_px = io.load_rgb(task.image_path)
            height, width = image.shape[:2]

            self.log.info(f"    图像尺寸: {width}x{height}, 焦距: {f_px:.1f}px")

            # 预处理
            internal_shape = (1536, 1536)
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

            # 推理（需要锁保护，因为模型不是线程安全的）
            with self._predict_lock:
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
            output_path = output_dir / f"{task.image_path.stem}.ply"
            save_ply(gaussians, f_px, (height, width), output_path)

            task.ply_output_path = output_path
            task.status = TaskStatus.PREDICTED
            task.predict_end_time = time.time()
            # 保存相机参数供语义融合使用
            task.focal_length = f_px
            task.image_width = width
            task.image_height = height

            predict_time = task.predict_end_time - task.predict_start_time
            self.log.ok(
                f"[{task.index+1}] 推理完成: {task.image_path.name} ({predict_time:.2f}s)"
            )
            self.log.info(f"    输出: {output_path.name}")

            return True

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.predict_end_time = time.time()
            self.log.error(f"[{task.index+1}] 推理失败: {task.image_path.name}")
            self.log.error(f"    错误类型: {type(e).__name__}")
            self.log.error(f"    错误信息: {e}")
            logger.exception(f"推理异常详情 - {task.image_path.name}")
            return False

    def _voxelize_single(
        self, task: ImageTask, output_dir: Path, navigation_dir: Path | None = None
    ) -> bool:
        """对单个PLY文件进行切片、体素化，可选语义融合。

        处理流程：SHARP输出 → 切片 → 体素化 → 语义融合

        Args:
            task: 图像任务
            output_dir: 体素化输出目录
            navigation_dir: 导航用点云输出目录（机器人坐标系）
        """
        if task.status != TaskStatus.PREDICTED or task.ply_output_path is None:
            return False

        task.status = TaskStatus.VOXELIZING
        task.voxel_start_time = time.time()

        self.log.progress(f"[{task.index+1}] 开始处理: {task.ply_output_path.name}")

        try:
            # 确定输入文件（可能经过切片）
            input_ply_path = task.ply_output_path

            # 步骤1：切片（如果启用）
            if self.config.enable_slice:
                sliced_path = self._apply_slice(task)
                if sliced_path is not None:
                    input_ply_path = sliced_path

            # 步骤2：体素化
            output_path = output_dir / f"vox_{task.ply_output_path.name}"
            self._voxelizer.process(
                input_ply_path,
                output_path,
                remove_ground=self.config.remove_ground,
                transform_coords=self.config.transform_coords,
            )

            # 清理临时切片文件
            if self.config.enable_slice and input_ply_path != task.ply_output_path:
                with contextlib.suppress(Exception):
                    input_ply_path.unlink()

            # 步骤3：语义融合（如果启用）
            if self.config.enable_semantic and self._detector is not None:
                self._apply_semantic_fusion(task, output_path, navigation_dir)

            task.voxel_output_path = output_path
            task.status = TaskStatus.COMPLETED
            task.voxel_end_time = time.time()

            voxel_time = task.voxel_end_time - task.voxel_start_time
            self.log.ok(
                f"[{task.index+1}] 处理完成: {task.ply_output_path.name} ({voxel_time:.2f}s)"
            )
            self.log.info(f"    输出: {output_path.name}")

            return True

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.voxel_end_time = time.time()
            self.log.error(f"[{task.index+1}] 处理失败: {task.ply_output_path.name}")
            self.log.error(f"    错误类型: {type(e).__name__}")
            self.log.error(f"    错误信息: {e}")
            logger.exception(f"处理异常详情 - {task.ply_output_path.name}")
            return False

    def _apply_slice(self, task: ImageTask) -> Path | None:
        """对点云执行半径切片，只保留摄像机附近的点。

        Args:
            task: 图像任务

        Returns:
            切片后的临时 PLY 文件路径，失败返回 None
        """
        import numpy as np

        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            self.log.warn("    plyfile 未安装，跳过切片")
            return None

        self.log.info(f"    执行切片 (半径: {self.config.slice_radius}m)")

        try:
            # 读取点云
            ply_data = PlyData.read(str(task.ply_output_path))
            vertex = ply_data["vertex"]

            # 获取坐标
            x = np.array(vertex["x"], dtype=np.float64)
            z = np.array(vertex["z"], dtype=np.float64)

            # 计算水平距离（X-Z平面，以原点为圆心）
            # OpenCV 坐标系: X右, Y下, Z前
            # 水平面是 X-Z 平面
            distance_xz = np.sqrt(x**2 + z**2)

            # 筛选在半径内的点
            mask = distance_xz <= self.config.slice_radius
            n_original = len(x)
            n_kept = mask.sum()

            self.log.info(
                f"    切片结果: {n_kept}/{n_original} 点 "
                f"({n_kept / n_original * 100:.1f}%)"
            )

            if n_kept == 0:
                self.log.warn("    切片后无点，跳过")
                return None

            if n_kept == n_original:
                self.log.info("    所有点都在半径内，跳过切片")
                return None

            # 构建新的顶点数据
            new_vertex_data = vertex.data[mask]

            # 保存到临时文件
            sliced_path = (
                task.ply_output_path.parent / f"sliced_{task.ply_output_path.name}"
            )
            new_vertex = PlyElement.describe(new_vertex_data, "vertex")
            PlyData([new_vertex], text=False).write(str(sliced_path))

            self.log.ok(f"    切片完成: {sliced_path.name}")
            return sliced_path

        except Exception as e:
            self.log.warn(f"    切片失败: {e}")
            logger.exception(f"切片异常 - {task.ply_output_path.name}")
            return None

    def _apply_semantic_fusion(
        self, task: ImageTask, voxel_ply_path: Path, navigation_dir: Path | None = None
    ) -> None:
        """对体素化后的点云应用语义融合。

        处理流程：
        1. 语义融合 -> 保存 vox_xxx.ply (OpenCV 坐标系，用于可视化)
        2. 坐标转换 -> 保存 nav_xxx.ply (机器人坐标系，用于导航)
        3. 障碍物提取 -> 保存 xxx_obstacles.json (机器人坐标系)

        Args:
            task: 图像任务
            voxel_ply_path: 体素化后的 PLY 文件路径
            navigation_dir: 导航用点云输出目录（机器人坐标系）
        """
        import cv2

        from aylm.tools.semantic_fusion import FusionConfig, SemanticFusion
        from aylm.tools.semantic_types import CameraIntrinsics

        self.log.info(f"    执行语义融合: {task.image_path.name}")

        try:
            # 读取原始图像
            image = cv2.imread(str(task.image_path))
            if image is None:
                self.log.warn(f"    无法读取图像，跳过语义融合: {task.image_path}")
                return

            height, width = image.shape[:2]

            # 执行目标检测
            detections = self._detector.detect(image, return_masks=True)
            self.log.info(f"    检测到 {len(detections)} 个目标")

            # 保存检测结果可视化图片（放在 PLY 输出目录）
            if detections:
                detection_image_path = (
                    voxel_ply_path.parent / f"det_{task.image_path.name}"
                )
                self._detector.save_detection_image(
                    image, detections, detection_image_path, draw_masks=True
                )
                self.log.ok(f"    检测结果图片已保存: {detection_image_path.name}")

            if not detections:
                self.log.info("    无检测结果，跳过语义融合")
                return

            # 读取体素化后的点云
            try:
                from plyfile import PlyData
            except ImportError:
                self.log.warn("    plyfile 未安装，跳过语义融合")
                return

            ply_data = PlyData.read(str(voxel_ply_path))
            vertex = ply_data["vertex"]

            import numpy as np

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

            # 使用 SHARP 推理时保存的相机内参（精确焦距）
            if task.focal_length is not None:
                f_px = task.focal_length
                self.log.info(f"    使用 SHARP 焦距: {f_px:.1f}px")
            else:
                # 回退：从图像重新读取焦距
                from sharp.utils import io

                _, _, f_px = io.load_rgb(task.image_path)
                self.log.info(f"    从 EXIF 读取焦距: {f_px:.1f}px")

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

            # 步骤1: 保存带语义标签的 PLY（OpenCV 坐标系，用于可视化）
            fusion.save_semantic_ply(
                semantic_pc,
                voxel_ply_path,
                include_semantic_colors=self.config.colorize_semantic,
            )
            self.log.ok(f"    语义融合完成 (OpenCV坐标系): {voxel_ply_path.name}")

            # 步骤2: 保存导航用点云（机器人坐标系，体素化）
            if navigation_dir is not None and self.config.output_navigation_ply:
                nav_ply_path = navigation_dir / f"nav_{voxel_ply_path.stem[4:]}.ply"
                fusion.save_navigation_ply(
                    semantic_pc, nav_ply_path, voxel_size=DEFAULT_VOXEL_SIZE
                )
                self.log.ok(
                    f"    导航点云已保存 (机器人坐标系, {DEFAULT_VOXEL_SIZE*100:.0f}cm体素): {nav_ply_path.name}"
                )

            # 步骤3: 提取障碍物并导出 JSON（机器人坐标系）
            from aylm.tools.obstacle_marker import ObstacleMarker, ObstacleMarkerConfig

            marker_intrinsics = CameraIntrinsics(
                fx=f_px, fy=f_px, cx=width / 2, cy=height / 2
            )
            marker = ObstacleMarker(ObstacleMarkerConfig())
            obstacles = marker.extract_obstacles_from_detections(
                points=points,
                detections=detections,
                image_shape=(height, width),
                intrinsics=marker_intrinsics,
            )

            if obstacles:
                # 将障碍物坐标转换为机器人坐标系
                obstacles_robot = self._transform_obstacles_to_robot(obstacles)
                json_path = (
                    voxel_ply_path.parent / f"{voxel_ply_path.stem}_obstacles.json"
                )
                marker.export_to_json(obstacles_robot, json_path)
                self.log.ok(f"    障碍物信息已导出 (机器人坐标系): {json_path.name}")

        except Exception as e:
            self.log.warn(f"    语义融合失败: {e}")
            logger.exception(f"语义融合异常 - {task.image_path.name}")

    def _transform_obstacles_to_robot(self, obstacles: list) -> list:
        """将障碍物边界框坐标从 OpenCV 坐标系转换到机器人坐标系。

        Args:
            obstacles: OpenCV 坐标系下的障碍物列表

        Returns:
            机器人坐标系下的障碍物列表
        """
        import numpy as np

        from aylm.tools.coordinate_utils import transform_opencv_to_robot
        from aylm.tools.obstacle_marker import ObstacleBox3D

        transformed = []
        for obs in obstacles:
            # 转换中心点
            center_opencv = np.array(obs.center)
            center_robot = transform_opencv_to_robot(center_opencv)

            # 转换尺寸（需要重新映射维度）
            # OpenCV: (width_x, height_y, depth_z) -> Robot: (depth_z, width_x, height_y)
            # 因为 Robot X = OpenCV Z, Robot Y = -OpenCV X, Robot Z = -OpenCV Y
            dims_opencv = obs.dimensions
            dims_robot = (dims_opencv[2], dims_opencv[0], dims_opencv[1])

            transformed_obs = ObstacleBox3D(
                center=tuple(center_robot),
                dimensions=dims_robot,
                label=obs.label,
                confidence=obs.confidence,
                point_indices=obs.point_indices,
            )
            transformed.append(transformed_obs)

        return transformed

    def _collect_images(self, input_path: Path) -> list[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".bmp"}

        if input_path.is_file():
            return [input_path] if input_path.suffix.lower() in extensions else []

        images = []
        for ext in extensions:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(images)

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        voxel_output_dir: Path | None = None,
    ) -> PipelineStats:
        """执行流水线处理。

        Args:
            input_path: 输入图像路径或目录
            output_dir: PLY输出目录
            voxel_output_dir: 体素化输出目录（默认为output_dir/voxelized）

        Returns:
            PipelineStats: 处理统计信息
        """
        self.log.header("A.YLM 流水线处理器 v2.0")

        # 初始化
        self.stats = PipelineStats()
        self.stats.pipeline_start_time = time.time()

        # 验证输入路径
        if not input_path.exists():
            self.log.error(f"输入路径不存在: {input_path}")
            self.stats.pipeline_end_time = time.time()
            return self.stats

        if voxel_output_dir is None:
            voxel_output_dir = output_dir / "voxelized"

        # 创建导航输出目录（如果启用）
        navigation_dir = (
            output_dir / "navigation"
            if self.config.output_navigation_ply and self.config.enable_semantic
            else None
        )

        # 创建输出目录
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            voxel_output_dir.mkdir(parents=True, exist_ok=True)
            if navigation_dir is not None:
                navigation_dir.mkdir(parents=True, exist_ok=True)
            self.log.info(f"PLY输出目录: {output_dir}")
            self.log.info(f"体素化输出目录: {voxel_output_dir}")
            if navigation_dir is not None:
                self.log.info(f"导航输出目录: {navigation_dir}")
        except PermissionError as e:
            self.log.error(f"无法创建输出目录: {e}")
            self.stats.pipeline_end_time = time.time()
            return self.stats

        # 收集图像
        self.log.section("阶段 1: 收集图像")
        image_paths = self._collect_images(input_path)

        if not image_paths:
            self.log.error(f"未找到图像文件: {input_path}")
            return self.stats

        self.stats.total_images = len(image_paths)
        self.log.ok(f"找到 {len(image_paths)} 张图像")

        for i, path in enumerate(image_paths):
            self.log.info(f"  [{i+1}] {path.name}")

        # 创建任务
        self._tasks = [
            ImageTask(image_path=path, index=i) for i, path in enumerate(image_paths)
        ]

        # 加载模型
        self.log.section("阶段 2: 加载模型")
        if not self._load_model():
            self.log.error("模型加载失败，终止处理")
            return self.stats

        # 加载体素化器
        self._load_voxelizer()

        # 加载语义检测器（如果启用）
        if self.config.enable_semantic:
            self._load_detector()

        # 执行流水线
        self.log.section("阶段 3: 流水线处理")
        self.log.info("流水线模式: 推理(N) || 体素化(N-1)")
        if self.config.enable_semantic:
            self.log.info("语义检测: 已启用")
        if navigation_dir is not None:
            self.log.info("导航输出: 已启用 (机器人坐标系)")
        self.log.info("")

        self._execute_pipeline(output_dir, voxel_output_dir, navigation_dir)

        # 统计结果
        self.stats.pipeline_end_time = time.time()

        for task in self._tasks:
            if task.status == TaskStatus.COMPLETED:
                self.stats.completed_images += 1
                if task.predict_start_time and task.predict_end_time:
                    self.stats.total_predict_time += (
                        task.predict_end_time - task.predict_start_time
                    )
                if task.voxel_start_time and task.voxel_end_time:
                    self.stats.total_voxel_time += (
                        task.voxel_end_time - task.voxel_start_time
                    )
            elif task.status == TaskStatus.FAILED:
                self.stats.failed_images += 1

        # 打印最终状态
        self.log.section("处理结果")
        self.log.task_status(self._tasks)
        self.log.stats(self.stats)

        # 自动卸载模型
        if self.config.auto_unload:
            self.log.section("阶段 4: 清理资源")
            self._unload_model()
            self._cleanup_voxelizer()
            self._cleanup_detector()

        return self.stats

    def process_async(
        self,
        input_path: Path,
        output_dir: Path,
        voxel_output_dir: Path | None = None,
        callback: Callable[[PipelineStats], None] | None = None,
    ) -> Future:
        """异步执行流水线处理。

        在后台线程中执行处理，立即返回 Future 对象。

        Args:
            input_path: 输入图像路径或目录
            output_dir: PLY输出目录
            voxel_output_dir: 体素化输出目录
            callback: 处理完成后的回调函数

        Returns:
            Future: 可用于获取结果或检查状态

        Example:
            >>> processor = PipelineProcessor(config)
            >>> future = processor.process_async(input_path, output_dir)
            >>> # 做其他事情...
            >>> if future.done():
            ...     stats = future.result()
            >>> # 或者等待完成
            >>> stats = future.result(timeout=300)
        """
        if self._async_executor is None:
            self._async_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="pipeline"
            )

        def _run():
            try:
                stats = self.process(input_path, output_dir, voxel_output_dir)
                if callback:
                    callback(stats)
                return stats
            except Exception:
                logger.exception("异步处理失败")
                raise

        self._async_future = self._async_executor.submit(_run)
        return self._async_future

    def is_processing(self) -> bool:
        """检查是否正在处理中。"""
        if self._async_future is None:
            return False
        return not self._async_future.done()

    def wait_for_completion(self, timeout: float | None = None) -> PipelineStats | None:
        """等待异步处理完成。

        Args:
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            PipelineStats 或 None（如果超时）
        """
        if self._async_future is None:
            return None
        try:
            return self._async_future.result(timeout=timeout)
        except TimeoutError:
            return None

    def cancel(self) -> bool:
        """取消正在进行的处理。"""
        self._stop_event.set()
        if self._async_future is not None:
            return self._async_future.cancel()
        return False

    def _execute_pipeline(
        self,
        output_dir: Path,
        voxel_output_dir: Path,
        navigation_dir: Path | None = None,
    ):
        """执行流水线处理逻辑。

        流水线策略:
        1. 第一张图片: 只做推理（无并行）
        2. 第2到N张图片: 推理第N张 || 体素化第N-1张（并行）
        3. 最后: 体素化最后一张图片（无并行）

        时间线示意:
            图片1: [====推理====]
            图片2:              [====推理====]
            图片1:              [====体素化====]
            图片3:                            [====推理====]
            图片2:                            [====体素化====]
            ...

        Args:
            output_dir: PLY 输出目录
            voxel_output_dir: 体素化输出目录
            navigation_dir: 导航用点云输出目录（机器人坐标系）
        """
        total = len(self._tasks)

        if total == 0:
            self.log.warn("没有任务需要处理")
            return

        # 使用线程池执行体素化（推理在主线程，因为GPU操作需要同步）
        with ThreadPoolExecutor(max_workers=1) as voxel_executor:
            voxel_future: Future | None = None
            prev_task_for_voxel: ImageTask | None = None

            for i, task in enumerate(self._tasks):
                self.log.info(f"\n{'─' * 40}")
                self.log.info(f"处理进度: {i+1}/{total}")

                # 显示当前阶段的并行状态
                if i == 0:
                    self.log.info("  阶段: 推理第1张（无并行）")
                elif i < total:
                    self.log.info(f"  阶段: 推理第{i+1}张 || 体素化第{i}张（并行）")

                # 如果有上一张图片需要体素化，启动异步体素化
                if prev_task_for_voxel is not None:
                    self.log.progress(
                        f"  启动并行体素化: [{prev_task_for_voxel.index+1}] {prev_task_for_voxel.image_path.name}"
                    )
                    voxel_future = voxel_executor.submit(
                        self._voxelize_single,
                        prev_task_for_voxel,
                        voxel_output_dir,
                        navigation_dir,
                    )

                # 执行当前图片的推理（主线程）
                predict_success = self._predict_single(task, output_dir)

                # 等待并行的体素化完成（如果有）
                if voxel_future is not None:
                    try:
                        voxel_future.result()
                    except Exception as e:
                        self.log.error(f"体素化任务异常: {e}")
                    voxel_future = None

                # 记录当前任务用于下一轮的体素化
                prev_task_for_voxel = task if predict_success else None

            # 处理最后一张图片的体素化（同步执行，无并行）
            if prev_task_for_voxel is not None:
                self.log.info(f"\n{'─' * 40}")
                self.log.info("最终阶段: 体素化最后一张图片")
                self._voxelize_single(
                    prev_task_for_voxel, voxel_output_dir, navigation_dir
                )


def run_pipeline(
    input_path: str,
    output_dir: str,
    voxel_size: float = 0.005,
    checkpoint_path: str | None = None,
    verbose: bool = True,
    auto_unload: bool = True,
    enable_slice: bool = True,
    slice_radius: float = 10.0,
    enable_semantic: bool = False,
    semantic_model: str = "yolo11n-seg.pt",
    semantic_confidence: float = 0.5,
    colorize_semantic: bool = True,
) -> PipelineStats:
    """便捷函数：运行流水线处理。

    Args:
        input_path: 输入图像路径或目录
        output_dir: 输出目录
        voxel_size: 体素尺寸（米）
        checkpoint_path: 模型检查点路径
        verbose: 是否详细输出
        auto_unload: 处理完成后自动卸载模型（默认True）
        enable_slice: 是否启用切片（默认True）
        slice_radius: 切片半径（米，默认10.0）
        enable_semantic: 是否启用语义检测
        semantic_model: YOLO 模型名称
        semantic_confidence: 检测置信度阈值
        colorize_semantic: 是否根据语义标签着色

    Returns:
        PipelineStats: 处理统计信息

    Example:
        >>> from aylm.tools.pipeline_processor import run_pipeline
        >>> stats = run_pipeline(
        ...     input_path="inputs/input_images",
        ...     output_dir="outputs/output_gaussians",
        ...     voxel_size=0.005,
        ...     slice_radius=10.0,
        ...     enable_semantic=True
        ... )
        >>> print(f"处理完成: {stats.completed_images}/{stats.total_images}")
    """
    config = PipelineConfig(
        voxel_size=voxel_size,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        verbose=verbose,
        auto_unload=auto_unload,
        enable_slice=enable_slice,
        slice_radius=slice_radius,
        enable_semantic=enable_semantic,
        semantic_model=semantic_model,
        semantic_confidence=semantic_confidence,
        colorize_semantic=colorize_semantic,
    )

    # 使用上下文管理器确保资源释放
    with PipelineProcessor(config) as processor:
        return processor.process(Path(input_path), Path(output_dir))


def run_pipeline_async(
    input_path: str,
    output_dir: str,
    voxel_size: float = 0.005,
    checkpoint_path: str | None = None,
    verbose: bool = True,
    callback: Callable[[PipelineStats], None] | None = None,
    enable_slice: bool = True,
    slice_radius: float = 10.0,
    enable_semantic: bool = False,
    semantic_model: str = "yolo11n-seg.pt",
    semantic_confidence: float = 0.5,
    colorize_semantic: bool = True,
) -> tuple[PipelineProcessor, Future]:
    """便捷函数：异步运行流水线处理。

    Args:
        input_path: 输入图像路径或目录
        output_dir: 输出目录
        voxel_size: 体素尺寸（米）
        checkpoint_path: 模型检查点路径
        verbose: 是否详细输出
        callback: 处理完成后的回调函数
        enable_slice: 是否启用切片（默认True）
        slice_radius: 切片半径（米，默认10.0）
        enable_semantic: 是否启用语义检测
        semantic_model: YOLO 模型名称
        semantic_confidence: 检测置信度阈值
        colorize_semantic: 是否根据语义标签着色

    Returns:
        Tuple[PipelineProcessor, Future]: 处理器实例和Future对象

    Example:
        >>> from aylm.tools.pipeline_processor import run_pipeline_async
        >>> processor, future = run_pipeline_async(
        ...     input_path="inputs/input_images",
        ...     output_dir="outputs/output_gaussians",
        ...     slice_radius=10.0,
        ...     enable_semantic=True
        ... )
        >>> # 做其他事情...
        >>> stats = future.result()  # 等待完成
        >>> processor.cleanup()  # 手动清理（或让processor被垃圾回收）
    """
    config = PipelineConfig(
        voxel_size=voxel_size,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        verbose=verbose,
        auto_unload=True,  # 异步模式下也自动卸载
        enable_slice=enable_slice,
        slice_radius=slice_radius,
        enable_semantic=enable_semantic,
        semantic_model=semantic_model,
        semantic_confidence=semantic_confidence,
        colorize_semantic=colorize_semantic,
    )

    processor = PipelineProcessor(config)
    future = processor.process_async(
        Path(input_path), Path(output_dir), callback=callback
    )
    return processor, future


if __name__ == "__main__":
    # 简单测试
    import sys

    if len(sys.argv) < 3:
        print("用法: python pipeline_processor.py <输入目录> <输出目录>")
        sys.exit(1)

    stats = run_pipeline(sys.argv[1], sys.argv[2])
    sys.exit(0 if stats.failed_images == 0 else 1)
