"""流水线处理器模块。

实现多图像流水线处理：
- 模型只加载一次到内存
- 第N张照片进行SHARP推理时，第N-1张照片同时进行体素化
- 支持N张和N+1的流水线作业模式
- 处理完成后自动卸载模型释放内存
- 支持异步处理模式

流水线示意图:
    时间 →
    图片1: [====模型推理====][====体素化====]
    图片2:                   [====模型推理====][====体素化====]
    图片3:                                     [====模型推理====][====体素化====]
    完成后: [====卸载模型====]
"""

import gc
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

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
    ply_output_path: Optional[Path] = None
    voxel_output_path: Optional[Path] = None
    predict_start_time: Optional[float] = None
    predict_end_time: Optional[float] = None
    voxel_start_time: Optional[float] = None
    voxel_end_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class PipelineConfig:
    """流水线配置。"""

    voxel_size: float = 0.005  # 体素尺寸（米）
    remove_ground: bool = True  # 是否移除地面
    transform_coords: bool = False  # 是否转换坐标系
    device: str = "auto"  # 设备选择
    verbose: bool = True  # 详细输出
    checkpoint_path: Optional[Path] = None  # 模型检查点路径
    auto_unload: bool = True  # 处理完成后自动卸载模型
    async_mode: bool = False  # 异步处理模式


@dataclass
class PipelineStats:
    """流水线统计信息。"""

    total_images: int = 0
    completed_images: int = 0
    failed_images: int = 0
    total_predict_time: float = 0.0
    total_voxel_time: float = 0.0
    pipeline_start_time: Optional[float] = None
    pipeline_end_time: Optional[float] = None

    @property
    def total_time(self) -> float:
        if self.pipeline_start_time and self.pipeline_end_time:
            return self.pipeline_end_time - self.pipeline_start_time
        return 0.0

    @property
    def avg_predict_time(self) -> float:
        if self.completed_images > 0:
            return self.total_predict_time / self.completed_images
        return 0.0

    @property
    def avg_voxel_time(self) -> float:
        if self.completed_images > 0:
            return self.total_voxel_time / self.completed_images
        return 0.0


class PipelineLogger:
    """流水线日志记录器，提供详细的格式化输出。"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._lock = threading.Lock()
        self._start_time = time.time()

    def _timestamp(self) -> str:
        """获取相对时间戳。"""
        elapsed = time.time() - self._start_time
        return f"[{elapsed:8.2f}s]"

    def _print(self, msg: str, level: str = "INFO"):
        """线程安全的打印。"""
        with self._lock:
            timestamp = self._timestamp()
            prefix = {
                "INFO": "   ",
                "STAGE": ">>>",
                "OK": " ✓ ",
                "WARN": " ! ",
                "ERROR": " ✗ ",
                "PROGRESS": " → ",
            }.get(level, "   ")
            print(f"{timestamp} {prefix} {msg}")

    def header(self, title: str):
        """打印标题头。"""
        with self._lock:
            print("\n" + "=" * 60)
            print(f"  {title}")
            print("=" * 60)

    def section(self, title: str):
        """打印分节标题。"""
        with self._lock:
            print(f"\n{'─' * 40}")
            print(f"  {title}")
            print(f"{'─' * 40}")

    def stage(self, msg: str):
        """打印阶段信息。"""
        self._print(msg, "STAGE")

    def info(self, msg: str):
        """打印普通信息。"""
        if self.verbose:
            self._print(msg, "INFO")

    def ok(self, msg: str):
        """打印成功信息。"""
        self._print(msg, "OK")

    def warn(self, msg: str):
        """打印警告信息。"""
        self._print(msg, "WARN")

    def error(self, msg: str):
        """打印错误信息。"""
        self._print(msg, "ERROR")

    def progress(self, msg: str):
        """打印进度信息。"""
        self._print(msg, "PROGRESS")

    def task_status(self, tasks: List[ImageTask]):
        """打印任务状态表格。"""
        with self._lock:
            print("\n┌─────┬────────────────────────────┬─────────────┐")
            print("│ No. │ 文件名                     │ 状态        │")
            print("├─────┼────────────────────────────┼─────────────┤")
            for task in tasks:
                name = task.image_path.name[:24]
                status_map = {
                    TaskStatus.PENDING: "⏳ 等待中",
                    TaskStatus.PREDICTING: "🔄 推理中",
                    TaskStatus.PREDICTED: "📦 待体素化",
                    TaskStatus.VOXELIZING: "🔄 体素化中",
                    TaskStatus.COMPLETED: "✅ 完成",
                    TaskStatus.FAILED: "❌ 失败",
                }
                status = status_map.get(task.status, "未知")
                print(f"│ {task.index+1:3d} │ {name:<26} │ {status:<11} │")
            print("└─────┴────────────────────────────┴─────────────┘")

    def stats(self, stats: PipelineStats):
        """打印统计信息。"""
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
            if stats.total_images > 1:
                # 计算流水线效率
                sequential_time = (
                    stats.avg_predict_time + stats.avg_voxel_time
                ) * stats.completed_images
                efficiency = (
                    sequential_time / stats.total_time if stats.total_time > 0 else 0
                )
                print(f"  流水线效率:     {efficiency:.1%}")
            print("=" * 60 + "\n")


class PipelineProcessor:
    """流水线处理器。

    实现模型推理和体素化的流水线并行处理。
    支持上下文管理器，自动清理资源。

    Example:
        # 方式1: 直接使用（自动卸载）
        processor = PipelineProcessor(config)
        stats = processor.process(input_path, output_dir)

        # 方式2: 上下文管理器（推荐，确保资源释放）
        with PipelineProcessor(config) as processor:
            stats = processor.process(input_path, output_dir)

        # 方式3: 异步处理
        processor = PipelineProcessor(config)
        future = processor.process_async(input_path, output_dir)
        # ... 做其他事情 ...
        stats = future.result()  # 等待完成
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.log = PipelineLogger(self.config.verbose)
        self.stats = PipelineStats()

        # 模型相关
        self._predictor = None
        self._device = None
        self._model_loaded = False

        # 体素化器
        self._voxelizer = None

        # 任务管理
        self._tasks: List[ImageTask] = []
        self._predict_queue: Queue = Queue()
        self._voxel_queue: Queue = Queue()

        # 线程控制
        self._stop_event = threading.Event()
        self._predict_lock = threading.Lock()

        # 异步执行器
        self._async_executor: Optional[ThreadPoolExecutor] = None
        self._async_future: Optional[Future] = None

    def __enter__(self):
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出，确保资源释放。"""
        self.cleanup()
        return False

    def __del__(self):
        """析构函数，确保资源释放。"""
        self.cleanup()

    def cleanup(self):
        """清理所有资源。"""
        self._unload_model()
        self._cleanup_voxelizer()
        self._cleanup_async()

    def _unload_model(self):
        """卸载模型，释放GPU/内存。"""
        if not self._model_loaded:
            return

        self.log.stage("卸载模型，释放内存...")

        try:
            # 清除模型引用
            if self._predictor is not None:
                # 将模型移到CPU（如果在GPU上）
                if self._device and self._device.type != "cpu":
                    try:
                        self._predictor.cpu()
                    except Exception:
                        pass

                # 删除模型
                del self._predictor
                self._predictor = None

            # 清除设备引用
            self._device = None
            self._model_loaded = False

            # 强制垃圾回收
            gc.collect()

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 清理MPS缓存（Apple Silicon）
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

            self.log.ok("模型已卸载，内存已释放")

        except Exception as e:
            self.log.warn(f"模型卸载时出现警告: {e}")

    def _cleanup_voxelizer(self):
        """清理体素化器。"""
        if self._voxelizer is not None:
            del self._voxelizer
            self._voxelizer = None

    def _cleanup_async(self):
        """清理异步执行器。"""
        if self._async_executor is not None:
            self._async_executor.shutdown(wait=False)
            self._async_executor = None
        self._async_future = None

    def _detect_device(self) -> torch.device:
        """检测可用设备。"""
        if self.config.device != "auto":
            return torch.device(self.config.device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, "mps") and torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self) -> bool:
        """加载SHARP模型到内存。"""
        self.log.stage("加载SHARP模型到内存...")

        try:
            from sharp.models import PredictorParams, create_predictor

            self._device = self._detect_device()
            self.log.info(f"使用设备: {self._device}")

            # 加载检查点
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

            # 创建预测器
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
        """加载体素化器。"""
        from aylm.tools.pointcloud_voxelizer import PointCloudVoxelizer, VoxelizerConfig

        vox_config = VoxelizerConfig(voxel_size=self.config.voxel_size)
        self._voxelizer = PointCloudVoxelizer(config=vox_config)
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

            image_resized_pt = F.interpolate(
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

    def _voxelize_single(self, task: ImageTask, output_dir: Path) -> bool:
        """对单个PLY文件进行体素化。"""
        if task.status != TaskStatus.PREDICTED or task.ply_output_path is None:
            return False

        task.status = TaskStatus.VOXELIZING
        task.voxel_start_time = time.time()

        self.log.progress(f"[{task.index+1}] 开始体素化: {task.ply_output_path.name}")

        try:
            output_path = output_dir / f"vox_{task.ply_output_path.name}"

            self._voxelizer.process(
                task.ply_output_path,
                output_path,
                remove_ground=self.config.remove_ground,
                transform_coords=self.config.transform_coords,
            )

            task.voxel_output_path = output_path
            task.status = TaskStatus.COMPLETED
            task.voxel_end_time = time.time()

            voxel_time = task.voxel_end_time - task.voxel_start_time
            self.log.ok(
                f"[{task.index+1}] 体素化完成: {task.ply_output_path.name} ({voxel_time:.2f}s)"
            )
            self.log.info(f"    输出: {output_path.name}")

            return True

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.voxel_end_time = time.time()
            self.log.error(f"[{task.index+1}] 体素化失败: {task.ply_output_path.name}")
            self.log.error(f"    错误类型: {type(e).__name__}")
            self.log.error(f"    错误信息: {e}")
            logger.exception(f"体素化异常详情 - {task.ply_output_path.name}")
            return False

    def _collect_images(self, input_path: Path) -> List[Path]:
        """收集输入目录中的图像文件。"""
        extensions = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".bmp"}

        if input_path.is_file():
            if input_path.suffix.lower() in extensions:
                return [input_path]
            return []

        images = []
        for ext in extensions:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))

        return sorted(images)

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        voxel_output_dir: Optional[Path] = None,
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

        # 创建输出目录
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            voxel_output_dir.mkdir(parents=True, exist_ok=True)
            self.log.info(f"PLY输出目录: {output_dir}")
            self.log.info(f"体素化输出目录: {voxel_output_dir}")
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

        # 执行流水线
        self.log.section("阶段 3: 流水线处理")
        self.log.info("流水线模式: 推理(N) || 体素化(N-1)")
        self.log.info("")

        self._execute_pipeline(output_dir, voxel_output_dir)

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

        return self.stats

    def process_async(
        self,
        input_path: Path,
        output_dir: Path,
        voxel_output_dir: Optional[Path] = None,
        callback: Optional[Callable[[PipelineStats], None]] = None,
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

    def wait_for_completion(
        self, timeout: Optional[float] = None
    ) -> Optional[PipelineStats]:
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

    def _execute_pipeline(self, output_dir: Path, voxel_output_dir: Path):
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
        """
        total = len(self._tasks)

        if total == 0:
            self.log.warn("没有任务需要处理")
            return

        # 使用线程池执行体素化（推理在主线程，因为GPU操作需要同步）
        with ThreadPoolExecutor(max_workers=1) as voxel_executor:
            voxel_future: Optional[Future] = None
            prev_task_for_voxel: Optional[ImageTask] = None

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
                        self._voxelize_single, prev_task_for_voxel, voxel_output_dir
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
                if predict_success:
                    prev_task_for_voxel = task
                else:
                    prev_task_for_voxel = None

            # 处理最后一张图片的体素化（同步执行，无并行）
            if prev_task_for_voxel is not None:
                self.log.info(f"\n{'─' * 40}")
                self.log.info("最终阶段: 体素化最后一张图片")
                self._voxelize_single(prev_task_for_voxel, voxel_output_dir)


def run_pipeline(
    input_path: str,
    output_dir: str,
    voxel_size: float = 0.005,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
    auto_unload: bool = True,
) -> PipelineStats:
    """便捷函数：运行流水线处理。

    Args:
        input_path: 输入图像路径或目录
        output_dir: 输出目录
        voxel_size: 体素尺寸（米）
        checkpoint_path: 模型检查点路径
        verbose: 是否详细输出
        auto_unload: 处理完成后自动卸载模型（默认True）

    Returns:
        PipelineStats: 处理统计信息

    Example:
        >>> from aylm.tools.pipeline_processor import run_pipeline
        >>> stats = run_pipeline(
        ...     input_path="inputs/input_images",
        ...     output_dir="outputs/output_gaussians",
        ...     voxel_size=0.005,
        ...     verbose=True
        ... )
        >>> print(f"处理完成: {stats.completed_images}/{stats.total_images}")
    """
    config = PipelineConfig(
        voxel_size=voxel_size,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        verbose=verbose,
        auto_unload=auto_unload,
    )

    # 使用上下文管理器确保资源释放
    with PipelineProcessor(config) as processor:
        return processor.process(Path(input_path), Path(output_dir))


def run_pipeline_async(
    input_path: str,
    output_dir: str,
    voxel_size: float = 0.005,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
    callback: Optional[Callable[[PipelineStats], None]] = None,
) -> Tuple["PipelineProcessor", Future]:
    """便捷函数：异步运行流水线处理。

    Args:
        input_path: 输入图像路径或目录
        output_dir: 输出目录
        voxel_size: 体素尺寸（米）
        checkpoint_path: 模型检查点路径
        verbose: 是否详细输出
        callback: 处理完成后的回调函数

    Returns:
        Tuple[PipelineProcessor, Future]: 处理器实例和Future对象

    Example:
        >>> from aylm.tools.pipeline_processor import run_pipeline_async
        >>> processor, future = run_pipeline_async(
        ...     input_path="inputs/input_images",
        ...     output_dir="outputs/output_gaussians",
        ...     callback=lambda stats: print(f"完成: {stats.completed_images}张")
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
