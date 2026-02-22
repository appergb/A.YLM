"""视频处理模块类型定义。

定义视频处理相关的数据类、枚举和配置参数，
与现有PipelineProcessor架构兼容，支持异步处理和GPU加速。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class VideoTaskStatus(Enum):
    """视频任务状态枚举。"""

    PENDING = "pending"  # 等待处理
    EXTRACTING = "extracting"  # 帧提取中
    EXTRACTED = "extracted"  # 帧提取完成
    PROCESSING = "processing"  # 帧处理中（推理+体素化）
    COMPLETED = "completed"  # 全部完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class FrameExtractionMethod(Enum):
    """帧提取方法枚举。"""

    UNIFORM = "uniform"  # 均匀采样
    KEYFRAME = "keyframe"  # 关键帧提取
    SCENE_CHANGE = "scene_change"  # 场景变化检测
    INTERVAL = "interval"  # 固定时间间隔


class VideoCodec(Enum):
    """视频编解码器枚举。"""

    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"
    PRORES = "prores"
    AUTO = "auto"  # 自动检测


class GPUAcceleration(Enum):
    """GPU加速类型枚举。"""

    NONE = "none"  # 不使用GPU加速
    CUDA = "cuda"  # NVIDIA CUDA
    VIDEOTOOLBOX = "videotoolbox"  # Apple VideoToolbox
    VAAPI = "vaapi"  # Linux VA-API
    AUTO = "auto"  # 自动检测


@dataclass
class VideoConfig:
    """视频处理配置参数。

    与PipelineConfig兼容，扩展视频特定参数。

    Attributes:
        frame_extraction_method: 帧提取方法
        target_fps: 目标帧率（None表示使用原始帧率）
        max_frames: 最大提取帧数（None表示不限制）
        frame_interval: 帧间隔（仅INTERVAL方法使用）
        scene_threshold: 场景变化阈值（仅SCENE_CHANGE方法使用）
        gpu_acceleration: GPU加速类型
        hw_decode: 是否使用硬件解码
        hw_encode: 是否使用硬件编码
        output_format: 输出帧格式
        quality: 输出质量（1-100）
        resize_width: 调整宽度（None保持原始）
        resize_height: 调整高度（None保持原始）
        keep_aspect_ratio: 保持宽高比
        temp_dir: 临时文件目录
        cleanup_temp: 处理完成后清理临时文件
        parallel_extraction: 并行提取帧数
        device: 推理设备（auto/cuda/mps/cpu）
        verbose: 详细输出
    """

    # 帧提取参数
    frame_extraction_method: FrameExtractionMethod = FrameExtractionMethod.UNIFORM
    target_fps: float | None = None
    max_frames: int | None = None
    frame_interval: float = 1.0  # 秒
    scene_threshold: float = 0.3

    # GPU加速参数
    gpu_acceleration: GPUAcceleration = GPUAcceleration.AUTO
    hw_decode: bool = True
    hw_encode: bool = False

    # 输出参数
    output_format: str = "png"
    quality: int = 95
    resize_width: int | None = None
    resize_height: int | None = None
    keep_aspect_ratio: bool = True

    # 处理参数
    temp_dir: Path | None = None
    cleanup_temp: bool = True
    parallel_extraction: int = 4

    # 与PipelineConfig兼容的参数
    device: str = "auto"
    verbose: bool = True


@dataclass
class VideoMetadata:
    """视频元数据。

    Attributes:
        path: 视频文件路径
        duration: 时长（秒）
        fps: 帧率
        total_frames: 总帧数
        width: 宽度
        height: 高度
        codec: 编解码器
        bitrate: 比特率（bps）
        has_audio: 是否包含音频
        rotation: 旋转角度
        creation_time: 创建时间
    """

    path: Path
    duration: float
    fps: float
    total_frames: int
    width: int
    height: int
    codec: VideoCodec = VideoCodec.AUTO
    bitrate: int | None = None
    has_audio: bool = False
    rotation: int = 0
    creation_time: str | None = None


@dataclass
class FrameInfo:
    """单帧信息。

    Attributes:
        index: 帧索引（在视频中的位置）
        timestamp: 时间戳（秒）
        output_path: 输出文件路径
        is_keyframe: 是否为关键帧
        scene_score: 场景变化分数
    """

    index: int
    timestamp: float
    output_path: Path | None = None
    is_keyframe: bool = False
    scene_score: float = 0.0


@dataclass
class FrameExtractionResult:
    """帧提取结果。

    Attributes:
        video_path: 源视频路径
        output_dir: 输出目录
        frames: 提取的帧信息列表
        total_extracted: 提取的帧数
        extraction_time: 提取耗时（秒）
        metadata: 视频元数据
        error_message: 错误信息（如果失败）
    """

    video_path: Path
    output_dir: Path
    frames: list[FrameInfo] = field(default_factory=list)
    total_extracted: int = 0
    extraction_time: float = 0.0
    metadata: VideoMetadata | None = None
    error_message: str | None = None

    @property
    def success(self) -> bool:
        """是否成功提取。"""
        return self.total_extracted > 0 and self.error_message is None

    @property
    def frame_paths(self) -> list[Path]:
        """获取所有帧文件路径。"""
        return [f.output_path for f in self.frames if f.output_path is not None]


@dataclass
class VideoTask:
    """视频处理任务。

    与ImageTask类似，跟踪单个视频的处理状态。

    Attributes:
        video_path: 视频文件路径
        index: 任务索引
        status: 任务状态
        config: 视频配置
        metadata: 视频元数据
        extraction_result: 帧提取结果
        output_dir: 输出目录
        processed_frames: 已处理帧数
        total_frames: 总帧数
        start_time: 开始时间
        end_time: 结束时间
        error_message: 错误信息
    """

    video_path: Path
    index: int
    status: VideoTaskStatus = VideoTaskStatus.PENDING
    config: VideoConfig | None = None
    metadata: VideoMetadata | None = None
    extraction_result: FrameExtractionResult | None = None
    output_dir: Path | None = None
    processed_frames: int = 0
    total_frames: int = 0
    start_time: float | None = None
    end_time: float | None = None
    error_message: str | None = None

    @property
    def progress(self) -> float:
        """获取处理进度（0.0-1.0）。"""
        if self.total_frames == 0:
            return 0.0
        return self.processed_frames / self.total_frames

    @property
    def elapsed_time(self) -> float | None:
        """获取已用时间（秒）。"""
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


@dataclass
class VideoProcessingStats:
    """视频处理统计信息。

    与PipelineStats类似，跟踪整体处理统计。

    Attributes:
        total_videos: 总视频数
        completed_videos: 完成的视频数
        failed_videos: 失败的视频数
        total_frames_extracted: 提取的总帧数
        total_frames_processed: 处理的总帧数
        total_extraction_time: 帧提取总耗时
        total_processing_time: 帧处理总耗时
        pipeline_start_time: 流水线开始时间
        pipeline_end_time: 流水线结束时间
    """

    total_videos: int = 0
    completed_videos: int = 0
    failed_videos: int = 0
    total_frames_extracted: int = 0
    total_frames_processed: int = 0
    total_extraction_time: float = 0.0
    total_processing_time: float = 0.0
    pipeline_start_time: float | None = None
    pipeline_end_time: float | None = None

    @property
    def total_time(self) -> float:
        """获取总耗时。"""
        if self.pipeline_start_time and self.pipeline_end_time:
            return self.pipeline_end_time - self.pipeline_start_time
        return 0.0

    @property
    def avg_extraction_time(self) -> float:
        """获取平均帧提取时间。"""
        if self.completed_videos > 0:
            return self.total_extraction_time / self.completed_videos
        return 0.0

    @property
    def avg_processing_time_per_frame(self) -> float:
        """获取每帧平均处理时间。"""
        if self.total_frames_processed > 0:
            return self.total_processing_time / self.total_frames_processed
        return 0.0

    @property
    def frames_per_second(self) -> float:
        """获取处理速度（帧/秒）。"""
        if self.total_time > 0:
            return self.total_frames_processed / self.total_time
        return 0.0
