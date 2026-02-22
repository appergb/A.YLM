"""Tests for video types module."""

import time
from pathlib import Path

from aylm.tools.video_types import (
    FrameExtractionMethod,
    FrameExtractionResult,
    FrameInfo,
    GPUAcceleration,
    VideoCodec,
    VideoConfig,
    VideoMetadata,
    VideoProcessingStats,
    VideoTask,
    VideoTaskStatus,
)


class TestEnums:
    """Test enum types."""

    def test_video_task_status_values(self) -> None:
        assert VideoTaskStatus.PENDING.value == "pending"
        assert VideoTaskStatus.EXTRACTING.value == "extracting"
        assert VideoTaskStatus.EXTRACTED.value == "extracted"
        assert VideoTaskStatus.PROCESSING.value == "processing"
        assert VideoTaskStatus.COMPLETED.value == "completed"
        assert VideoTaskStatus.FAILED.value == "failed"
        assert VideoTaskStatus.CANCELLED.value == "cancelled"
        assert len(VideoTaskStatus) == 7

    def test_frame_extraction_method_values(self) -> None:
        assert FrameExtractionMethod.UNIFORM.value == "uniform"
        assert FrameExtractionMethod.KEYFRAME.value == "keyframe"
        assert FrameExtractionMethod.SCENE_CHANGE.value == "scene_change"
        assert FrameExtractionMethod.INTERVAL.value == "interval"
        assert len(FrameExtractionMethod) == 4

    def test_video_codec_values(self) -> None:
        assert VideoCodec.H264.value == "h264"
        assert VideoCodec.H265.value == "h265"
        assert VideoCodec.VP9.value == "vp9"
        assert VideoCodec.AV1.value == "av1"
        assert VideoCodec.PRORES.value == "prores"
        assert VideoCodec.AUTO.value == "auto"
        assert len(VideoCodec) == 6

    def test_gpu_acceleration_values(self) -> None:
        assert GPUAcceleration.NONE.value == "none"
        assert GPUAcceleration.CUDA.value == "cuda"
        assert GPUAcceleration.VIDEOTOOLBOX.value == "videotoolbox"
        assert GPUAcceleration.VAAPI.value == "vaapi"
        assert GPUAcceleration.AUTO.value == "auto"
        assert len(GPUAcceleration) == 5


class TestVideoConfig:
    """Test VideoConfig dataclass."""

    def test_default_values(self) -> None:
        config = VideoConfig()
        assert config.frame_extraction_method == FrameExtractionMethod.UNIFORM
        assert config.target_fps is None
        assert config.max_frames is None
        assert config.frame_interval == 1.0
        assert config.scene_threshold == 0.3
        assert config.gpu_acceleration == GPUAcceleration.AUTO
        assert config.hw_decode is True
        assert config.hw_encode is False
        assert config.output_format == "png"
        assert config.quality == 95
        assert config.resize_width is None
        assert config.resize_height is None
        assert config.keep_aspect_ratio is True
        assert config.temp_dir is None
        assert config.cleanup_temp is True
        assert config.parallel_extraction == 4
        assert config.device == "auto"
        assert config.verbose is True

    def test_custom_values(self) -> None:
        config = VideoConfig(
            frame_extraction_method=FrameExtractionMethod.KEYFRAME,
            target_fps=30.0,
            max_frames=100,
            quality=80,
            device="cuda",
        )
        assert config.frame_extraction_method == FrameExtractionMethod.KEYFRAME
        assert config.target_fps == 30.0
        assert config.max_frames == 100
        assert config.quality == 80
        assert config.device == "cuda"


class TestVideoMetadata:
    """Test VideoMetadata dataclass."""

    def test_required_fields(self) -> None:
        metadata = VideoMetadata(
            path=Path("/test/video.mp4"),
            duration=120.5,
            fps=30.0,
            total_frames=3615,
            width=1920,
            height=1080,
        )
        assert metadata.path == Path("/test/video.mp4")
        assert metadata.duration == 120.5
        assert metadata.fps == 30.0
        assert metadata.total_frames == 3615
        assert metadata.width == 1920
        assert metadata.height == 1080

    def test_default_optional_fields(self) -> None:
        metadata = VideoMetadata(
            path=Path("/test/video.mp4"),
            duration=60.0,
            fps=24.0,
            total_frames=1440,
            width=1280,
            height=720,
        )
        assert metadata.codec == VideoCodec.AUTO
        assert metadata.bitrate is None
        assert metadata.has_audio is False
        assert metadata.rotation == 0
        assert metadata.creation_time is None

    def test_custom_optional_fields(self) -> None:
        metadata = VideoMetadata(
            path=Path("/test/video.mp4"),
            duration=60.0,
            fps=24.0,
            total_frames=1440,
            width=1280,
            height=720,
            codec=VideoCodec.H265,
            bitrate=5000000,
            has_audio=True,
            rotation=90,
            creation_time="2024-01-01T12:00:00",
        )
        assert metadata.codec == VideoCodec.H265
        assert metadata.bitrate == 5000000
        assert metadata.has_audio is True
        assert metadata.rotation == 90
        assert metadata.creation_time == "2024-01-01T12:00:00"


class TestFrameInfo:
    """Test FrameInfo dataclass."""

    def test_required_fields(self) -> None:
        frame = FrameInfo(index=0, timestamp=0.0)
        assert frame.index == 0
        assert frame.timestamp == 0.0

    def test_default_optional_fields(self) -> None:
        frame = FrameInfo(index=10, timestamp=0.5)
        assert frame.output_path is None
        assert frame.is_keyframe is False
        assert frame.scene_score == 0.0

    def test_custom_optional_fields(self) -> None:
        frame = FrameInfo(
            index=100,
            timestamp=4.0,
            output_path=Path("/output/frame_100.png"),
            is_keyframe=True,
            scene_score=0.85,
        )
        assert frame.output_path == Path("/output/frame_100.png")
        assert frame.is_keyframe is True
        assert frame.scene_score == 0.85


class TestFrameExtractionResult:
    """Test FrameExtractionResult dataclass."""

    def test_required_fields(self) -> None:
        result = FrameExtractionResult(
            video_path=Path("/test/video.mp4"),
            output_dir=Path("/output"),
        )
        assert result.video_path == Path("/test/video.mp4")
        assert result.output_dir == Path("/output")

    def test_default_values(self) -> None:
        result = FrameExtractionResult(
            video_path=Path("/test/video.mp4"),
            output_dir=Path("/output"),
        )
        assert result.frames == []
        assert result.total_extracted == 0
        assert result.extraction_time == 0.0
        assert result.metadata is None
        assert result.error_message is None

    def test_success_property(self) -> None:
        # Success case
        result = FrameExtractionResult(
            video_path=Path("/test/video.mp4"),
            output_dir=Path("/output"),
            total_extracted=10,
            error_message=None,
        )
        assert result.success is True

        # No frames case
        result = FrameExtractionResult(
            video_path=Path("/test/video.mp4"),
            output_dir=Path("/output"),
            total_extracted=0,
        )
        assert result.success is False

        # Error case
        result = FrameExtractionResult(
            video_path=Path("/test/video.mp4"),
            output_dir=Path("/output"),
            total_extracted=5,
            error_message="Some error",
        )
        assert result.success is False

    def test_frame_paths_property(self) -> None:
        frames = [
            FrameInfo(index=0, timestamp=0.0, output_path=Path("/output/frame_0.png")),
            FrameInfo(index=1, timestamp=0.1, output_path=Path("/output/frame_1.png")),
            FrameInfo(index=2, timestamp=0.2, output_path=None),
        ]
        result = FrameExtractionResult(
            video_path=Path("/test/video.mp4"),
            output_dir=Path("/output"),
            frames=frames,
        )
        paths = result.frame_paths
        assert len(paths) == 2
        assert Path("/output/frame_0.png") in paths
        assert Path("/output/frame_1.png") in paths


class TestVideoTask:
    """Test VideoTask dataclass."""

    def test_required_fields(self) -> None:
        task = VideoTask(video_path=Path("/test/video.mp4"), index=0)
        assert task.video_path == Path("/test/video.mp4")
        assert task.index == 0

    def test_default_values(self) -> None:
        task = VideoTask(video_path=Path("/test/video.mp4"), index=0)
        assert task.status == VideoTaskStatus.PENDING
        assert task.config is None
        assert task.metadata is None
        assert task.extraction_result is None
        assert task.output_dir is None
        assert task.processed_frames == 0
        assert task.total_frames == 0
        assert task.start_time is None
        assert task.end_time is None
        assert task.error_message is None

    def test_progress_property(self) -> None:
        # Zero total
        task = VideoTask(video_path=Path("/test/video.mp4"), index=0, total_frames=0)
        assert task.progress == 0.0

        # Partial
        task = VideoTask(
            video_path=Path("/test/video.mp4"),
            index=0,
            processed_frames=50,
            total_frames=100,
        )
        assert task.progress == 0.5

        # Complete
        task = VideoTask(
            video_path=Path("/test/video.mp4"),
            index=0,
            processed_frames=100,
            total_frames=100,
        )
        assert task.progress == 1.0

    def test_elapsed_time_property(self) -> None:
        # Not started
        task = VideoTask(video_path=Path("/test/video.mp4"), index=0)
        assert task.elapsed_time is None

        # In progress
        task = VideoTask(
            video_path=Path("/test/video.mp4"),
            index=0,
            start_time=time.time() - 5.0,
        )
        elapsed = task.elapsed_time
        assert elapsed is not None
        assert 5.0 <= elapsed < 6.0

        # Completed
        start = time.time() - 10.0
        end = time.time() - 5.0
        task = VideoTask(
            video_path=Path("/test/video.mp4"),
            index=0,
            start_time=start,
            end_time=end,
        )
        elapsed = task.elapsed_time
        assert elapsed is not None
        assert 4.9 < elapsed < 5.1


class TestVideoProcessingStats:
    """Test VideoProcessingStats dataclass."""

    def test_default_values(self) -> None:
        stats = VideoProcessingStats()
        assert stats.total_videos == 0
        assert stats.completed_videos == 0
        assert stats.failed_videos == 0
        assert stats.total_frames_extracted == 0
        assert stats.total_frames_processed == 0
        assert stats.total_extraction_time == 0.0
        assert stats.total_processing_time == 0.0
        assert stats.pipeline_start_time is None
        assert stats.pipeline_end_time is None

    def test_total_time_property(self) -> None:
        stats = VideoProcessingStats()
        assert stats.total_time == 0.0

        stats = VideoProcessingStats(pipeline_start_time=100.0, pipeline_end_time=150.0)
        assert stats.total_time == 50.0

    def test_avg_extraction_time_property(self) -> None:
        stats = VideoProcessingStats()
        assert stats.avg_extraction_time == 0.0

        stats = VideoProcessingStats(completed_videos=5, total_extraction_time=25.0)
        assert stats.avg_extraction_time == 5.0

    def test_avg_processing_time_per_frame(self) -> None:
        stats = VideoProcessingStats()
        assert stats.avg_processing_time_per_frame == 0.0

        stats = VideoProcessingStats(
            total_frames_processed=100, total_processing_time=50.0
        )
        assert stats.avg_processing_time_per_frame == 0.5

    def test_frames_per_second(self) -> None:
        stats = VideoProcessingStats()
        assert stats.frames_per_second == 0.0

        stats = VideoProcessingStats(
            total_frames_processed=300,
            pipeline_start_time=100.0,
            pipeline_end_time=110.0,
        )
        assert stats.frames_per_second == 30.0
