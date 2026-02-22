"""Tests for video extractor module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from aylm.tools.video_extractor import (
    VideoExtractor,
    extract_video_frames,
)
from aylm.tools.video_types import (
    FrameExtractionMethod,
    VideoConfig,
    VideoMetadata,
)


class TestVideoExtractor:
    """Test VideoExtractor class."""

    def test_init_default_config(self) -> None:
        extractor = VideoExtractor()
        assert extractor.config is not None
        assert extractor.input_dir == Path("inputs/videos")
        assert extractor.output_base_dir == Path("inputs/extracted_frames")

    def test_init_custom_config(self) -> None:
        config = VideoConfig(target_fps=15.0, quality=80)
        extractor = VideoExtractor(config=config)
        assert extractor.config.target_fps == 15.0
        assert extractor.config.quality == 80

    def test_init_custom_paths(self) -> None:
        extractor = VideoExtractor(
            input_dir=Path("/custom/input"),
            output_base_dir=Path("/custom/output"),
        )
        assert extractor.input_dir == Path("/custom/input")
        assert extractor.output_base_dir == Path("/custom/output")


class TestCompressFrame:
    """Test frame compression functionality."""

    def test_compress_no_resize(self) -> None:
        extractor = VideoExtractor()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.compress_frame(frame)
        assert result.shape == (480, 640, 3)

    def test_compress_with_width_only(self) -> None:
        config = VideoConfig(resize_width=320, keep_aspect_ratio=True)
        extractor = VideoExtractor(config=config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.compress_frame(frame)
        assert result.shape[1] == 320
        assert result.shape[0] == 240

    def test_compress_with_height_only(self) -> None:
        config = VideoConfig(resize_height=240, keep_aspect_ratio=True)
        extractor = VideoExtractor(config=config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.compress_frame(frame)
        assert result.shape[0] == 240
        assert result.shape[1] == 320

    def test_compress_with_both_dimensions(self) -> None:
        config = VideoConfig(
            resize_width=320, resize_height=240, keep_aspect_ratio=True
        )
        extractor = VideoExtractor(config=config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.compress_frame(frame)
        assert result.shape[1] <= 320
        assert result.shape[0] <= 240

    def test_compress_without_aspect_ratio(self) -> None:
        config = VideoConfig(
            resize_width=320, resize_height=180, keep_aspect_ratio=False
        )
        extractor = VideoExtractor(config=config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.compress_frame(frame)
        assert result.shape == (180, 320, 3)


class TestCalculateFrameIndices:
    """Test frame index calculation."""

    def _create_metadata(
        self, duration: float = 10.0, fps: float = 30.0, total_frames: int = 300
    ) -> VideoMetadata:
        return VideoMetadata(
            path=Path("/test.mp4"),
            duration=duration,
            fps=fps,
            total_frames=total_frames,
            width=1920,
            height=1080,
        )

    def test_interval_method(self) -> None:
        config = VideoConfig(
            frame_extraction_method=FrameExtractionMethod.INTERVAL,
            frame_interval=1.0,
        )
        extractor = VideoExtractor(config=config)
        metadata = self._create_metadata()
        indices = extractor._calculate_frame_indices(metadata)
        assert len(indices) == 10
        assert indices[0][0] == 0
        assert indices[1][0] == 30

    def test_uniform_method_with_target_fps(self) -> None:
        config = VideoConfig(
            frame_extraction_method=FrameExtractionMethod.UNIFORM,
            target_fps=10.0,
        )
        extractor = VideoExtractor(config=config)
        metadata = self._create_metadata()
        indices = extractor._calculate_frame_indices(metadata)
        assert len(indices) == 100

    def test_uniform_method_all_frames(self) -> None:
        config = VideoConfig(
            frame_extraction_method=FrameExtractionMethod.UNIFORM,
            target_fps=None,
        )
        extractor = VideoExtractor(config=config)
        metadata = self._create_metadata(duration=1.0, total_frames=30)
        indices = extractor._calculate_frame_indices(metadata)
        assert len(indices) == 30

    def test_keyframe_method(self) -> None:
        config = VideoConfig(frame_extraction_method=FrameExtractionMethod.KEYFRAME)
        extractor = VideoExtractor(config=config)
        metadata = self._create_metadata()
        indices = extractor._calculate_frame_indices(metadata)
        assert len(indices) == 10

    def test_max_frames_limit(self) -> None:
        config = VideoConfig(
            frame_extraction_method=FrameExtractionMethod.UNIFORM,
            target_fps=None,
            max_frames=10,
        )
        extractor = VideoExtractor(config=config)
        metadata = self._create_metadata()
        indices = extractor._calculate_frame_indices(metadata)
        assert len(indices) == 10


class TestGetVideoMetadata:
    """Test video metadata extraction."""

    def test_get_metadata_invalid_file(self) -> None:
        extractor = VideoExtractor()
        with pytest.raises(ValueError, match="Cannot open video file"):
            extractor.get_video_metadata(Path("/nonexistent/video.mp4"))


class TestExtractFrames:
    """Test frame extraction functionality."""

    def test_extract_frames_invalid_video(self) -> None:
        extractor = VideoExtractor()
        result = extractor.extract_frames(Path("/nonexistent/video.mp4"))
        assert result.success is False
        assert result.error_message is not None

    def test_extract_frames_async_invalid_video(self) -> None:
        extractor = VideoExtractor()
        result = extractor.extract_frames_async(Path("/nonexistent/video.mp4"))
        assert result.success is False
        assert result.error_message is not None


class TestExtractVideoFramesFunction:
    """Test the convenience function."""

    def test_extract_video_frames_invalid(self) -> None:
        result = extract_video_frames(Path("/nonexistent/video.mp4"))
        assert result.success is False


class TestFrameExtractionWithMock:
    """Test frame extraction with mocked video capture."""

    def _mock_cap_get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)

    @patch("cv2.VideoCapture")
    def test_get_metadata_with_mock(self, mock_capture_class: MagicMock) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = self._mock_cap_get
        mock_capture_class.return_value = mock_cap

        extractor = VideoExtractor()
        metadata = extractor.get_video_metadata(Path("/test/video.mp4"))

        assert metadata.fps == 30.0
        assert metadata.total_frames == 300
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.duration == 10.0

    @patch("cv2.VideoCapture")
    @patch("cv2.imwrite")
    def test_extract_single_frame_with_mock(
        self, mock_imwrite: MagicMock, mock_capture_class: MagicMock
    ) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 30.0}.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture_class.return_value = mock_cap
        mock_imwrite.return_value = True

        extractor = VideoExtractor()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frame_000000.png"
            frame_info = extractor.extract_frame(
                Path("/test/video.mp4"), 0, output_path
            )

            assert frame_info is not None
            assert frame_info.index == 0
            assert frame_info.output_path == output_path


class TestOutputFormats:
    """Test different output format configurations."""

    @pytest.mark.parametrize(
        ("output_format", "expected_ext"),
        [("png", "png"), ("jpg", "jpg"), ("jpeg", "jpeg"), ("webp", "webp")],
    )
    def test_output_format_extension(
        self, output_format: str, expected_ext: str
    ) -> None:
        config = VideoConfig(output_format=output_format)
        extractor = VideoExtractor(config=config)
        assert extractor.config.output_format == output_format


class TestProgressCallback:
    """Test progress callback functionality."""

    @patch("cv2.VideoCapture")
    def test_progress_callback_called(self, mock_capture_class: MagicMock) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture_class.return_value = mock_cap

        callback_calls: list[tuple[int, int, float]] = []

        def progress_callback(current: int, total: int, percentage: float) -> None:
            callback_calls.append((current, total, percentage))

        config = VideoConfig(
            frame_extraction_method=FrameExtractionMethod.KEYFRAME,
            max_frames=1,
        )
        extractor = VideoExtractor(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor.extract_frames(
                Path("/test/video.mp4"),
                output_dir=Path(tmpdir),
                progress_callback=progress_callback,
            )

        assert len(callback_calls) >= 1
