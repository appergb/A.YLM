"""Tests for video pipeline module."""

from unittest.mock import MagicMock, patch

import pytest

from aylm.tools.video_pipeline import (
    VideoPipelineConfig,
    VideoPipelineProcessor,
)


class TestVideoPipelineConfig:
    """Test VideoPipelineConfig dataclass."""

    def test_default_values(self) -> None:
        config = VideoPipelineConfig()
        assert config.video_config is None
        assert config.voxel_size == 0.005
        assert config.remove_ground is True
        assert config.transform_coords is False
        assert config.use_gpu is True
        assert config.frame_queue_size == 10
        assert config.checkpoint_path is None
        assert config.device == "auto"
        assert config.verbose is True
        assert config.auto_unload is True

    def test_custom_values(self) -> None:
        config = VideoPipelineConfig(
            voxel_size=0.01,
            remove_ground=False,
            use_gpu=False,
            device="cpu",
            verbose=False,
        )
        assert config.voxel_size == 0.01
        assert config.remove_ground is False
        assert config.use_gpu is False
        assert config.device == "cpu"
        assert config.verbose is False


class TestVideoPipelineProcessor:
    """Test VideoPipelineProcessor class."""

    def test_init_default_config(self) -> None:
        processor = VideoPipelineProcessor()
        assert processor.config.voxel_size == 0.005
        assert processor._model_loaded is False

    def test_init_custom_config(self) -> None:
        config = VideoPipelineConfig(voxel_size=0.01)
        processor = VideoPipelineProcessor(config=config)
        assert processor.config.voxel_size == 0.01

    def test_detect_device_auto(self) -> None:
        config = VideoPipelineConfig(device="auto")
        processor = VideoPipelineProcessor(config=config)
        device = processor._detect_device()
        assert device.type in ["cpu", "cuda", "mps"]

    def test_detect_device_explicit(self) -> None:
        config = VideoPipelineConfig(device="cpu")
        processor = VideoPipelineProcessor(config=config)
        device = processor._detect_device()
        assert device.type == "cpu"

    def test_cleanup_without_model(self) -> None:
        processor = VideoPipelineProcessor()
        processor.cleanup()

    def test_unload_model_not_loaded(self) -> None:
        processor = VideoPipelineProcessor()
        processor._unload_model()


class TestVideoPipelineProcessorIntegration:
    """Integration tests for VideoPipelineProcessor."""

    @pytest.mark.skip(reason="Requires SHARP model to be installed")
    def test_process_nonexistent_video(self) -> None:
        pass

    @pytest.mark.skip(reason="Requires SHARP model to be installed")
    def test_process_with_video(self) -> None:
        pass


class TestProcessVideoFunction:
    """Test the convenience function."""

    def test_process_video_returns_stats(self) -> None:
        pass


class TestVideoPipelineStats:
    """Test pipeline statistics."""

    def test_stats_initialization(self) -> None:
        processor = VideoPipelineProcessor()
        assert processor.stats.total_videos == 0
        assert processor.stats.completed_videos == 0
        assert processor.stats.failed_videos == 0


class TestFrameQueue:
    """Test frame queue functionality."""

    def test_frame_queue_size(self) -> None:
        config = VideoPipelineConfig(frame_queue_size=5)
        processor = VideoPipelineProcessor(config=config)
        assert processor._frame_queue.maxsize == 5

    def test_frame_queue_default_size(self) -> None:
        processor = VideoPipelineProcessor()
        assert processor._frame_queue.maxsize == 10


class TestVoxelizerConfiguration:
    """Test voxelizer configuration."""

    def test_voxelizer_not_loaded_initially(self) -> None:
        processor = VideoPipelineProcessor()
        assert processor._voxelizer is None

    @patch("aylm.tools.video_pipeline.PointCloudVoxelizer")
    def test_load_voxelizer(self, mock_voxelizer_class: MagicMock) -> None:
        mock_voxelizer = MagicMock()
        mock_voxelizer_class.return_value = mock_voxelizer

        config = VideoPipelineConfig(voxel_size=0.01, use_gpu=False)
        processor = VideoPipelineProcessor(config=config)
        processor._load_voxelizer()

        assert processor._voxelizer is not None
        mock_voxelizer_class.assert_called_once()


class TestModelLoading:
    """Test model loading functionality."""

    def test_model_not_loaded_initially(self) -> None:
        processor = VideoPipelineProcessor()
        assert processor._model_loaded is False
        assert processor._predictor is None

    @pytest.mark.skip(reason="Requires sharp module to be installed")
    @patch("aylm.tools.video_pipeline.create_predictor")
    @patch("torch.hub.load_state_dict_from_url")
    def test_load_model_from_url(
        self, mock_load_url: MagicMock, mock_create_predictor: MagicMock
    ) -> None:
        pass


class TestStopEvent:
    """Test stop event functionality."""

    def test_stop_event_initially_clear(self) -> None:
        processor = VideoPipelineProcessor()
        assert not processor._stop_event.is_set()

    def test_cleanup_sets_stop_event(self) -> None:
        processor = VideoPipelineProcessor()
        processor.cleanup()
        assert processor._stop_event.is_set()


class TestExtractionDoneEvent:
    """Test extraction done event functionality."""

    def test_extraction_done_initially_clear(self) -> None:
        processor = VideoPipelineProcessor()
        assert not processor._extraction_done.is_set()
