"""Tests for pipeline processor module."""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from aylm.tools.pipeline_processor import (
    ImageTask,
    PipelineConfig,
    PipelineLogger,
    PipelineProcessor,
    PipelineStats,
    TaskStatus,
)


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_all_status_values_exist(self) -> None:
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.PREDICTING.value == "predicting"
        assert TaskStatus.PREDICTED.value == "predicted"
        assert TaskStatus.VOXELIZING.value == "voxelizing"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert len(TaskStatus) == 6


class TestImageTask:
    """Test ImageTask dataclass."""

    def test_required_fields(self) -> None:
        task = ImageTask(image_path=Path("/test/image.jpg"), index=0)
        assert task.image_path == Path("/test/image.jpg")
        assert task.index == 0

    def test_default_values(self) -> None:
        task = ImageTask(image_path=Path("/test/image.jpg"), index=0)
        assert task.status == TaskStatus.PENDING
        assert task.ply_output_path is None
        assert task.voxel_output_path is None
        assert task.predict_start_time is None
        assert task.predict_end_time is None
        assert task.voxel_start_time is None
        assert task.voxel_end_time is None
        assert task.error_message is None

    def test_status_transitions(self) -> None:
        task = ImageTask(image_path=Path("/test/image.jpg"), index=0)
        task.status = TaskStatus.PREDICTING
        assert task.status == TaskStatus.PREDICTING
        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_values(self) -> None:
        config = PipelineConfig()
        assert config.voxel_size == 0.005
        assert config.remove_ground is True
        assert config.transform_coords is False
        assert config.device == "auto"
        assert config.verbose is True
        assert config.checkpoint_path is None
        assert config.auto_unload is True
        assert config.async_mode is False

    def test_custom_values(self) -> None:
        config = PipelineConfig(
            voxel_size=0.01,
            remove_ground=False,
            device="cuda",
            verbose=False,
        )
        assert config.voxel_size == 0.01
        assert config.remove_ground is False
        assert config.device == "cuda"
        assert config.verbose is False


class TestPipelineStats:
    """Test PipelineStats dataclass."""

    def test_default_values(self) -> None:
        stats = PipelineStats()
        assert stats.total_images == 0
        assert stats.completed_images == 0
        assert stats.failed_images == 0
        assert stats.total_predict_time == 0.0
        assert stats.total_voxel_time == 0.0
        assert stats.pipeline_start_time is None
        assert stats.pipeline_end_time is None

    def test_total_time_property(self) -> None:
        stats = PipelineStats()
        assert stats.total_time == 0.0

        stats = PipelineStats(pipeline_start_time=100.0, pipeline_end_time=150.0)
        assert stats.total_time == 50.0

    def test_avg_predict_time(self) -> None:
        stats = PipelineStats()
        assert stats.avg_predict_time == 0.0

        stats = PipelineStats(completed_images=5, total_predict_time=25.0)
        assert stats.avg_predict_time == 5.0

    def test_avg_voxel_time(self) -> None:
        stats = PipelineStats()
        assert stats.avg_voxel_time == 0.0

        stats = PipelineStats(completed_images=4, total_voxel_time=20.0)
        assert stats.avg_voxel_time == 5.0


class TestPipelineLogger:
    """Test PipelineLogger class."""

    def test_init(self) -> None:
        logger = PipelineLogger(verbose=True)
        assert logger.verbose is True

        logger = PipelineLogger(verbose=False)
        assert logger.verbose is False

    def test_timestamp_format(self) -> None:
        logger = PipelineLogger()
        timestamp = logger._timestamp()
        assert timestamp.startswith("[")
        assert timestamp.endswith("s]")

    def test_header_output(self, capsys: Any) -> None:
        logger = PipelineLogger()
        logger.header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out
        assert "=" in captured.out

    def test_section_output(self, capsys: Any) -> None:
        logger = PipelineLogger()
        logger.section("Test Section")
        captured = capsys.readouterr()
        assert "Test Section" in captured.out

    def test_info_verbose(self, capsys: Any) -> None:
        logger = PipelineLogger(verbose=True)
        logger.info("Test info message")
        captured = capsys.readouterr()
        assert "Test info message" in captured.out

    def test_info_quiet(self, capsys: Any) -> None:
        logger = PipelineLogger(verbose=False)
        logger.info("Test info message")
        captured = capsys.readouterr()
        assert "Test info message" not in captured.out

    def test_ok_output(self, capsys: Any) -> None:
        logger = PipelineLogger()
        logger.ok("Success message")
        captured = capsys.readouterr()
        assert "Success message" in captured.out

    def test_error_output(self, capsys: Any) -> None:
        logger = PipelineLogger()
        logger.error("Error message")
        captured = capsys.readouterr()
        assert "Error message" in captured.out


class TestPipelineProcessor:
    """Test PipelineProcessor class."""

    def test_init_default_config(self) -> None:
        processor = PipelineProcessor()
        assert processor.config.voxel_size == 0.005
        assert processor._model_loaded is False

    def test_init_custom_config(self) -> None:
        config = PipelineConfig(voxel_size=0.01)
        processor = PipelineProcessor(config=config)
        assert processor.config.voxel_size == 0.01

    def test_context_manager(self) -> None:
        with PipelineProcessor() as processor:
            assert processor is not None

    def test_collect_images_single_file(self) -> None:
        processor = PipelineProcessor()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            filepath = Path(f.name)

        try:
            images = processor._collect_images(filepath)
            assert len(images) == 1
            assert images[0] == filepath
        finally:
            filepath.unlink()

    def test_collect_images_directory(self) -> None:
        processor = PipelineProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "image1.jpg").touch()
            (tmppath / "image2.png").touch()
            (tmppath / "image3.jpeg").touch()
            (tmppath / "not_image.txt").touch()

            images = processor._collect_images(tmppath)
            assert len(images) == 3

    def test_collect_images_empty_directory(self) -> None:
        processor = PipelineProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            images = processor._collect_images(Path(tmpdir))
            assert len(images) == 0

    def test_collect_images_unsupported_extension(self) -> None:
        processor = PipelineProcessor()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            filepath = Path(f.name)

        try:
            images = processor._collect_images(filepath)
            assert len(images) == 0
        finally:
            filepath.unlink()

    def test_detect_device_auto(self) -> None:
        config = PipelineConfig(device="auto")
        processor = PipelineProcessor(config=config)
        device = processor._detect_device()
        assert device.type in ["cpu", "cuda", "mps"]

    def test_detect_device_explicit(self) -> None:
        config = PipelineConfig(device="cpu")
        processor = PipelineProcessor(config=config)
        device = processor._detect_device()
        assert device.type == "cpu"

    def test_cleanup_without_model(self) -> None:
        processor = PipelineProcessor()
        processor.cleanup()

    def test_is_processing_not_started(self) -> None:
        processor = PipelineProcessor()
        assert processor.is_processing() is False

    def test_cancel_not_started(self) -> None:
        processor = PipelineProcessor()
        assert processor.cancel() is False


class TestPipelineProcessorIntegration:
    """Integration tests for PipelineProcessor."""

    def test_process_nonexistent_input(self) -> None:
        processor = PipelineProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = processor.process(Path("/nonexistent/path"), Path(tmpdir))
            assert stats.total_images == 0

    def test_process_empty_directory(self) -> None:
        processor = PipelineProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            stats = processor.process(input_dir, output_dir)
            assert stats.total_images == 0

    @pytest.mark.skip(reason="Requires SHARP model to be installed")
    def test_process_with_images(self) -> None:
        pass


class TestTaskStatusTable:
    """Test task status table output."""

    def test_task_status_output(self, capsys: Any) -> None:
        logger = PipelineLogger()
        tasks = [
            ImageTask(image_path=Path("/test/image1.jpg"), index=0),
            ImageTask(
                image_path=Path("/test/image2.jpg"),
                index=1,
                status=TaskStatus.COMPLETED,
            ),
            ImageTask(
                image_path=Path("/test/image3.jpg"),
                index=2,
                status=TaskStatus.FAILED,
            ),
        ]
        logger.task_status(tasks)
        captured = capsys.readouterr()
        assert "image1.jpg" in captured.out
        assert "image2.jpg" in captured.out
        assert "image3.jpg" in captured.out


class TestPipelineStatsOutput:
    """Test pipeline stats output."""

    def test_stats_output(self, capsys: Any) -> None:
        logger = PipelineLogger()
        stats = PipelineStats(
            total_images=10,
            completed_images=8,
            failed_images=2,
            total_predict_time=40.0,
            total_voxel_time=20.0,
            pipeline_start_time=0.0,
            pipeline_end_time=60.0,
        )
        logger.stats(stats)
        captured = capsys.readouterr()
        assert "10" in captured.out
        assert "8" in captured.out
        assert "2" in captured.out
