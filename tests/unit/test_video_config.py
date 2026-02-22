"""Tests for video configuration module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from aylm.tools.video_config import (
    DEFAULT_CONFIG,
    ConfigValidationError,
    get_default_config,
    load_config,
    load_or_create_config,
    save_config,
    validate_config,
)
from aylm.tools.video_types import (
    FrameExtractionMethod,
    GPUAcceleration,
    VideoConfig,
)


class TestDefaultConfig:
    """Test default configuration."""

    def test_get_default_config_returns_video_config(self) -> None:
        config = get_default_config()
        assert isinstance(config, VideoConfig)

    def test_default_config_has_expected_values(self) -> None:
        config = get_default_config()
        assert config.frame_extraction_method == FrameExtractionMethod.INTERVAL
        assert config.gpu_acceleration == GPUAcceleration.AUTO
        assert config.output_format == "png"
        assert config.quality == 95
        assert config.device == "auto"
        assert config.verbose is True

    def test_default_config_dict_matches_video_config(self) -> None:
        assert DEFAULT_CONFIG["frame_extraction_method"] == "interval"
        assert DEFAULT_CONFIG["gpu_acceleration"] == "auto"
        assert DEFAULT_CONFIG["quality"] == 95


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_valid_config(self) -> None:
        config = {
            "frame_extraction_method": "uniform",
            "gpu_acceleration": "auto",
            "quality": 95,
            "scene_threshold": 0.5,
        }
        validate_config(config)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("frame_extraction_method", "invalid_method"),
            ("gpu_acceleration", "invalid_gpu"),
            ("quality", 150),
            ("quality", 0),
            ("target_fps", -1),
            ("max_frames", -10),
            ("scene_threshold", 1.5),
            ("output_format", "gif"),
            ("device", "tpu"),
        ],
    )
    def test_validate_invalid_values(self, field: str, value: object) -> None:
        config = {field: value}
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert field in str(exc_info.value)

    def test_validate_none_values_allowed(self) -> None:
        config = {
            "target_fps": None,
            "max_frames": None,
            "resize_width": None,
            "resize_height": None,
        }
        validate_config(config)


class TestLoadConfig:
    """Test configuration loading from YAML files."""

    def test_load_config_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {"frame_extraction_method": "keyframe", "quality": 80, "device": "cpu"},
                f,
            )
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            assert config.frame_extraction_method == FrameExtractionMethod.KEYFRAME
            assert config.quality == 80
            assert config.device == "cpu"
            assert config.gpu_acceleration == GPUAcceleration.AUTO
        finally:
            config_path.unlink()

    def test_load_config_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_config_invalid_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            config_path.unlink()

    def test_load_config_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            default_config = get_default_config()
            assert config.quality == default_config.quality
            assert config.device == default_config.device
        finally:
            config_path.unlink()


class TestSaveConfig:
    """Test configuration saving to YAML files."""

    def test_save_config_to_file(self) -> None:
        config = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            save_config(config, config_path)

            assert config_path.exists()
            with open(config_path) as f:
                saved_data = yaml.safe_load(f)

            assert saved_data["frame_extraction_method"] == "interval"
            assert saved_data["quality"] == 95

    def test_save_and_load_roundtrip(self) -> None:
        original = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "roundtrip.yaml"
            save_config(original, config_path)
            loaded = load_config(config_path)

            assert loaded.frame_extraction_method == original.frame_extraction_method
            assert loaded.quality == original.quality
            assert loaded.device == original.device
            assert loaded.gpu_acceleration == original.gpu_acceleration

    def test_save_config_creates_parent_dirs(self) -> None:
        config = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nested" / "dir" / "config.yaml"
            save_config(config, config_path)
            assert config_path.exists()


class TestLoadOrCreateConfig:
    """Test load_or_create_config function."""

    def test_load_or_create_with_none_path(self) -> None:
        config = load_or_create_config(None)
        default = get_default_config()
        assert config.quality == default.quality

    def test_load_or_create_with_nonexistent_path(self) -> None:
        config = load_or_create_config(Path("/nonexistent/config.yaml"))
        default = get_default_config()
        assert config.quality == default.quality

    def test_load_or_create_with_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"quality": 50}, f)
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_or_create_config(config_path)
            assert config.quality == 50
        finally:
            config_path.unlink()


class TestParameterizedValidation:
    """Test parameterized validation for enums."""

    @pytest.mark.parametrize(
        "method", ["uniform", "keyframe", "scene_change", "interval"]
    )
    def test_valid_frame_extraction_methods(self, method: str) -> None:
        validate_config({"frame_extraction_method": method})

    @pytest.mark.parametrize("accel", ["none", "cuda", "videotoolbox", "vaapi", "auto"])
    def test_valid_gpu_acceleration_types(self, accel: str) -> None:
        validate_config({"gpu_acceleration": accel})

    @pytest.mark.parametrize("fmt", ["png", "jpg", "jpeg", "webp", "bmp"])
    def test_valid_output_formats(self, fmt: str) -> None:
        validate_config({"output_format": fmt})

    @pytest.mark.parametrize("device", ["auto", "cuda", "mps", "cpu"])
    def test_valid_device_types(self, device: str) -> None:
        validate_config({"device": device})
