"""视频配置文件解析模块。

提供YAML配置文件的加载、验证和默认配置管理功能。
"""

import logging
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .video_types import FrameExtractionMethod, GPUAcceleration, VideoConfig

logger = logging.getLogger(__name__)


# 默认配置值
DEFAULT_CONFIG: dict[str, Any] = {
    "frame_extraction_method": "interval",
    "target_fps": None,
    "max_frames": None,
    "frame_interval": 2.0,
    "scene_threshold": 0.3,
    "gpu_acceleration": "auto",
    "hw_decode": True,
    "hw_encode": False,
    "output_format": "png",
    "quality": 95,
    "resize_width": None,
    "resize_height": None,
    "keep_aspect_ratio": True,
    "temp_dir": None,
    "cleanup_temp": True,
    "parallel_extraction": 4,
    "device": "auto",
    "verbose": True,
}


class ConfigValidationError(Exception):
    """配置验证错误。"""


def _validate_range(value: Any, min_val: float, max_val: float, name: str) -> None:
    """验证数值范围。"""
    if value is not None and not (min_val <= value <= max_val):
        raise ConfigValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )


def _validate_enum(value: str, enum_class: type[Enum], name: str) -> None:
    """验证枚举值。"""
    valid_values = [e.value for e in enum_class]
    if value not in valid_values:
        raise ConfigValidationError(
            f"{name} must be one of {valid_values}, got '{value}'"
        )


def _validate_positive(value: Any, name: str) -> None:
    """验证正数。"""
    if value is not None and value <= 0:
        raise ConfigValidationError(f"{name} must be positive, got {value}")


def validate_config(config: dict[str, Any]) -> None:
    """验证配置字典。

    Args:
        config: 配置字典

    Raises:
        ConfigValidationError: 配置验证失败
    """
    # 验证帧提取方法
    if "frame_extraction_method" in config:
        _validate_enum(
            config["frame_extraction_method"],
            FrameExtractionMethod,
            "frame_extraction_method",
        )

    # 验证GPU加速类型
    if "gpu_acceleration" in config:
        _validate_enum(
            config["gpu_acceleration"],
            GPUAcceleration,
            "gpu_acceleration",
        )

    # 验证数值范围
    _validate_positive(config.get("target_fps"), "target_fps")
    _validate_positive(config.get("max_frames"), "max_frames")
    _validate_positive(config.get("frame_interval"), "frame_interval")
    _validate_range(config.get("scene_threshold"), 0.0, 1.0, "scene_threshold")
    _validate_range(config.get("quality"), 1, 100, "quality")
    _validate_positive(config.get("resize_width"), "resize_width")
    _validate_positive(config.get("resize_height"), "resize_height")
    _validate_positive(config.get("parallel_extraction"), "parallel_extraction")

    # 验证输出格式
    valid_formats = ["png", "jpg", "jpeg", "webp", "bmp"]
    output_format = config.get("output_format", "png").lower()
    if output_format not in valid_formats:
        raise ConfigValidationError(
            f"output_format must be one of {valid_formats}, got '{output_format}'"
        )

    # 验证设备
    valid_devices = ["auto", "cuda", "mps", "cpu"]
    device = config.get("device", "auto").lower()
    if device not in valid_devices:
        raise ConfigValidationError(
            f"device must be one of {valid_devices}, got '{device}'"
        )


def load_config(config_path: Path) -> VideoConfig:
    """从YAML文件加载配置。

    Args:
        config_path: 配置文件路径

    Returns:
        VideoConfig: 视频配置对象

    Raises:
        FileNotFoundError: 配置文件不存在
        ConfigValidationError: 配置验证失败
        yaml.YAMLError: YAML解析错误
    """
    logger.info(f"Loading config from {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    # 合并默认配置
    merged_config = {**DEFAULT_CONFIG, **user_config}

    # 验证配置
    validate_config(merged_config)

    # 转换为VideoConfig对象
    return _dict_to_video_config(merged_config)


def _dict_to_video_config(config: dict[str, Any]) -> VideoConfig:
    """将配置字典转换为VideoConfig对象。"""
    # 转换枚举类型
    frame_method = FrameExtractionMethod(config["frame_extraction_method"])
    gpu_accel = GPUAcceleration(config["gpu_acceleration"])

    # 转换路径
    temp_dir = Path(config["temp_dir"]) if config["temp_dir"] else None

    return VideoConfig(
        frame_extraction_method=frame_method,
        target_fps=config["target_fps"],
        max_frames=config["max_frames"],
        frame_interval=config["frame_interval"],
        scene_threshold=config["scene_threshold"],
        gpu_acceleration=gpu_accel,
        hw_decode=config["hw_decode"],
        hw_encode=config["hw_encode"],
        output_format=config["output_format"],
        quality=config["quality"],
        resize_width=config["resize_width"],
        resize_height=config["resize_height"],
        keep_aspect_ratio=config["keep_aspect_ratio"],
        temp_dir=temp_dir,
        cleanup_temp=config["cleanup_temp"],
        parallel_extraction=config["parallel_extraction"],
        device=config["device"],
        verbose=config["verbose"],
    )


def get_default_config() -> VideoConfig:
    """获取默认配置。

    Returns:
        VideoConfig: 默认视频配置对象
    """
    return _dict_to_video_config(DEFAULT_CONFIG)


def save_config(config: VideoConfig, config_path: Path) -> None:
    """保存配置到YAML文件。

    Args:
        config: 视频配置对象
        config_path: 配置文件路径
    """
    logger.info(f"Saving config to {config_path}")

    config_dict = asdict(config)
    # 转换枚举和路径为可序列化格式
    config_dict["frame_extraction_method"] = config.frame_extraction_method.value
    config_dict["gpu_acceleration"] = config.gpu_acceleration.value
    config_dict["temp_dir"] = str(config.temp_dir) if config.temp_dir else None

    # 确保目录存在
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def load_or_create_config(config_path: Path | None = None) -> VideoConfig:
    """加载配置文件，如果不存在则使用默认配置。

    Args:
        config_path: 配置文件路径，None则使用默认配置

    Returns:
        VideoConfig: 视频配置对象
    """
    if config_path is None:
        logger.info("No config path provided, using default config")
        return get_default_config()

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using default config")
        return get_default_config()

    return load_config(config_path)
