"""Configuration for the offline navigation demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MLX_MODEL = "mlx-community/SmolVLM2-500M-Video-Instruct-mlx"


@dataclass
class NavigationDemoConfig:
    """Runtime configuration for the offline navigation demo."""

    artifacts_dir: Path
    output_dir: Path
    provider: str = "heuristic"
    model_name: str = DEFAULT_MLX_MODEL
    window_size: int = 4
    window_stride: int = 2
    max_frames: int | None = None
    approval_threshold: float = 0.6
    ego_speed: float = 1.0
    ego_heading: float = 0.0
    overlay_fps: float = 4.0
    max_tokens: int = 256
    temperature: float = 0.0
    prompt_file: Path | None = None
    output_video_name: str = "navigation_demo.mp4"
    summary_file_name: str = "run_summary.json"
    command_log_name: str = "commands.jsonl"
    render_video: bool = True

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir)
        self.output_dir = Path(self.output_dir)

        if self.provider not in {"mlx-vlm", "heuristic"}:
            raise ValueError(
                f"Unsupported provider '{self.provider}'. "
                "Expected one of: mlx-vlm, heuristic."
            )
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.window_stride < 1:
            raise ValueError("window_stride must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1 when provided")
        if self.approval_threshold < 0.0 or self.approval_threshold > 1.0:
            raise ValueError("approval_threshold must be between 0.0 and 1.0")
        if self.overlay_fps <= 0:
            raise ValueError("overlay_fps must be > 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.prompt_file is not None:
            self.prompt_file = Path(self.prompt_file)
