"""Artifact loading and obstacle summarization for the navigation demo."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class FrameArtifact:
    """A single extracted video frame plus its paired A-YLM outputs."""

    index: int
    stem: str
    frame_path: Path
    obstacle_path: Path | None
    timestamp: float
    obstacles: list[dict[str, Any]]
    obstacle_payload: dict[str, Any]
    pipeline_evaluation: dict[str, Any] | None


@dataclass
class ObstacleSummary:
    """Compact scene summary passed to the proposer and renderer."""

    obstacle_count: int
    tracked_count: int
    dynamic_count: int
    nearest_overall_m: float | None
    nearest_ahead_m: float | None
    nearest_left_m: float | None
    nearest_right_m: float | None
    scene_safety_score: float | None
    likely_blocked: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "obstacle_count": self.obstacle_count,
            "tracked_count": self.tracked_count,
            "dynamic_count": self.dynamic_count,
            "nearest_overall_m": self.nearest_overall_m,
            "nearest_ahead_m": self.nearest_ahead_m,
            "nearest_left_m": self.nearest_left_m,
            "nearest_right_m": self.nearest_right_m,
            "scene_safety_score": self.scene_safety_score,
            "likely_blocked": self.likely_blocked,
        }

    def to_prompt_text(self) -> str:
        """Format the summary for prompt construction."""
        return "\n".join(
            [
                f"- obstacles: {self.obstacle_count}",
                f"- tracked_obstacles: {self.tracked_count}",
                f"- dynamic_obstacles: {self.dynamic_count}",
                f"- nearest_overall_m: {_format_distance(self.nearest_overall_m)}",
                f"- nearest_ahead_m: {_format_distance(self.nearest_ahead_m)}",
                f"- nearest_left_m: {_format_distance(self.nearest_left_m)}",
                f"- nearest_right_m: {_format_distance(self.nearest_right_m)}",
                (
                    "- pipeline_scene_safety_score: "
                    f"{self.scene_safety_score:.2f}"
                    if self.scene_safety_score is not None
                    else "- pipeline_scene_safety_score: unknown"
                ),
                f"- likely_blocked: {'yes' if self.likely_blocked else 'no'}",
            ]
        )


class NavigationArtifacts:
    """Read `aylm video process` outputs in a stable, frame-aligned way."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.frames_dir = self.artifacts_dir / "extracted_frames"
        self.voxel_dir = self.artifacts_dir / "voxelized"

    def discover_frames(self, max_frames: int | None = None) -> list[FrameArtifact]:
        """Load extracted frames and pair them with obstacle JSON files."""
        if not self.frames_dir.exists():
            raise FileNotFoundError(
                f"Missing extracted frames directory: {self.frames_dir}"
            )

        frame_paths = sorted(
            path
            for path in self.frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS
        )
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]
        if not frame_paths:
            raise FileNotFoundError(f"No extracted frames found in: {self.frames_dir}")

        artifacts: list[FrameArtifact] = []
        for index, frame_path in enumerate(frame_paths):
            obstacle_path = self.voxel_dir / f"vox_{frame_path.stem}_obstacles.json"
            payload = self._load_obstacle_payload(obstacle_path, frame_path.stem)
            timestamp = float(payload.get("timestamp", index))
            artifacts.append(
                FrameArtifact(
                    index=index,
                    stem=frame_path.stem,
                    frame_path=frame_path,
                    obstacle_path=obstacle_path if obstacle_path.exists() else None,
                    timestamp=timestamp,
                    obstacles=list(payload.get("obstacles", [])),
                    obstacle_payload=payload,
                    pipeline_evaluation=payload.get("constitution_evaluation"),
                )
            )

        return artifacts

    @staticmethod
    def summarize_frame(
        frame: FrameArtifact,
        lane_half_width: float = 1.25,
    ) -> ObstacleSummary:
        """Create a compact scene summary from one obstacle payload."""
        obstacles = frame.obstacles
        overall_distances: list[float] = []
        ahead_distances: list[float] = []
        left_distances: list[float] = []
        right_distances: list[float] = []
        tracked_count = 0
        dynamic_count = 0

        for obstacle in obstacles:
            center = obstacle.get("center_robot") or obstacle.get("center")
            if not _is_valid_center(center):
                continue

            x_pos = float(center[0])
            y_pos = float(center[1])
            z_pos = float(center[2])
            distance = math.sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos)
            overall_distances.append(distance)

            if obstacle.get("track_id") is not None:
                tracked_count += 1

            motion = obstacle.get("motion", {})
            if isinstance(motion, dict) and motion and not motion.get(
                "is_stationary", True
            ):
                dynamic_count += 1

            if x_pos > 0 and abs(y_pos) <= lane_half_width:
                ahead_distances.append(distance)
            if x_pos > -0.5 and y_pos > 0.5:
                left_distances.append(distance)
            if x_pos > -0.5 and y_pos < -0.5:
                right_distances.append(distance)

        pipeline_score = None
        evaluation = frame.pipeline_evaluation or {}
        if isinstance(evaluation, dict):
            safety = evaluation.get("safety_score", {})
            if isinstance(safety, dict) and isinstance(
                safety.get("overall"), (int, float)
            ):
                pipeline_score = float(safety["overall"])

        nearest_ahead = min(ahead_distances) if ahead_distances else None
        likely_blocked = bool(
            (nearest_ahead is not None and nearest_ahead < 1.5)
            or (dynamic_count > 0 and nearest_ahead is not None and nearest_ahead < 3.0)
        )

        return ObstacleSummary(
            obstacle_count=len(obstacles),
            tracked_count=tracked_count,
            dynamic_count=dynamic_count,
            nearest_overall_m=min(overall_distances) if overall_distances else None,
            nearest_ahead_m=nearest_ahead,
            nearest_left_m=min(left_distances) if left_distances else None,
            nearest_right_m=min(right_distances) if right_distances else None,
            scene_safety_score=pipeline_score,
            likely_blocked=likely_blocked,
        )

    @staticmethod
    def _load_obstacle_payload(
        obstacle_path: Path,
        frame_stem: str,
    ) -> dict[str, Any]:
        if not obstacle_path.exists():
            logger.debug("Missing obstacle JSON for frame %s", frame_stem)
            return {
                "frame_stem": frame_stem,
                "obstacles": [],
            }

        with open(obstacle_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if "obstacles" not in payload:
            payload["obstacles"] = []
        payload.setdefault("frame_stem", frame_stem)
        return payload


def _format_distance(value: float | None) -> str:
    """Human-readable distance formatting."""
    if value is None:
        return "clear"
    return f"{value:.2f}"


def _is_valid_center(center: Any) -> bool:
    """Return True when the payload contains a 3D center point."""
    return isinstance(center, (list, tuple)) and len(center) >= 3
