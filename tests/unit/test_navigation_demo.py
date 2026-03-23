"""Tests for the offline navigation demo package."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from aylm.cli import create_parser
from aylm.navigation_demo.artifacts import NavigationArtifacts
from aylm.navigation_demo.config import NavigationDemoConfig
from aylm.navigation_demo.providers import (
    HeuristicCommandProposer,
    normalize_command,
)
from aylm.navigation_demo.runner import NavigationDemoRunner


def test_navigation_artifacts_pairs_frames_and_json(tmp_path: Path) -> None:
    artifacts_dir = _create_artifacts_root(tmp_path)
    _write_frame(artifacts_dir / "extracted_frames" / "frame_000001.jpg", color=32)
    _write_frame(artifacts_dir / "extracted_frames" / "frame_000002.jpg", color=64)
    _write_obstacles(
        artifacts_dir / "voxelized" / "vox_frame_000001_obstacles.json",
        timestamp=0.5,
        obstacles=[
            {
                "center_robot": [2.0, 0.0, 0.0],
                "dimensions_robot": [1.0, 1.0, 1.0],
                "_label": "BOX",
                "confidence": 0.9,
            }
        ],
    )

    frames = NavigationArtifacts(artifacts_dir).discover_frames()

    assert len(frames) == 2
    assert frames[0].timestamp == 0.5
    assert len(frames[0].obstacles) == 1
    assert frames[1].obstacles == []
    assert frames[1].obstacle_path is None


def test_normalize_command_accepts_validator_alternative_trajectory() -> None:
    payload = {
        "decision_type": "trajectory",
        "trajectory": [
            {"position": [0.0, 0.0, 0.0], "timestamp": 0.0},
            {"position": [1.0, 0.2, 0.0], "timestamp": 1.0},
        ],
        "target_speed": 0.3,
        "metadata": {"source": "safety_alternative"},
    }

    normalized = normalize_command(payload, default_target_speed=1.0)

    assert normalized["type"] == "trajectory"
    assert len(normalized["points"]) == 2
    assert normalized["target_speed"] == 0.3


def test_heuristic_proposer_stops_for_close_obstacle(tmp_path: Path) -> None:
    proposer = HeuristicCommandProposer()
    artifacts_dir = _create_artifacts_root(tmp_path)
    frame = _make_frame_artifact(
        artifacts_dir=artifacts_dir,
        stem="frame_000100",
        timestamp=0.0,
        obstacles=[
            {
                "center_robot": [0.8, 0.0, 0.0],
                "dimensions_robot": [0.4, 0.4, 0.4],
                "_label": "PERSON",
                "confidence": 0.95,
            }
        ],
    )
    summary = NavigationArtifacts(artifacts_dir).summarize_frame(frame)

    proposal = proposer.propose(
        frame_paths=[frame.frame_path],
        summary=summary,
        current_speed=1.0,
        previous_command=None,
    )

    assert proposal.command["type"] == "control"
    assert proposal.command["brake"] == 1.0
    assert proposal.command["target_speed"] == 0.0


def test_nav_demo_cli_parser_wires_expected_arguments() -> None:
    parser = create_parser()

    args = parser.parse_args(
        [
            "nav-demo",
            "--input",
            "outputs/video_demo",
            "--provider",
            "heuristic",
            "--window-size",
            "3",
            "--no-render",
        ]
    )

    assert args.command == "nav-demo"
    assert args.provider == "heuristic"
    assert args.window_size == 3
    assert args.no_render is True
    assert args.func.__name__ == "cmd_nav_demo"


def test_navigation_demo_runner_emits_logs_without_render(tmp_path: Path) -> None:
    artifacts_dir = _create_artifacts_root(tmp_path)
    frame_stems = [
        "frame_000001",
        "frame_000002",
        "frame_000003",
        "frame_000004",
    ]
    for index, stem in enumerate(frame_stems, start=1):
        _write_frame(
            artifacts_dir / "extracted_frames" / f"{stem}.jpg",
            color=index * 20,
        )

    _write_obstacles(
        artifacts_dir / "voxelized" / "vox_frame_000001_obstacles.json",
        timestamp=0.0,
        obstacles=[],
    )
    _write_obstacles(
        artifacts_dir / "voxelized" / "vox_frame_000002_obstacles.json",
        timestamp=1.0,
        obstacles=[
            {
                "center_robot": [3.0, 0.2, 0.0],
                "dimensions_robot": [1.2, 0.8, 1.0],
                "_label": "VEHICLE",
                "confidence": 0.9,
            }
        ],
    )
    _write_obstacles(
        artifacts_dir / "voxelized" / "vox_frame_000003_obstacles.json",
        timestamp=2.0,
        obstacles=[
            {
                "center_robot": [0.9, 0.1, 0.0],
                "dimensions_robot": [0.5, 0.5, 1.7],
                "_label": "PERSON",
                "confidence": 0.95,
            }
        ],
    )
    _write_obstacles(
        artifacts_dir / "voxelized" / "vox_frame_000004_obstacles.json",
        timestamp=3.0,
        obstacles=[],
    )

    config = NavigationDemoConfig(
        artifacts_dir=artifacts_dir,
        output_dir=tmp_path / "nav_demo_output",
        provider="heuristic",
        window_size=2,
        window_stride=1,
        ego_speed=1.0,
        render_video=False,
    )
    summary = NavigationDemoRunner(config).run()

    command_log = config.output_dir / config.command_log_name
    summary_path = config.output_dir / config.summary_file_name
    resolved_config = config.output_dir / "resolved_config.json"

    assert summary["decision_count"] == 3
    assert summary["rendered_video"] is None
    assert command_log.exists()
    assert summary_path.exists()
    assert resolved_config.exists()

    with open(command_log, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert len(records) == 3
    assert any(record["proposed_command"]["brake"] == 1.0 for record in records)
    assert records[-1]["scene_summary"]["obstacle_count"] == 0


def _create_artifacts_root(base_dir: Path) -> Path:
    artifacts_dir = base_dir / "video_demo"
    (artifacts_dir / "extracted_frames").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "voxelized").mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def _write_frame(path: Path, color: int) -> None:
    image = np.full((64, 96, 3), color, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write test frame: {path}")


def _write_obstacles(
    path: Path,
    *,
    timestamp: float,
    obstacles: list[dict],
) -> None:
    payload = {
        "timestamp": timestamp,
        "obstacles": obstacles,
        "constitution_evaluation": {
            "safety_score": {
                "overall": 0.9,
            }
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def _make_frame_artifact(
    *,
    artifacts_dir: Path,
    stem: str,
    timestamp: float,
    obstacles: list[dict],
):
    _write_frame(artifacts_dir / "extracted_frames" / f"{stem}.jpg", color=48)
    _write_obstacles(
        artifacts_dir / "voxelized" / f"vox_{stem}_obstacles.json",
        timestamp=timestamp,
        obstacles=obstacles,
    )
    return NavigationArtifacts(artifacts_dir).discover_frames()[0]
