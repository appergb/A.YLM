"""Video overlay renderer for navigation demo outputs."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import cv2

from .artifacts import FrameArtifact
from .providers import format_command


class OverlayRenderer:
    """Render approved/rejected actions back onto extracted video frames."""

    def render(
        self,
        *,
        frames: list[FrameArtifact],
        records: list[dict[str, Any]],
        output_path: Path,
        fps: float,
    ) -> Path:
        if not frames:
            raise ValueError("No frames available for rendering")

        first_frame = cv2.imread(str(frames[0].frame_path))
        if first_frame is None:
            raise RuntimeError(f"Failed to read frame: {frames[0].frame_path}")
        height, width = first_frame.shape[:2]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for: {output_path}")

        record_by_index = {record["anchor_index"]: record for record in records}
        active_record: dict[str, Any] | None = None

        try:
            for frame in frames:
                if frame.index in record_by_index:
                    active_record = record_by_index[frame.index]

                image = cv2.imread(str(frame.frame_path))
                if image is None:
                    raise RuntimeError(f"Failed to read frame: {frame.frame_path}")

                _draw_overlay(
                    image=image,
                    frame=frame,
                    record=active_record,
                    frame_count=len(frames),
                )
                writer.write(image)
        finally:
            writer.release()

        return output_path


def _draw_overlay(
    *,
    image,
    frame: FrameArtifact,
    record: dict[str, Any] | None,
    frame_count: int,
) -> None:
    overlay = image.copy()
    color = (20, 20, 20)
    if record:
        color = (28, 74, 40) if record["proposal_approved"] else (25, 40, 98)
    cv2.rectangle(overlay, (20, 20), (image.shape[1] - 20, 230), color, -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)

    lines = [
        "A-YLM Offline Navigation Demo",
        (
            f"Frame {frame.index + 1}/{frame_count} | {frame.stem} "
            f"| t={frame.timestamp:.2f}s"
        ),
    ]

    if record is None:
        lines.append("Decision: warming up frame window")
    else:
        lines.extend(
            [
                (
                    f"Proposal: {format_command(record['proposed_command'])} "
                    f"| provider={record['provider']}"
                ),
                (
                    f"Executed: {format_command(record['executed_command'])} "
                    f"| proposal="
                    f"{'approved' if record['proposal_approved'] else 'rejected'} "
                    f"| executed_score={record['executed_score']:.2f} "
                    f"| trend={record['executed_trend']}"
                ),
                (
                    "Scene: "
                    f"ahead="
                    f"{_format_distance(record['scene_summary']['nearest_ahead_m'])}m "
                    f"left="
                    f"{_format_distance(record['scene_summary']['nearest_left_m'])}m "
                    f"right="
                    f"{_format_distance(record['scene_summary']['nearest_right_m'])}m "
                    f"dynamic={record['scene_summary']['dynamic_count']} "
                    f"count={record['scene_summary']['obstacle_count']}"
                ),
            ]
        )
        if record["fallback_source"]:
            lines.append(f"Fallback: {record['fallback_source']}")
        reason = record.get("proposal_reason") or ""
        if reason:
            lines.extend(textwrap.wrap(f"Reason: {reason}", width=75)[:2])

    y_pos = 52
    for line in lines:
        cv2.putText(
            image,
            line,
            (36, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        y_pos += 28


def _format_distance(value: float | None) -> str:
    if value is None:
        return "clear"
    return f"{value:.2f}"
