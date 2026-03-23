"""Offline navigation demo runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aylm.api.session import ConstitutionSession
from aylm.constitution import CommandValidator
from aylm.tools.json_utils import numpy_safe_dump

from .artifacts import NavigationArtifacts
from .config import NavigationDemoConfig
from .overlay import OverlayRenderer
from .providers import build_proposer, normalize_command, safe_stop_command

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """One navigation decision anchored to a frame window."""

    anchor_index: int
    anchor_stem: str
    timestamp: float
    provider: str
    proposed_command: dict[str, Any]
    executed_command: dict[str, Any]
    proposal_evaluation: dict[str, Any]
    executed_evaluation: dict[str, Any]
    fallback_source: str | None
    scene_summary: dict[str, Any]
    raw_provider_output: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a JSON-serializable dictionary."""
        proposal_reason = self.proposed_command.get(
            "reason"
        ) or self.proposal_evaluation.get("reason", "")
        return {
            "anchor_index": self.anchor_index,
            "anchor_stem": self.anchor_stem,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "proposed_command": self.proposed_command,
            "executed_command": self.executed_command,
            "proposal_evaluation": self.proposal_evaluation,
            "executed_evaluation": self.executed_evaluation,
            "proposal_approved": bool(self.proposal_evaluation.get("approved", False)),
            "executed_score": float(self.executed_evaluation.get("safety_score", 0.0)),
            "executed_trend": self.executed_evaluation.get("trend", "unknown"),
            "proposal_reason": proposal_reason,
            "fallback_source": self.fallback_source,
            "scene_summary": self.scene_summary,
            "raw_provider_output": self.raw_provider_output,
            "prompt": self.prompt,
        }


class NavigationDemoRunner:
    """Run the offline video navigation demo from A-YLM video artifacts."""

    def __init__(self, config: NavigationDemoConfig):
        self.config = config
        self.artifacts = NavigationArtifacts(config.artifacts_dir)

    def run(self) -> dict[str, Any]:
        """Execute the offline navigation demo end to end."""
        frames = self.artifacts.discover_frames(max_frames=self.config.max_frames)
        if len(frames) < self.config.window_size:
            raise ValueError(
                f"Need at least {self.config.window_size} extracted frames, "
                f"found {len(frames)}"
            )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        prompt_prefix = self._load_prompt_prefix()
        proposer = build_proposer(
            provider=self.config.provider,
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            prompt_prefix=prompt_prefix,
        )
        validator = CommandValidator(approval_threshold=self.config.approval_threshold)
        session = ConstitutionSession(
            ego_speed=self.config.ego_speed,
            ego_heading=self.config.ego_heading,
            approval_threshold=self.config.approval_threshold,
        )

        decision_records: list[DecisionRecord] = []
        previous_command: dict[str, Any] | None = None
        for anchor_index in _decision_indices(
            total_frames=len(frames),
            window_size=self.config.window_size,
            window_stride=self.config.window_stride,
        ):
            window = frames[
                anchor_index - self.config.window_size + 1 : anchor_index + 1
            ]
            anchor_frame = frames[anchor_index]
            summary = self.artifacts.summarize_frame(anchor_frame)

            proposal = proposer.propose(
                frame_paths=[item.frame_path for item in window],
                summary=summary,
                current_speed=session.ego_speed,
                previous_command=previous_command,
            )

            proposal_eval = validator.validate(
                command=proposal.command,
                ego_speed=session.ego_speed,
                ego_heading=session.ego_heading,
                obstacles=anchor_frame.obstacles,
            ).to_dict()

            executed_command, fallback_source = _select_executed_command(
                proposed_command=proposal.command,
                proposal_evaluation=proposal_eval,
                current_speed=session.ego_speed,
            )
            executed_eval = session.evaluate(
                obstacles=anchor_frame.obstacles,
                command=executed_command,
                timestamp=anchor_frame.timestamp,
            )

            decision_records.append(
                DecisionRecord(
                    anchor_index=anchor_index,
                    anchor_stem=anchor_frame.stem,
                    timestamp=anchor_frame.timestamp,
                    provider=proposal.provider,
                    proposed_command=proposal.command,
                    executed_command=executed_command,
                    proposal_evaluation=proposal_eval,
                    executed_evaluation=executed_eval,
                    fallback_source=fallback_source,
                    scene_summary=summary.to_dict(),
                    raw_provider_output=proposal.raw_text,
                    prompt=proposal.prompt,
                )
            )
            previous_command = executed_command
            _update_session_speed(session, executed_command)

        decision_dicts = [record.to_dict() for record in decision_records]
        command_log_path = self.config.output_dir / self.config.command_log_name
        self._write_command_log(command_log_path, decision_dicts)

        output_video = None
        if self.config.render_video:
            output_video = OverlayRenderer().render(
                frames=frames,
                records=decision_dicts,
                output_path=self.config.output_dir / self.config.output_video_name,
                fps=self.config.overlay_fps,
            )

        summary = {
            "artifacts_dir": str(self.config.artifacts_dir),
            "output_dir": str(self.config.output_dir),
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "total_frames": len(frames),
            "decision_count": len(decision_dicts),
            "approved_proposal_count": sum(
                1 for record in decision_dicts if record["proposal_approved"]
            ),
            "fallback_count": sum(
                1 for record in decision_dicts if record["fallback_source"] is not None
            ),
            "avg_executed_score": (
                sum(record["executed_score"] for record in decision_dicts)
                / len(decision_dicts)
            ),
            "min_executed_score": min(
                record["executed_score"] for record in decision_dicts
            ),
            "final_trend": decision_dicts[-1]["executed_trend"],
            "rendered_video": str(output_video) if output_video else None,
            "command_log": str(command_log_path),
            "window_size": self.config.window_size,
            "window_stride": self.config.window_stride,
        }
        self._write_json(self.config.output_dir / self.config.summary_file_name, summary)
        self._write_json(
            self.config.output_dir / "resolved_config.json",
            _config_to_dict(self.config),
        )

        return summary

    @staticmethod
    def _write_command_log(path: Path, records: list[dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                numpy_safe_dump(record, handle, ensure_ascii=False)
                handle.write("\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            numpy_safe_dump(payload, handle, indent=2, ensure_ascii=False)

    def _load_prompt_prefix(self) -> str | None:
        if self.config.prompt_file is None:
            return None
        with open(self.config.prompt_file, encoding="utf-8") as handle:
            return handle.read().strip()


def _decision_indices(
    *,
    total_frames: int,
    window_size: int,
    window_stride: int,
) -> list[int]:
    indices = list(range(window_size - 1, total_frames, window_stride))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices


def _select_executed_command(
    *,
    proposed_command: dict[str, Any],
    proposal_evaluation: dict[str, Any],
    current_speed: float,
) -> tuple[dict[str, Any], str | None]:
    if proposal_evaluation.get("approved") is True:
        return proposed_command, None

    alternative = proposal_evaluation.get("alternative_decision")
    if isinstance(alternative, dict) and alternative:
        return (
            normalize_command(
                alternative,
                default_target_speed=current_speed,
            ),
            "validator_alternative",
        )

    return safe_stop_command("validator rejected command"), "safe_stop"


def _update_session_speed(
    session: ConstitutionSession,
    executed_command: dict[str, Any],
) -> None:
    target_speed = executed_command.get("target_speed")
    if isinstance(target_speed, (int, float)):
        session.update_ego(speed=float(target_speed))
        return

    if executed_command.get("type") == "waypoint":
        speed = executed_command.get("speed")
        if isinstance(speed, (int, float)):
            session.update_ego(speed=float(speed))


def _config_to_dict(config: NavigationDemoConfig) -> dict[str, Any]:
    return {
        "artifacts_dir": str(config.artifacts_dir),
        "output_dir": str(config.output_dir),
        "provider": config.provider,
        "model_name": config.model_name,
        "window_size": config.window_size,
        "window_stride": config.window_stride,
        "max_frames": config.max_frames,
        "approval_threshold": config.approval_threshold,
        "ego_speed": config.ego_speed,
        "ego_heading": config.ego_heading,
        "overlay_fps": config.overlay_fps,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "prompt_file": str(config.prompt_file) if config.prompt_file else None,
        "output_video_name": config.output_video_name,
        "summary_file_name": config.summary_file_name,
        "command_log_name": config.command_log_name,
        "render_video": config.render_video,
    }
