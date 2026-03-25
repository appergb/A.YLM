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
from .calibrator import FrameFeedback, SessionCalibrator
from .config import NavigationDemoConfig
from .learning_store import LearningStore
from .overlay import OverlayRenderer
from .prompt_calibrator import PromptCalibrator
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

        # 初始化自校准器
        calibrator, prompt_calibrator = self._init_calibration(validator)

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

            # 获取校准上下文并构建提示词后缀
            prompt_suffix = _build_prompt_suffix(calibrator, prompt_calibrator)

            proposal = proposer.propose(
                frame_paths=[item.frame_path for item in window],
                summary=summary,
                current_speed=session.ego_speed,
                previous_command=previous_command,
                prompt_suffix=prompt_suffix,
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

            # 向校准器反馈本帧结果
            _feed_calibrator(
                calibrator=calibrator,
                anchor_index=anchor_index,
                anchor_frame=anchor_frame,
                proposal_eval=proposal_eval,
                executed_eval=executed_eval,
            )

            # 如果校准器调整了阈值，同步到验证器
            if calibrator is not None:
                cal_ctx = calibrator.get_context()
                if cal_ctx.adjusted_threshold != validator.approval_threshold:
                    validator.approval_threshold = cal_ctx.adjusted_threshold

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

        # 会话结束：最终校准并持久化
        calibration_summary = _finalize_calibration(calibrator)

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

        run_summary: dict[str, Any] = {
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
        if calibration_summary:
            run_summary["calibration"] = calibration_summary

        self._write_json(
            self.config.output_dir / self.config.summary_file_name, run_summary
        )
        self._write_json(
            self.config.output_dir / "resolved_config.json",
            _config_to_dict(self.config),
        )

        return run_summary

    def _init_calibration(
        self,
        validator: CommandValidator,
    ) -> tuple[SessionCalibrator | None, PromptCalibrator | None]:
        """初始化校准器。如果禁用校准则返回 (None, None)。"""
        if not self.config.enable_calibration:
            return None, None

        learning_store = None
        if self.config.learning_store_path:
            learning_store = LearningStore(self.config.learning_store_path)

        calibrator = SessionCalibrator(
            base_threshold=self.config.approval_threshold,
            calibration_interval=self.config.calibration_interval,
            violation_count_trigger=self.config.violation_count_trigger,
            learning_store=learning_store,
        )

        # 加载跨会话基线
        if learning_store is not None:
            baseline = learning_store.load_baseline()
            if baseline is not None:
                calibrator.apply_baseline(baseline)
                validator.approval_threshold = baseline.threshold
                logger.info(
                    "已从学习存储加载基线: threshold=%.2f",
                    baseline.threshold,
                )

        prompt_cal = PromptCalibrator()
        return calibrator, prompt_cal

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


def _build_prompt_suffix(
    calibrator: SessionCalibrator | None,
    prompt_calibrator: PromptCalibrator | None,
) -> str | None:
    """从校准器获取当前上下文并构建提示词后缀。"""
    if calibrator is None or prompt_calibrator is None:
        return None
    context = calibrator.get_context()
    suffix = prompt_calibrator.build_prompt_suffix(context)
    return suffix if suffix else None


def _feed_calibrator(
    *,
    calibrator: SessionCalibrator | None,
    anchor_index: int,
    anchor_frame: Any,
    proposal_eval: dict[str, Any],
    executed_eval: dict[str, Any],
) -> None:
    """将本帧评估结果反馈给校准器。"""
    if calibrator is None:
        return

    # 提取违规原则名
    violations = [
        v.get("principle", v.get("name", "unknown"))
        for v in proposal_eval.get("violations", [])
    ]

    # 提取训练信号中的纠正提示
    correction_hints: list[str] = []
    training_signal = proposal_eval.get("evaluation_detail", {}).get("training_signal")
    if isinstance(training_signal, dict):
        signal_type = training_signal.get("signal_type")
        target = training_signal.get("correction_target")
        if signal_type == "correction" and isinstance(target, dict):
            for hint in target.get("corrections", []):
                if isinstance(hint, str):
                    correction_hints.append(hint)

    feedback = FrameFeedback(
        frame_index=anchor_index,
        timestamp=float(anchor_frame.timestamp),
        proposed_approved=bool(proposal_eval.get("approved", False)),
        proposal_score=float(proposal_eval.get("safety_score", 0.0)),
        executed_score=float(executed_eval.get("safety_score", 0.0)),
        violations=violations,
        signal_type=(
            training_signal.get("signal_type")
            if isinstance(training_signal, dict)
            else None
        ),
        correction_hints=correction_hints,
    )
    calibrator.record_frame(feedback)


def _finalize_calibration(
    calibrator: SessionCalibrator | None,
) -> dict[str, Any] | None:
    """会话结束时最终校准并返回校准摘要。"""
    if calibrator is None:
        return None

    final_context = calibrator.finalize()
    return {
        "rounds": final_context.calibration_round,
        "final_threshold": final_context.adjusted_threshold,
        "final_weights": final_context.adjusted_weights,
        "active_hints": final_context.safety_hints,
        "violation_summary": final_context.violation_summary,
        "frames_analyzed": final_context.frame_count,
    }


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
        "enable_calibration": config.enable_calibration,
        "calibration_interval": config.calibration_interval,
        "violation_count_trigger": config.violation_count_trigger,
        "learning_store_path": (
            str(config.learning_store_path) if config.learning_store_path else None
        ),
    }
