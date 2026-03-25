"""Command proposal providers for the offline navigation demo."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ObstacleSummary

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """You are a cautious low-speed navigation policy for a robot or small vehicle.
Use the image sequence and structured scene summary to propose one short-horizon navigation command.
Output exactly one JSON object and nothing else.

Allowed schemas:
{"type":"control","steering":0.0,"throttle":0.0,"brake":0.0,"target_speed":0.0,"reason":"..."}
{"type":"waypoint","target":[x,y,z],"speed":0.0,"reason":"..."}
{"type":"trajectory","points":[[x,y,z,t], ...],"target_speed":0.0,"reason":"..."}

Rules:
- Prefer conservative commands.
- Keep steering gentle unless the path ahead is blocked.
- If the scene is uncertain or blocked, brake or stop.
- Never output markdown.
"""


@dataclass
class CommandProposal:
    """Normalized command plus provider metadata."""

    command: dict[str, Any]
    raw_text: str
    provider: str
    prompt: str


class BaseCommandProposer:
    """Interface for proposal providers."""

    provider_name = "base"

    def __init__(self, prompt_prefix: str | None = None):
        self.prompt_prefix = prompt_prefix.strip() if prompt_prefix else DEFAULT_PROMPT

    def propose(
        self,
        *,
        frame_paths: list[Path],
        summary: ObstacleSummary,
        current_speed: float,
        previous_command: dict[str, Any] | None,
        prompt_suffix: str | None = None,
    ) -> CommandProposal:
        raise NotImplementedError

    def build_prompt(
        self,
        *,
        frame_paths: list[Path],
        summary: ObstacleSummary,
        current_speed: float,
        previous_command: dict[str, Any] | None,
    ) -> str:
        """Build the text prompt shared by MLX-VLM-like providers."""
        previous_line = (
            json.dumps(previous_command, ensure_ascii=False)
            if previous_command
            else "none"
        )
        frame_labels = ", ".join(path.stem for path in frame_paths)
        return "\n".join(
            [
                self.prompt_prefix,
                f"Current ego speed: {current_speed:.2f} m/s",
                f"Frames (oldest to newest): {frame_labels}",
                f"Previous executed command: {previous_line}",
                "Scene summary:",
                summary.to_prompt_text(),
            ]
        )


class HeuristicCommandProposer(BaseCommandProposer):
    """Dependency-free fallback used for tests and safe local bring-up."""

    provider_name = "heuristic"

    def propose(
        self,
        *,
        frame_paths: list[Path],
        summary: ObstacleSummary,
        current_speed: float,
        previous_command: dict[str, Any] | None,
        prompt_suffix: str | None = None,
    ) -> CommandProposal:
        del frame_paths, previous_command

        if summary.nearest_ahead_m is not None and summary.nearest_ahead_m < 1.2:
            command = safe_stop_command("obstacle too close ahead")
        elif summary.likely_blocked:
            steer_left = _clearance(summary.nearest_left_m)
            steer_right = _clearance(summary.nearest_right_m)
            steering = 0.18 if steer_left >= steer_right else -0.18
            command = {
                "type": "control",
                "steering": steering,
                "throttle": 0.05,
                "brake": 0.25,
                "target_speed": min(max(current_speed * 0.5, 0.2), 0.8),
                "reason": "slow avoidance around a blocked path",
            }
        else:
            command = {
                "type": "control",
                "steering": 0.0,
                "throttle": 0.25,
                "brake": 0.0,
                "target_speed": min(max(current_speed + 0.2, 0.4), 1.2),
                "reason": "path ahead appears clear",
            }

        return CommandProposal(
            command=normalize_command(command, default_target_speed=current_speed),
            raw_text=json.dumps(command, ensure_ascii=False),
            provider=self.provider_name,
            prompt=f"heuristic\n{prompt_suffix}" if prompt_suffix else "heuristic",
        )


class MlxVlmCommandProposer(BaseCommandProposer):
    """MLX-VLM proposer that treats a short frame window as multi-image input."""

    provider_name = "mlx-vlm"

    def __init__(
        self,
        *,
        model_name: str,
        max_tokens: int,
        temperature: float,
        prompt_prefix: str | None = None,
    ):
        super().__init__(prompt_prefix=prompt_prefix)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model: Any = None
        self._processor: Any = None
        self._config: Any = None
        self._generate_fn = None
        self._apply_chat_template = None

    def propose(
        self,
        *,
        frame_paths: list[Path],
        summary: ObstacleSummary,
        current_speed: float,
        previous_command: dict[str, Any] | None,
        prompt_suffix: str | None = None,
    ) -> CommandProposal:
        self._ensure_loaded()

        from PIL import Image

        prompt = self.build_prompt(
            frame_paths=frame_paths,
            summary=summary,
            current_speed=current_speed,
            previous_command=previous_command,
        )
        if prompt_suffix:
            prompt = prompt + "\n\n" + prompt_suffix
        images = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                images.append(image.convert("RGB"))

        formatted_prompt = self._apply_chat_template(
            self._processor,
            self._config,
            prompt,
            num_images=len(images),
        )

        try:
            raw_output = self._generate_fn(
                self._model,
                self._processor,
                formatted_prompt,
                images,
                verbose=False,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except TypeError:
            # Older mlx-vlm releases accept a smaller argument set.
            raw_output = self._generate_fn(
                self._model,
                self._processor,
                formatted_prompt,
                images,
                verbose=False,
            )

        command = extract_and_normalize_command(
            raw_output,
            default_target_speed=current_speed,
        )
        return CommandProposal(
            command=command,
            raw_text=str(raw_output),
            provider=self.provider_name,
            prompt=prompt,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:
            raise ImportError(
                "mlx-vlm is required for provider 'mlx-vlm'. "
                "Install it with: pip install -U mlx-vlm"
            ) from exc

        self._model, self._processor = load(self.model_name)
        self._config = getattr(self._model, "config", None)
        if self._config is None:
            try:
                from mlx_vlm.utils import load_config
            except ImportError:
                load_config = None
            if load_config is not None:
                self._config = load_config(self.model_name)

        self._generate_fn = generate
        self._apply_chat_template = apply_chat_template


def build_proposer(
    *,
    provider: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    prompt_prefix: str | None,
) -> BaseCommandProposer:
    """Construct a proposer from CLI/runtime configuration."""
    if provider == "heuristic":
        return HeuristicCommandProposer(prompt_prefix=prompt_prefix)
    if provider == "mlx-vlm":
        return MlxVlmCommandProposer(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_prefix=prompt_prefix,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def extract_and_normalize_command(
    raw_output: Any,
    *,
    default_target_speed: float,
) -> dict[str, Any]:
    """Extract the first JSON command from a model response and normalize it."""
    text = str(raw_output).strip()
    payload = _extract_json_payload(text)
    if payload is None:
        logger.warning("Could not parse JSON from provider output, using text fallback")
        return normalize_command(text, default_target_speed=default_target_speed)
    return normalize_command(payload, default_target_speed=default_target_speed)


def normalize_command(
    payload: Any,
    *,
    default_target_speed: float,
) -> dict[str, Any]:
    """Normalize provider output to one of the command formats A-YLM accepts."""
    if isinstance(payload, str):
        return _command_from_text(payload, default_target_speed=default_target_speed)

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported command payload type: {type(payload).__name__}")

    if "command" in payload and isinstance(payload["command"], dict):
        payload = payload["command"]

    normalized = dict(payload)
    if "decision_type" in normalized and "type" not in normalized:
        normalized["type"] = normalized["decision_type"]

    command_type = normalized.get("type")
    if command_type is None:
        if {"steering", "throttle", "brake"} & normalized.keys():
            command_type = "control"
        elif "target" in normalized:
            command_type = "waypoint"
        elif "trajectory" in normalized or "points" in normalized:
            command_type = "trajectory"
        elif "action" in normalized:
            return _command_from_text(
                str(normalized["action"]),
                default_target_speed=default_target_speed,
            )
        else:
            raise ValueError("Command payload missing 'type'")

    if command_type == "control":
        return _normalize_control(normalized, default_target_speed)
    if command_type == "waypoint":
        return _normalize_waypoint(normalized, default_target_speed)
    if command_type == "trajectory":
        return _normalize_trajectory(normalized, default_target_speed)

    raise ValueError(f"Unsupported command type: {command_type}")


def safe_stop_command(reason: str) -> dict[str, Any]:
    """Conservative fallback used when proposal parsing or approval fails."""
    return {
        "type": "control",
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
        "target_speed": 0.0,
        "reason": reason,
    }


def format_command(command: dict[str, Any]) -> str:
    """Return a short single-line representation of a command."""
    command_type = command.get("type") or command.get("decision_type")
    if command_type == "control":
        return (
            "control "
            f"s={float(command.get('steering', 0.0)):+.2f} "
            f"t={float(command.get('throttle', 0.0)):.2f} "
            f"b={float(command.get('brake', 0.0)):.2f}"
        )
    if command_type == "waypoint":
        target = command.get("target", [0.0, 0.0, 0.0])
        return f"waypoint x={target[0]:.2f} y={target[1]:.2f} z={target[2]:.2f}"
    if command_type == "trajectory":
        points = command.get("points", command.get("trajectory", []))
        return f"trajectory points={len(points)}"
    return str(command_type)


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    while start != -1:
        brace_level = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                brace_level += 1
            elif char == "}":
                brace_level -= 1
                if brace_level == 0:
                    candidate = text[start : index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _normalize_control(
    payload: dict[str, Any], default_target_speed: float
) -> dict[str, Any]:
    steering = _clamp(
        _to_float(payload.get("steering", payload.get("steer", 0.0))),
        -0.6,
        0.6,
    )
    throttle = _clamp(_to_float(payload.get("throttle", 0.0)), 0.0, 1.0)
    brake = _clamp(_to_float(payload.get("brake", 0.0)), 0.0, 1.0)
    target_speed = payload.get("target_speed")
    if target_speed is None:
        target_speed = max(default_target_speed + (throttle - brake) * 0.5, 0.0)

    return {
        "type": "control",
        "steering": steering,
        "throttle": throttle,
        "brake": brake,
        "target_speed": round(_to_float(target_speed), 3),
        "reason": str(payload.get("reason", "")).strip(),
    }


def _normalize_waypoint(
    payload: dict[str, Any],
    default_target_speed: float,
) -> dict[str, Any]:
    target = payload.get("target") or payload.get("position") or [0.0, 0.0, 0.0]
    if not isinstance(target, (list, tuple)) or len(target) < 3:
        raise ValueError("Waypoint command requires target=[x,y,z]")

    return {
        "type": "waypoint",
        "target": [round(_to_float(value), 3) for value in target[:3]],
        "speed": round(_to_float(payload.get("speed", default_target_speed)), 3),
        "reason": str(payload.get("reason", "")).strip(),
    }


def _normalize_trajectory(
    payload: dict[str, Any],
    default_target_speed: float,
) -> dict[str, Any]:
    raw_points = payload.get("points", payload.get("trajectory", []))
    points: list[list[float]] = []
    for raw_point in raw_points:
        if isinstance(raw_point, dict):
            position = raw_point.get("position", [0.0, 0.0, 0.0])
            timestamp = raw_point.get("timestamp", 0.0)
            point = [*_to_xyz(position), round(_to_float(timestamp), 3)]
        elif isinstance(raw_point, (list, tuple)) and len(raw_point) >= 3:
            point = [round(_to_float(value), 3) for value in raw_point[:4]]
            if len(point) == 3:
                point.append(0.0)
        else:
            continue
        points.append(point)

    if not points:
        return {
            "type": "trajectory",
            "points": [[0.0, 0.0, 0.0, 0.0]],
            "target_speed": round(default_target_speed, 3),
            "reason": str(payload.get("reason", "")).strip(),
        }

    return {
        "type": "trajectory",
        "points": points,
        "target_speed": round(
            _to_float(payload.get("target_speed", default_target_speed)),
            3,
        ),
        "reason": str(payload.get("reason", "")).strip(),
    }


def _command_from_text(text: str, default_target_speed: float) -> dict[str, Any]:
    normalized = text.strip().lower()
    if any(token in normalized for token in {"stop", "brake", "停车", "刹车"}):
        return safe_stop_command("provider requested stop")
    if "left" in normalized or "左" in normalized:
        return {
            "type": "control",
            "steering": 0.18,
            "throttle": 0.12,
            "brake": 0.0,
            "target_speed": round(max(default_target_speed, 0.4), 3),
            "reason": "text fallback steer left",
        }
    if "right" in normalized or "右" in normalized:
        return {
            "type": "control",
            "steering": -0.18,
            "throttle": 0.12,
            "brake": 0.0,
            "target_speed": round(max(default_target_speed, 0.4), 3),
            "reason": "text fallback steer right",
        }
    return {
        "type": "control",
        "steering": 0.0,
        "throttle": 0.15,
        "brake": 0.0,
        "target_speed": round(max(default_target_speed, 0.4), 3),
        "reason": "text fallback keep moving",
    }


def _clearance(distance: float | None) -> float:
    if distance is None:
        return 10.0
    return distance


def _to_float(value: Any) -> float:
    return float(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_xyz(position: Any) -> list[float]:
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        raise ValueError("Trajectory position must have at least three values")
    return [round(_to_float(value), 3) for value in position[:3]]
