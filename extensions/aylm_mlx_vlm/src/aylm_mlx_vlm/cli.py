"""CLI for MLX-VLM training signal demo."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MLX-VLM navigation demo and export training signals."
    )
    parser.add_argument("-i", "--input", required=True, help="A-YLM video output dir")
    parser.add_argument("-o", "--output", help="Output directory for demo artifacts")
    parser.add_argument("--model", help="MLX-VLM model name or local path")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--ego-speed", type=float, default=1.0)
    parser.add_argument("--ego-heading", type=float, default=0.0)
    parser.add_argument("--overlay-fps", type=float, default=4.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-file")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--signals-out",
        help="Path to training_signals.jsonl (default: output/training_signals.jsonl)",
    )
    return parser.parse_args()


def _extract_training_signals(command_log: Path) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    with command_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            evaluation = record.get("proposal_evaluation", {})
            detail = evaluation.get("evaluation_detail", {})
            signal = detail.get("training_signal")
            if isinstance(signal, dict):
                signals.append(signal)
    return signals


def _write_signals(signals: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for signal in signals:
            handle.write(json.dumps(signal, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()

    from aylm.navigation_demo import NavigationDemoConfig, NavigationDemoRunner
    from aylm.navigation_demo.config import DEFAULT_MLX_MODEL

    artifacts_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else artifacts_dir.parent / "nav_demo"
    model_name = args.model or DEFAULT_MLX_MODEL

    config = NavigationDemoConfig(
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        provider="mlx-vlm",
        model_name=model_name,
        window_size=args.window_size,
        window_stride=args.window_stride,
        max_frames=args.max_frames,
        approval_threshold=args.threshold,
        ego_speed=args.ego_speed,
        ego_heading=args.ego_heading,
        overlay_fps=args.overlay_fps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        prompt_file=Path(args.prompt_file) if args.prompt_file else None,
        render_video=not args.no_render,
    )

    summary = NavigationDemoRunner(config).run()
    command_log = Path(summary["command_log"])

    signals = _extract_training_signals(command_log)
    signals_out = (
        Path(args.signals_out)
        if args.signals_out
        else output_dir / "training_signals.jsonl"
    )
    _write_signals(signals, signals_out)

    counts = Counter(signal.get("signal_type") for signal in signals)
    print("\nMLX-VLM training demo complete.")
    print(f"  Command log: {command_log}")
    print(f"  Training signals: {signals_out}")
    print(
        "  Signal counts: "
        + ", ".join(f"{k}:{v}" for k, v in counts.items() if k)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
