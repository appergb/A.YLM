#!/bin/bash

set -euo pipefail

if [[ -n "${ZSH_VERSION:-}" ]]; then
    SCRIPT_PATH="${(%):-%N}"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_PATH=""
OUTPUT_VIDEO="outputs/video_demo"
OUTPUT_NAV="outputs/nav_demo"
EGO_SPEED="1.0"
EGO_HEADING="0.0"
USE_GPU="true"
WINDOW_SIZE="4"
WINDOW_STRIDE="2"
MODEL_NAME=""
MAX_FRAMES=""
PROMPT_FILE=""
NO_RENDER="false"
SKIP_INSTALL="false"
SKIP_VIDEO="false"
PROVIDER="mlx-vlm"
PROVIDER_SET="false"

usage() {
    cat << EOF
Usage: $(basename "$0") [--input <video_or_artifacts>] [options]

Options:
  --input, -i         Input video file OR artifacts directory
                      (default: first video found under inputs/)
  --output-video      Video pipeline output dir (default: outputs/video_demo)
  --output-nav        Nav demo output dir (default: outputs/nav_demo)
  --ego-speed         Ego speed in m/s (default: 1.0)
  --ego-heading       Ego heading in radians (default: 0.0)
  --no-gpu            Disable GPU for video processing
  --window-size       Nav demo window size (default: 4)
  --window-stride     Nav demo window stride (default: 2)
  --model             MLX-VLM model name/path
  --max-frames        Max frames for nav demo
  --prompt-file       Prompt file for MLX-VLM
  --no-render         Skip overlay rendering
  --skip-install      Skip pip install -e for core + extension
  --skip-video        Skip video processing even if input is a file
  --provider          Provider for nav demo (default: mlx-vlm)
  -h, --help          Show this help

Examples:
  scripts/run_mlx_demo.sh -i demo.mp4
  scripts/run_mlx_demo.sh -i outputs/video_demo --skip-video --provider heuristic
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            INPUT_PATH="$2"; shift 2;;
        --output-video)
            OUTPUT_VIDEO="$2"; shift 2;;
        --output-nav)
            OUTPUT_NAV="$2"; shift 2;;
        --ego-speed)
            EGO_SPEED="$2"; shift 2;;
        --ego-heading)
            EGO_HEADING="$2"; shift 2;;
        --no-gpu)
            USE_GPU="false"; shift;;
        --window-size)
            WINDOW_SIZE="$2"; shift 2;;
        --window-stride)
            WINDOW_STRIDE="$2"; shift 2;;
        --model)
            MODEL_NAME="$2"; shift 2;;
        --max-frames)
            MAX_FRAMES="$2"; shift 2;;
        --prompt-file)
            PROMPT_FILE="$2"; shift 2;;
        --no-render)
            NO_RENDER="true"; shift;;
        --skip-install)
            SKIP_INSTALL="true"; shift;;
        --skip-video)
            SKIP_VIDEO="true"; shift;;
        --provider)
            PROVIDER="$2"; PROVIDER_SET="true"; shift 2;;
        -h|--help)
            usage; exit 0;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage; exit 1;;
    esac
done

video_patterns=(-iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" -o -iname "*.avi" -o -iname "*.m4v")

is_artifacts_dir() {
    local dir="$1"
    [[ -d "$dir/extracted_frames" || -d "$dir/voxelized" || -d "$dir/gaussians" ]]
}

pick_first_video() {
    local dir="$1"
    local candidates=()
    while IFS= read -r -d '' file; do
        candidates+=("$file")
    done < <(find "$dir" -type f \( "${video_patterns[@]}" \) -print0 2>/dev/null | sort -z)

    if [[ "${#candidates[@]}" -eq 0 ]]; then
        return 1
    fi

    echo "${candidates[0]}"
}

if [[ -z "$INPUT_PATH" ]]; then
    INPUT_PATH="$REPO_DIR/inputs"
fi

if [[ ! -e "$INPUT_PATH" ]]; then
    echo "[ERROR] Input path not found: $INPUT_PATH" >&2
    exit 1
fi

source "$REPO_DIR/scripts/activate_project_env.sh"

if [[ "$SKIP_INSTALL" != "true" ]]; then
    python -m pip install -e "$REPO_DIR"
    python -m pip install -e "$REPO_DIR/extensions/aylm_mlx_vlm"
fi

ARTIFACTS_DIR=""
if [[ -d "$INPUT_PATH" ]]; then
    if is_artifacts_dir "$INPUT_PATH"; then
        ARTIFACTS_DIR="$INPUT_PATH"
    else
        SELECTED_VIDEO="$(pick_first_video "$INPUT_PATH")" || {
            echo "[ERROR] No video files found under: $INPUT_PATH" >&2
            exit 1
        }
        INPUT_PATH="$SELECTED_VIDEO"
        echo "[INFO] Auto-selected video: $INPUT_PATH"
    fi
fi

if [[ -z "$ARTIFACTS_DIR" ]]; then
    if [[ "$SKIP_VIDEO" == "true" ]]; then
        echo "[ERROR] --skip-video was set but input is not a directory." >&2
        exit 1
    fi
    VIDEO_CMD=(aylm video process -i "$INPUT_PATH" -o "$OUTPUT_VIDEO" --ego-speed "$EGO_SPEED")
    if [[ "$USE_GPU" == "true" ]]; then
        VIDEO_CMD+=(--use-gpu)
    fi
    "${VIDEO_CMD[@]}"
    ARTIFACTS_DIR="$OUTPUT_VIDEO"
fi

adjust_window_for_frames() {
    local frames_dir="$1/extracted_frames"
    [[ -d "$frames_dir" ]] || return 0

    local frame_count
    frame_count=$(find "$frames_dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l | tr -d ' ')
    if [[ -n "$frame_count" && "$frame_count" -gt 0 && "$frame_count" -lt "$WINDOW_SIZE" ]]; then
        echo "[WARN] Only ${frame_count} extracted frames found; shrinking window-size to ${frame_count} and stride to 1."
        WINDOW_SIZE="$frame_count"
        WINDOW_STRIDE="1"
    fi
}

adjust_window_for_frames "$ARTIFACTS_DIR"

RUN_CMD=()
if command -v aylm-mlx-train >/dev/null 2>&1; then
    RUN_CMD=(aylm-mlx-train -i "$ARTIFACTS_DIR" -o "$OUTPUT_NAV")
    [[ -n "$MODEL_NAME" ]] && RUN_CMD+=(--model "$MODEL_NAME")
    [[ -n "$MAX_FRAMES" ]] && RUN_CMD+=(--max-frames "$MAX_FRAMES")
    [[ -n "$PROMPT_FILE" ]] && RUN_CMD+=(--prompt-file "$PROMPT_FILE")
    [[ "$NO_RENDER" == "true" ]] && RUN_CMD+=(--no-render)
    RUN_CMD+=(--window-size "$WINDOW_SIZE" --window-stride "$WINDOW_STRIDE")
else
    if [[ "$PROVIDER" == "mlx-vlm" && "$PROVIDER_SET" != "true" ]]; then
        PROVIDER="heuristic"
    fi
    echo "[WARN] aylm-mlx-train not found. Falling back to nav-demo (provider: $PROVIDER)."
    RUN_CMD=(aylm nav-demo -i "$ARTIFACTS_DIR" -o "$OUTPUT_NAV" --provider "$PROVIDER")
    RUN_CMD+=(--window-size "$WINDOW_SIZE" --window-stride "$WINDOW_STRIDE")
    RUN_CMD+=(--ego-speed "$EGO_SPEED" --ego-heading "$EGO_HEADING")
    [[ -n "$MAX_FRAMES" ]] && RUN_CMD+=(--max-frames "$MAX_FRAMES")
    [[ "$NO_RENDER" == "true" ]] && RUN_CMD+=(--no-render)
    "${RUN_CMD[@]}"

    OUTPUT_NAV_DIR="$OUTPUT_NAV" python - << 'PY'
import json
import os
from pathlib import Path

nav_dir = Path(os.environ.get("OUTPUT_NAV_DIR", "outputs/nav_demo"))
command_log = nav_dir / "commands.jsonl"
signals_out = nav_dir / "training_signals.jsonl"
signals = []
if command_log.exists():
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
    signals_out.parent.mkdir(parents=True, exist_ok=True)
    with signals_out.open("w", encoding="utf-8") as handle:
        for signal in signals:
            handle.write(json.dumps(signal, ensure_ascii=False) + "\n")
    print(f"[INFO] Exported {len(signals)} training signals to {signals_out}")
else:
    print("[WARN] commands.jsonl not found; training_signals.jsonl not generated.")
PY
    exit 0
fi

"${RUN_CMD[@]}"
