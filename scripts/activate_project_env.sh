#!/bin/bash

set -euo pipefail

if [[ -n "${ZSH_VERSION:-}" ]]; then
    SCRIPT_PATH="${(%):-%N}"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLATFORM="$(uname -s)"
if [[ "$PLATFORM" == "Darwin" ]]; then
    DEFAULT_ENV_PATH="$REPO_DIR/.conda/aylm-macos-mps"
    SETUP_SCRIPT="$SCRIPT_DIR/setup_macos_gpu_env.sh"
elif [[ "$PLATFORM" == "Linux" ]]; then
    DEFAULT_ENV_PATH="$REPO_DIR/.conda/aylm-linux"
    SETUP_SCRIPT="$SCRIPT_DIR/setup_linux_env.sh"
else
    echo "[ERROR] Unsupported platform for this helper: $PLATFORM" >&2
    exit 1
fi
ENV_PATH="${AYLM_ENV_PATH:-$DEFAULT_ENV_PATH}"
CONDA_ROOT_DEFAULT="/opt/homebrew/Caskroom/miniforge/base/bin/conda"
CONDA_BIN="${CONDA_EXE:-$CONDA_ROOT_DEFAULT}"

resolve_conda_bin() {
    if [[ -x "$CONDA_BIN" ]]; then
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
        return 0
    fi
    return 1
}

detect_shell_name() {
    if [[ -n "${ZSH_VERSION:-}" ]]; then
        echo "zsh"
    else
        echo "bash"
    fi
}

init_conda_shell() {
    local shell_name
    shell_name="$(detect_shell_name)"
    eval "$("$CONDA_BIN" "shell.${shell_name}" hook)"
}

ensure_env_path() {
    if [[ -x "$ENV_PATH/bin/python" ]]; then
        return 0
    fi

    echo "[STEP] Project environment not found, bootstrapping..."
    "$SETUP_SCRIPT" "$DEFAULT_ENV_PATH"
    ENV_PATH="$DEFAULT_ENV_PATH"
}

is_sourced() {
    if [[ -n "${ZSH_EVAL_CONTEXT:-}" ]]; then
        [[ "$ZSH_EVAL_CONTEXT" == *:file ]]
        return
    fi

    [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

resolve_conda_bin || {
    echo "[ERROR] conda was not found. Install Miniforge/Conda first." >&2
    exit 1
}

ensure_env_path
init_conda_shell
conda activate "$ENV_PATH"
export YOLO_VERBOSE="${YOLO_VERBOSE:-false}"

if is_sourced; then
    echo "[OK] Activated project environment: $ENV_PATH"
    return 0
fi

if [[ $# -gt 0 ]]; then
    exec "$@"
fi

echo "[OK] Activated project environment in a child shell: $ENV_PATH"
echo "To keep it in your current shell, run:"
echo "  source scripts/activate_project_env.sh"
