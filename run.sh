#!/bin/bash
# AYLM v2 启动脚本
# 自动管理虚拟环境和依赖

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    cat << EOF
AYLM v2 - 3D Gaussian Splatting 点云处理工具

用法: ./run.sh [选项] [参数]

选项:
  --help, -h        显示帮助信息
  --setup           仅设置环境（创建虚拟环境并安装依赖）
  --voxelize        运行体素化处理
  --predict         运行预测流程
  --full            运行完整处理流程（默认）

示例:
  ./run.sh --setup                    # 初始化环境
  ./run.sh --voxelize input.ply       # 体素化处理
  ./run.sh --predict input.ply        # 预测处理
  ./run.sh                            # 运行完整流程
EOF
}

find_venv() {
    # 按优先级查找虚拟环境
    for venv_name in "aylm_env" "venv" ".venv"; do
        if [[ -d "$SCRIPT_DIR/$venv_name" ]]; then
            echo "$SCRIPT_DIR/$venv_name"
            return 0
        fi
    done
    return 1
}

create_venv() {
    local venv_path="$SCRIPT_DIR/aylm_env"
    log_info "创建虚拟环境: $venv_path"
    python3 -m venv "$venv_path"
    echo "$venv_path"
}

activate_venv() {
    local venv_path="$1"
    if [[ -f "$venv_path/bin/activate" ]]; then
        source "$venv_path/bin/activate"
        log_info "已激活虚拟环境: $venv_path"
    else
        log_error "无法激活虚拟环境: $venv_path"
        exit 1
    fi
}

check_and_install_deps() {
    log_info "检查依赖..."

    # 核心依赖列表
    local deps=("numpy" "scipy" "plyfile" "torch" "torchvision" "Pillow" "opencv-python" "matplotlib")
    local missing=()

    for dep in "${deps[@]}"; do
        # 处理包名映射
        local import_name="$dep"
        case "$dep" in
            "Pillow") import_name="PIL" ;;
            "opencv-python") import_name="cv2" ;;
        esac

        if ! python3 -c "import $import_name" 2>/dev/null; then
            missing+=("$dep")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_info "安装缺失依赖: ${missing[*]}"
        pip install --quiet "${missing[@]}"
    else
        log_info "所有依赖已就绪"
    fi

    # 安装项目本身（开发模式）
    if ! python3 -c "import aylm" 2>/dev/null; then
        log_info "安装 aylm 包..."
        pip install --quiet -e .
    fi
}

setup_env() {
    local venv_path

    if venv_path=$(find_venv); then
        log_info "找到虚拟环境: $venv_path"
    else
        venv_path=$(create_venv)
    fi

    activate_venv "$venv_path"
    check_and_install_deps
}

run_voxelize() {
    log_info "运行体素化处理..."
    local input="${1:-inputs/point_cloud.ply}"
    python3 -m aylm.cli voxelize "$input" "$@"
}

run_predict() {
    log_info "运行预测流程..."
    local input="${1:-inputs/point_cloud.ply}"
    python3 -m aylm.cli predict "$input" "$@"
}

run_full() {
    log_info "运行完整处理流程..."
    python3 -m aylm.cli process "$@"
}

main() {
    local action="full"
    local args=()

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                show_help
                exit 0
                ;;
            --setup)
                action="setup"
                shift
                ;;
            --voxelize)
                action="voxelize"
                shift
                ;;
            --predict)
                action="predict"
                shift
                ;;
            --full)
                action="full"
                shift
                ;;
            *)
                args+=("$1")
                shift
                ;;
        esac
    done

    # 设置环境
    setup_env

    # 执行操作
    case "$action" in
        setup)
            log_info "环境设置完成"
            ;;
        voxelize)
            run_voxelize "${args[@]}"
            ;;
        predict)
            run_predict "${args[@]}"
            ;;
        full)
            run_full "${args[@]}"
            ;;
    esac

    log_info "完成"
}

main "$@"
